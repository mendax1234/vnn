// Integer CUDA implementation of the active CNN V2 RTL's Q6.10 equations.
#include <cuda_runtime.h>
#include "../cpu/cnn_fixed.h"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using word = fixed_v2::word;
static constexpr int sizes[] = {32768,32768,8192,16384,16384,4096,8192,8192,128};
static const char* names[] = {"conv1","conv2","pool1","conv3","conv4","pool2","conv5","conv6","gap"};
static const char* classes[] = {"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t n) { check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
struct Event {
    cudaEvent_t e = nullptr;
    Event() { check(cudaEventCreate(&e)); }
    ~Event() { if (e) cudaEventDestroy(e); }
};

// 16 x 8 output tile, one output channel per block, 8 input channels per
// shared-memory tile. Every multiply is signed int16 x int16 -> int32.
// Only accumulator bits [25:10] are observable, so modulo-2^32 unsigned
// accumulation is exact even when the wider RTL accumulator wraps.
template<int IC, int OC, int H> __global__ void convolution(
    const word* input, const word* weights, const word* biases, word* output) {
    constexpr int BW = 16, BH = 8, TILE_C = 8, SW = BW + 2, SH = BH + 2;
    __shared__ word tile[TILE_C * SH * SW];
    __shared__ word kernel[TILE_C * 9];
    const int tid = threadIdx.y * BW + threadIdx.x;
    const int oc = blockIdx.z, x = blockIdx.x * BW + threadIdx.x, y = blockIdx.y * BH + threadIdx.y;
    uint32_t acc = static_cast<uint32_t>(int32_t(biases[oc]) * 1024);
    for (int base = 0; base < IC; base += TILE_C) {
        for (int i = tid; i < TILE_C * SH * SW; i += BW * BH) {
            int ic = base + i / (SH * SW);
            int iy = blockIdx.y * BH + (i % (SH * SW)) / SW - 1;
            int ix = blockIdx.x * BW + i % SW - 1;
            tile[i] = (ic < IC && iy >= 0 && iy < H && ix >= 0 && ix < H)
                ? input[(ic * H + iy) * H + ix] : 0;
        }
        if (tid < TILE_C * 9) {
            int ic = base + tid / 9;
            kernel[tid] = ic < IC ? weights[(oc * IC + ic) * 9 + tid % 9] : 0;
        }
        __syncthreads();
        if (x < H && y < H) {
            #pragma unroll
            for (int c = 0; c < TILE_C; ++c) {
                if (base + c < IC) {
                    #pragma unroll
                    for (int ky = 0; ky < 3; ++ky) {
                        #pragma unroll
                        for (int kx = 0; kx < 3; ++kx) {
                            int32_t product = int32_t(tile[(c * SH + threadIdx.y + ky) * SW + threadIdx.x + kx])
                                * int32_t(kernel[c * 9 + ky * 3 + kx]);
                            acc += static_cast<uint32_t>(product);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    if (x < H && y < H) {
        const uint32_t bits = (acc >> 10) & 65535u;
        // Wrapped signed Q6.10 value, then ReLU. No saturation here.
        output[(oc * H + y) * H + x] = (bits & 32768u) ? 0 : static_cast<word>(bits);
    }
}
__global__ void pool(const word* input, word* output, int channels, int h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = h / 2, p = oh * oh;
    if (i >= channels * p) return;
    int src = (i / p) * h * h + (i % p / oh) * 2 * h + (i % oh) * 2;
    int a = max(int(input[src]), int(input[src + 1]));
    int b = max(int(input[src + h]), int(input[src + h + 1]));
    output[i] = static_cast<word>(max(a, b));
}
__global__ void gap(const word* input, word* output) {
    int c = threadIdx.x;
    if (c >= 128) return;
    int32_t sum = 0;
    for (int i = 0; i < 64; ++i) sum += int32_t(input[c * 64 + i]);
    // All activations are nonnegative after ReLU; divide truncates downward.
    output[c] = static_cast<word>(sum / 64);
}
__global__ void linear(const word* input, const word* weights, const word* biases, int64_t* output) {
    int oc = threadIdx.x;
    if (oc >= 10) return;
    int64_t acc = int64_t(biases[oc]) * 1024;
    for (int i = 0; i < 128; ++i)
        acc += int32_t(input[i]) * int32_t(weights[oc * 128 + i]);
    constexpr uint64_t modulus = uint64_t(1) << 39;
    const uint64_t bits = static_cast<uint64_t>(acc) & (modulus - 1);
    output[oc] = (bits & (modulus >> 1)) ? -static_cast<int64_t>(modulus - bits) : static_cast<int64_t>(bits);
}
struct Model {
    Buffer<word> parameters{fixed_v2::parameter_words};
    // Keep all stages for exact CPU/GPU layer comparisons outside timing.
    Buffer<word> activations{32768 * 2 + 8192 + 16384 * 2 + 4096 + 8192 * 2 + 128};
    word* stage[9]{};
    const word* w[7]{};
    const word* b[7]{};
    explicit Model(const fixed_v2::Parameters& p) {
        check(cudaMemcpy(parameters.p, p.words.data(), p.words.size() * sizeof(word), cudaMemcpyHostToDevice));
        size_t offset = 0;
        for (int i = 0; i < 9; ++i) { stage[i] = activations.p + offset; offset += sizes[i]; }
        for (int i = 0; i < 7; ++i) {
            w[i] = parameters.p + (p.weights[i] - p.words.data());
            b[i] = parameters.p + (p.biases[i] - p.words.data());
        }
    }
    template<int IC, int OC, int H> void conv(const word* input, word* output, int l) {
        convolution<IC, OC, H><<<dim3((H + 15) / 16, (H + 7) / 8, OC), dim3(16, 8)>>>(input, w[l], b[l], output);
    }
    void infer(const word* input, int64_t* logits) {
        conv<3,32,32>(input, stage[0], 0);
        conv<32,32,32>(stage[0], stage[1], 1);
        pool<<<32,256>>>(stage[1], stage[2], 32, 32);
        conv<32,64,16>(stage[2], stage[3], 2);
        conv<64,64,16>(stage[3], stage[4], 3);
        pool<<<16,256>>>(stage[4], stage[5], 64, 16);
        conv<64,128,8>(stage[5], stage[6], 4);
        conv<128,128,8>(stage[6], stage[7], 5);
        gap<<<1,128>>>(stage[7], stage[8]);
        linear<<<1,32>>>(stage[8], w[6], b[6], logits);
        check(cudaGetLastError());
    }
};
static std::vector<uint8_t> read_bytes(const char* path, size_t count, bool exact) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open ") + path);
    std::vector<uint8_t> bytes(count);
    in.read(reinterpret_cast<char*>(bytes.data()), count);
    if (size_t(in.gcount()) != count || (exact && in.peek() != std::char_traits<char>::eof()))
        throw std::runtime_error(std::string("incorrect file size: ") + path);
    return bytes;
}
static int number(const char* text, int maximum) {
    const std::string value(text);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("counts must be positive integers");
    long long n = std::stoll(value);
    if (n < 1 || n > maximum) throw std::runtime_error("count out of range");
    return int(n);
}
template<class T> static std::vector<T> download(const T* ptr, size_t n) {
    std::vector<T> host(n);
    check(cudaMemcpy(host.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}
template<class T> static void dump(const char* directory, const char* name, const std::vector<T>& values) {
    std::ofstream out(std::string(directory) + "/" + name + ".bin", std::ios::binary);
    for (T value : values) {
        uint64_t bits = static_cast<uint64_t>(value);
        for (size_t i = 0; i < sizeof(T); ++i) out.put(static_cast<char>((bits >> (8 * i)) & 255));
    }
    if (!out) throw std::runtime_error("cannot write dump; directory must exist");
}
int main(int argc, char** argv) {
    try {
        if (argc > 8) throw std::runtime_error("too many arguments");
        int n = argc > 1 ? number(argv[1], 10000) : 100;
        const char* cifar = argc > 2 ? argv[2] : "../../cifar-10-binary/cifar-10-batches-bin/test_batch.bin";
        int warmup = argc > 3 ? number(argv[3], 100000) : 5;
        int repeats = argc > 4 ? number(argv[4], 100000) : 3;
        const char* params = argc > 5 ? argv[5] : "../../cifar-10-binary/cifar-10-batches-bin/weights_biases_v2_int16.bin";
        int verify = argc > 6 ? number(argv[6], n) : 0;
        const char* directory = argc > 7 ? argv[7] : nullptr;
        const auto bytes = read_bytes(params, fixed_v2::parameter_words * 2, true);
        fixed_v2::Parameters p;
        p.words.resize(fixed_v2::parameter_words);
        for (size_t i = 0; i < p.words.size(); ++i)
            p.words[i] = static_cast<word>(fixed_v2::signed_bits(uint16_t(bytes[2 * i]) | (uint16_t(bytes[2 * i + 1]) << 8), 16));
        p.bind();
        const auto raw = read_bytes(cifar, size_t(n) * 3073, false);
        std::vector<word> images(size_t(n) * 3072);
        const float mean[] = {.4914f,.4822f,.4465f}, deviation[] = {.2470f,.2435f,.2616f};
        for (int i = 0; i < n; ++i) {
            if (raw[size_t(i) * 3073] > 9) throw std::runtime_error("invalid CIFAR label at record " + std::to_string(i));
            for (int j = 0; j < 3072; ++j) {
                float pixel = raw[size_t(i) * 3073 + j + 1] / 255.f;
                images[size_t(i) * 3072 + j] = fixed_v2::quantize((pixel - mean[j / 1024]) / deviation[j / 1024]);
            }
        }
        cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, 0));
        printf("GPU %s (SM %d.%d)\nARITHMETIC FIXED_Q6_10\n", prop.name, prop.major, prop.minor);
        printf("Integer kernels; int16 storage, int32 convolution products/modular sums, signed 39-bit FC logits; batch size 1\nPARAMETERS %s\n", params);
        Model model(p);
        Buffer<word> device_images{images.size()};
        Buffer<int64_t> device_logits{size_t(n) * 10};
        check(cudaMemcpy(device_images.p, images.data(), images.size() * sizeof(word), cudaMemcpyHostToDevice));
        for (int i = 0; i < warmup; ++i) model.infer(device_images.p + size_t(i % n) * 3072, device_logits.p);
        check(cudaDeviceSynchronize());
        Event start, end;
        double total_ms = 0;
        for (int pass = 0; pass < repeats; ++pass) {
            const auto wall = std::chrono::steady_clock::now();
            check(cudaEventRecord(start.e));
            for (int i = 0; i < n; ++i) model.infer(device_images.p + size_t(i) * 3072, device_logits.p + size_t(i) * 10);
            check(cudaEventRecord(end.e)); check(cudaEventSynchronize(end.e));
            float ms = 0; check(cudaEventElapsedTime(&ms, start.e, end.e));
            double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall).count();
            total_ms += ms;
            printf("INFERENCE_TIME_MS %.6f\nWALL_INFERENCE_TIME_MS %.6f\nMS_PER_IMAGE %.6f\nIMAGES_PER_SECOND %.3f\n", ms, wall_ms, ms / n, n * 1000. / ms);
        }
        printf("AGGREGATE_IMAGES_PER_SECOND %.3f\n", double(n) * repeats * 1000. / total_ms);
        const auto logits = download(device_logits.p, size_t(n) * 10);
        int correct = 0, totals[10]{}, hits[10]{};
        printf("PREDICTIONS");
        for (int i = 0; i < n; ++i) {
            const auto begin = logits.begin() + size_t(i) * 10;
            int pred = int(std::max_element(begin, begin + 10) - begin), label = raw[size_t(i) * 3073];
            printf(" %d", pred); ++totals[label];
            if (pred == label) { ++correct; ++hits[label]; }
        }
        printf("\nTop-1 accuracy: %d / %d = %.2f%%\n", correct, n, 100. * correct / n);
        for (int c = 0; c < 10; ++c) printf("%-12s %4d / %4d  %.2f%%\n", classes[c], hits[c], totals[c], totals[c] ? 100. * hits[c] / totals[c] : 0.);
        fixed_v2::Workspace reference;
        for (int i = 0; i < std::max(verify, directory ? 1 : 0); ++i) {
            model.infer(device_images.p + size_t(i) * 3072, device_logits.p);
            const auto gpu_logits = download(device_logits.p, 10);
            if (i < verify) fixed_v2::infer(images.data() + size_t(i) * 3072, p, reference);
            for (int s = 0; s < 9; ++s) {
                const auto gpu = download(model.stage[s], sizes[s]);
                if (i < verify && gpu != reference.stages[s])
                    throw std::runtime_error("CPU mismatch at image " + std::to_string(i) + " layer " + names[s]);
                if (directory && i == 0) dump(directory, names[s], gpu);
            }
            if (i < verify && gpu_logits != reference.logits)
                throw std::runtime_error("CPU logit mismatch at image " + std::to_string(i));
            if (directory && i == 0) {
                dump(directory, "input", std::vector<word>(images.begin(), images.begin() + 3072));
                dump(directory, "logits", gpu_logits);
            }
        }
        if (verify) printf("VERIFY_PASS %d images: all nine stages and ten logits match CPU Q6.10 exactly\n", verify);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR: %s\nUsage: %s [images:1-10000] [cifar.bin] [warmup] [repeats] [params.bin] [verify_images:1-images] [dump_directory]\n", e.what(), argv[0]);
        return 1;
    }
}

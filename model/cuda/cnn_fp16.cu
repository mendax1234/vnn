// CNN V2: NCHW FP16 storage, FP32 accumulation, batch-one CUDA inference.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef CNN_WEIGHTS_HEADER
#include "weights_selected.h"
#endif
#include CNN_WEIGHTS_HEADER
static_assert(sizeof(W_CONV1) == 864 * sizeof(float) && sizeof(B_CONV1) == 32 * sizeof(float), "CONV1 shape/type mismatch");
static_assert(sizeof(W_CONV2) == 9216 * sizeof(float) && sizeof(B_CONV2) == 32 * sizeof(float), "CONV2 shape/type mismatch");
static_assert(sizeof(W_CONV3) == 18432 * sizeof(float) && sizeof(B_CONV3) == 64 * sizeof(float), "CONV3 shape/type mismatch");
static_assert(sizeof(W_CONV4) == 36864 * sizeof(float) && sizeof(B_CONV4) == 64 * sizeof(float), "CONV4 shape/type mismatch");
static_assert(sizeof(W_CONV5) == 73728 * sizeof(float) && sizeof(B_CONV5) == 128 * sizeof(float), "CONV5 shape/type mismatch");
static_assert(sizeof(W_CONV6) == 147456 * sizeof(float) && sizeof(B_CONV6) == 128 * sizeof(float), "CONV6 shape/type mismatch");
static_assert(sizeof(W_FC) == 1280 * sizeof(float) && sizeof(B_FC) == 10 * sizeof(float), "FC shape/type mismatch");

static void cuda_check(cudaError_t s) {
    if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
static void blas_check(cublasStatus_t s) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS error " + std::to_string(int(s)));
}
template<class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t n) { cuda_check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
struct Blas {
    cublasHandle_t h = nullptr;
    Blas() { blas_check(cublasCreate(&h)); }
    ~Blas() { if (h) cublasDestroy(h); }
};
struct Event {
    cudaEvent_t e = nullptr;
    Event() { cuda_check(cudaEventCreate(&e)); }
    ~Event() { if (e) cudaEventDestroy(e); }
};
static constexpr int channels[] = {3, 32, 32, 64, 64, 128, 128};
static const float* weights[] = {&W_CONV1[0][0][0][0], &W_CONV2[0][0][0][0],
    &W_CONV3[0][0][0][0], &W_CONV4[0][0][0][0], &W_CONV5[0][0][0][0], &W_CONV6[0][0][0][0]};
static const float* biases[] = {B_CONV1, B_CONV2, B_CONV3, B_CONV4, B_CONV5, B_CONV6};
static constexpr float mean[] = {.4914f, .4822f, .4465f};
static constexpr float deviation[] = {.2470f, .2435f, .2616f};
static const char* classes[] = {"airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"};
static float rounded(float v) { return __half2float(__float2half_rn(v)); }

// Column-major P x K matrix. Each column is one (input channel, ky, kx).
__global__ void im2col(const __half* input, __half* columns, int ic, int h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int p = h * h;
    if (i >= p * ic * 9) return;
    int spatial = i % p, k = i / p;
    int y = spatial / h + (k % 9) / 3 - 1;
    int x = spatial % h + k % 3 - 1;
    columns[i] = (y >= 0 && y < h && x >= 0 && x < h)
        ? input[(k / 9) * p + y * h + x] : __float2half_rn(0.f);
}
__global__ void bias_relu(const float* input, const __half* bias, __half* output, int n, int p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = __float2half_rn(fmaxf(0.f, input[i] + __half2float(bias[i / p])));
}
__global__ void pool(const __half* input, __half* output, int c, int h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = h / 2, p = oh * oh;
    if (i >= c * p) return;
    int src = (i / p) * h * h + (i % p / oh) * 2 * h + (i % oh) * 2;
    float v = fmaxf(__half2float(input[src]), __half2float(input[src + 1]));
    v = fmaxf(v, __half2float(input[src + h]));
    output[i] = __float2half_rn(fmaxf(v, __half2float(input[src + h + 1])));
}
__global__ void gap(const __half* input, __half* output) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= 128) return;
    float sum = 0.f;
    for (int i = 0; i < 64; ++i) sum += __half2float(input[c * 64 + i]);
    output[c] = __float2half_rn(sum / 64.f);
}
__global__ void linear(const __half* input, const __half* w, const __half* b, float* output) {
    int o = threadIdx.x;
    if (o >= 10) return;
    float sum = __half2float(b[o]);
    for (int i = 0; i < 128; ++i)
        sum = fmaf(__half2float(input[i]), __half2float(w[o * 128 + i]), sum);
    output[o] = sum;
}
static void upload(__half* dest, const float* source, size_t n) {
    std::vector<__half> host(n);
    for (size_t i = 0; i < n; ++i) host[i] = __float2half_rn(source[i]);
    cuda_check(cudaMemcpy(dest, host.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
}
struct Model {
    Blas blas;
    Buffer<__half> w{286560}, b{448}, fcw{1280}, fcb{10};
    Buffer<__half> a{32768}, z{32768}, columns{32 * 9 * 32 * 32}, features{128};
    Buffer<float> accum{32768};
    size_t offsets[6]{};
    size_t bias_offsets[6]{};
    Model() {
        // Disallow reduced-precision intermediate reductions; sums stay FP32.
        blas_check(cublasSetMathMode(blas.h, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
        size_t wi = 0, bi = 0;
        for (int l = 0; l < 6; ++l) {
            offsets[l] = wi; bias_offsets[l] = bi;
            size_t n = channels[l] * channels[l + 1] * 9;
            upload(w.p + wi, weights[l], n); upload(b.p + bi, biases[l], channels[l + 1]);
            wi += n; bi += channels[l + 1];
        }
        upload(fcw.p, &W_FC[0][0], 1280); upload(fcb.p, B_FC, 10);
    }
    void infer(const __half* image, float* logits) {
        const __half* current = image;
        __half* next = a.p;
        int h = 32;
        const float alpha = 1.f, beta = 0.f;
        for (int l = 0; l < 6; ++l) {
            int p = h * h, k = channels[l] * 9, oc = channels[l + 1];
            im2col<<<(p * k + 255) / 256, 256>>>(current, columns.p, channels[l], h);
            // P x K times K x OC -> P x OC, which is NCHW output storage.
            blas_check(cublasGemmEx(blas.h, CUBLAS_OP_N, CUBLAS_OP_N, p, oc, k,
                &alpha, columns.p, CUDA_R_16F, p, w.p + offsets[l], CUDA_R_16F, k,
                &beta, accum.p, CUDA_R_32F, p, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            bias_relu<<<(p * oc + 255) / 256, 256>>>(accum.p, b.p + bias_offsets[l], next, p * oc, p);
            current = next; next = (next == a.p) ? z.p : a.p;
            if (l == 1 || l == 3) {
                pool<<<(oc * p / 4 + 255) / 256, 256>>>(current, next, oc, h);
                h /= 2; current = next; next = (next == a.p) ? z.p : a.p;
            }
        }
        gap<<<1, 128>>>(current, features.p);
        linear<<<1, 32>>>(features.p, fcw.p, fcb.p, logits);
        cuda_check(cudaGetLastError());
    }
};

// Independent direct-convolution reference. Half mode emulates FP16 rounding
// boundaries with FP32 sums; float mode measures the FP16 change from FP32.
static std::vector<float> reference(std::vector<float> input, bool half) {
    auto q = [half](float v) { return half ? rounded(v) : v; };
    for (float& v : input) v = q(v);
    int h = 32;
    for (int l = 0; l < 6; ++l) {
        int oc = channels[l + 1], ic = channels[l], p = h * h;
        std::vector<float> output(oc * p), kernel(ic * oc * 9);
        for (size_t i = 0; i < kernel.size(); ++i) kernel[i] = q(weights[l][i]);
        for (int o = 0; o < oc; ++o) for (int y = 0; y < h; ++y) for (int x = 0; x < h; ++x) {
            float sum = 0.f;
            for (int c = 0; c < ic; ++c) for (int ky = 0; ky < 3; ++ky) for (int kx = 0; kx < 3; ++kx) {
                int iy = y + ky - 1, ix = x + kx - 1;
                if (iy >= 0 && iy < h && ix >= 0 && ix < h)
                    sum = std::fma(input[c * p + iy * h + ix], kernel[((o * ic + c) * 3 + ky) * 3 + kx], sum);
            }
            output[o * p + y * h + x] = q(std::max(0.f, sum + q(biases[l][o])));
        }
        if (l == 1 || l == 3) {
            int oh = h / 2;
            input.resize(oc * oh * oh);
            for (int c = 0; c < oc; ++c) for (int y = 0; y < oh; ++y) for (int x = 0; x < oh; ++x) {
                int i = c * p + y * 2 * h + x * 2;
                input[c * oh * oh + y * oh + x] = std::max(std::max(output[i], output[i + 1]),
                    std::max(output[i + h], output[i + h + 1]));
            }
            h = oh;
        } else input.swap(output);
    }
    std::vector<float> pooled(128), logits(10);
    for (int c = 0; c < 128; ++c) {
        float sum = 0.f;
        for (int i = 0; i < 64; ++i) sum += input[c * 64 + i];
        pooled[c] = q(sum / 64.f);
    }
    for (int o = 0; o < 10; ++o) {
        float sum = q(B_FC[o]);
        for (int i = 0; i < 128; ++i) sum = std::fma(pooled[i], q(W_FC[o][i]), sum);
        logits[o] = sum;
    }
    return logits;
}
static int argmax(const float* v) { return int(std::max_element(v, v + 10) - v); }
static int number(const char* text, int maximum) {
    std::string value(text); size_t used = 0;
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("counts must be positive integers");
    long long n = std::stoll(value, &used);
    if (used != value.size() || n < 1 || n > maximum) throw std::runtime_error("count out of range");
    return int(n);
}
int main(int argc, char** argv) {
    try {
        if (argc > 7) throw std::runtime_error("too many arguments");
        int n = argc > 1 ? number(argv[1], 10000) : 100;
        const char* path = argc > 2 ? argv[2] : "../../cifar-10-binary/cifar-10-batches-bin/test_batch.bin";
        int warmup = argc > 3 ? number(argv[3], 100000) : 5;
        int repeats = argc > 4 ? number(argv[4], 100000) : 3;
        int verify = argc > 5 ? number(argv[5], n) : 0;
        const char* dump = argc > 6 ? argv[6] : nullptr;
        cudaDeviceProp prop{}; cuda_check(cudaGetDeviceProperties(&prop, 0));
        if (prop.major < 7) throw std::runtime_error("requires compute capability >= 7.0");
        printf("GPU %s (SM %d.%d)\nFP16 weights/activations, FP32 accumulation/logits; batch size 1\n", prop.name, prop.major, prop.minor);
        printf("WEIGHTS_HEADER %s\n", CNN_WEIGHTS_HEADER);
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error(std::string("cannot open ") + path);
        std::vector<__half> images(size_t(n) * 3072);
        std::vector<int> labels(n);
        std::vector<float> original(size_t(verify) * 3072);
        unsigned char record[3073];
        for (int i = 0; i < n; ++i) {
            if (!file.read(reinterpret_cast<char*>(record), 3073) || record[0] > 9)
                throw std::runtime_error("incomplete record or invalid label at " + std::to_string(i));
            labels[i] = record[0];
            for (int j = 0; j < 3072; ++j) {
                float v = (record[j + 1] / 255.f - mean[j / 1024]) / deviation[j / 1024];
                images[size_t(i) * 3072 + j] = __float2half_rn(v);
                if (i < verify) original[size_t(i) * 3072 + j] = v;
            }
        }
        Model model;
        Buffer<__half> device_images{images.size()};
        Buffer<float> device_logits{size_t(n) * 10};
        cuda_check(cudaMemcpy(device_images.p, images.data(), images.size() * sizeof(__half), cudaMemcpyHostToDevice));
        for (int i = 0; i < warmup; ++i) model.infer(device_images.p + size_t(i % n) * 3072, device_logits.p);
        cuda_check(cudaDeviceSynchronize());
        Event start, end;
        for (int pass = 0; pass < repeats; ++pass) {
            const auto wall_start = std::chrono::steady_clock::now();
            cuda_check(cudaEventRecord(start.e));
            for (int i = 0; i < n; ++i) model.infer(device_images.p + size_t(i) * 3072, device_logits.p + size_t(i) * 10);
            cuda_check(cudaEventRecord(end.e)); cuda_check(cudaEventSynchronize(end.e));
            float ms = 0; cuda_check(cudaEventElapsedTime(&ms, start.e, end.e));
            double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_start).count();
            printf("INFERENCE_TIME_MS %.6f\nWALL_INFERENCE_TIME_MS %.6f\nMS_PER_IMAGE %.6f\nIMAGES_PER_SECOND %.3f\n", ms, wall_ms, ms / n, n * 1000. / ms);
        }
        std::vector<float> logits(size_t(n) * 10);
        cuda_check(cudaMemcpy(logits.data(), device_logits.p, logits.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (float v : logits) if (!std::isfinite(v)) throw std::runtime_error("non-finite GPU logit");
        int correct = 0, totals[10]{}, hits[10]{};
        printf("PREDICTIONS");
        for (int i = 0; i < n; ++i) {
            int pred = argmax(logits.data() + size_t(i) * 10); printf(" %d", pred);
            ++totals[labels[i]];
            if (pred == labels[i]) { ++correct; ++hits[labels[i]]; }
        }
        printf("\nTop-1 accuracy: %d / %d = %.2f%%\n", correct, n, 100. * correct / n);
        for (int c = 0; c < 10; ++c) printf("%-12s %4d / %4d  %.2f%%\n", classes[c], hits[c], totals[c], totals[c] ? 100. * hits[c] / totals[c] : 0.);
        if (dump) {
            std::ofstream out(dump, std::ios::binary);
            out.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
            if (!out) throw std::runtime_error("failed to write logits");
        }
        float max_half_error = 0.f, max_float_error = 0.f;
        int agreement = 0;
        for (int i = 0; i < verify; ++i) {
            std::vector<float> input(original.begin() + size_t(i) * 3072, original.begin() + size_t(i + 1) * 3072);
            auto half_ref = reference(input, true), float_ref = reference(input, false);
            const float* gpu = logits.data() + size_t(i) * 10;
            for (int c = 0; c < 10; ++c) {
                float error = std::abs(gpu[c] - half_ref[c]);
                max_half_error = std::max(max_half_error, error);
                max_float_error = std::max(max_float_error, std::abs(gpu[c] - float_ref[c]));
                if (!std::isfinite(half_ref[c]) || error > .01f + .003f * std::abs(half_ref[c]))
                    throw std::runtime_error("FP16 CPU reference mismatch at image " + std::to_string(i));
            }
            agreement += argmax(gpu) == argmax(float_ref.data());
        }
        if (verify) printf("VERIFY_PASS %d images; max FP16-reference logit error %.7f; max FP32 logit difference %.7f; FP32 prediction agreement %d/%d\n", verify, max_half_error, max_float_error, agreement, verify);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR: %s\nUsage: %s [images:1-10000] [cifar.bin] [warmup] [repeats] [verify_images:1-images] [logits.bin]\n", e.what(), argv[0]);
        return 1;
    }
}

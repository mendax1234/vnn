// Standalone single-threaded Q6.10 CPU benchmark; no Vitis/PYNQ dependency.
#include "cnn_fixed.h"
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

static int positive_integer(const char* text) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno || end == text || *end || value < 1 || value > INT_MAX)
        throw std::runtime_error("Expected a positive integer");
    return static_cast<int>(value);
}
static std::vector<uint8_t> read_bytes(const char* path, size_t count, bool exact) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error(std::string("Cannot open ") + path);
    std::vector<uint8_t> bytes(count);
    stream.read(reinterpret_cast<char*>(bytes.data()), count);
    if (static_cast<size_t>(stream.gcount()) != count ||
        (exact && stream.peek() != std::char_traits<char>::eof()))
        throw std::runtime_error(std::string("Incorrect file size: ") + path);
    return bytes;
}
template<class T> static void dump(const std::string& directory, const char* name,
                                   const T* data, size_t count) {
    // Portable little-endian debugging files; written outside timed passes.
    std::ofstream stream(directory + "/" + name + ".bin", std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot write dump; directory must exist");
    for (size_t i = 0; i < count; ++i) {
        const uint64_t bits = static_cast<uint64_t>(data[i]);
        for (size_t b = 0; b < sizeof(T); ++b) stream.put(static_cast<char>((bits >> (8*b)) & 255));
    }
    if (!stream) throw std::runtime_error("Failed to write dump");
}
int main(int argc, char** argv) {
    try {
        if (argc > 7) throw std::runtime_error(
            "Usage: software_model_fixed [images] [cifar] [warmup] [repeats] [params] [dump_directory]");
        const int n = argc > 1 ? positive_integer(argv[1]) : 100;
        const int warmup = argc > 3 ? positive_integer(argv[3]) : 1;
        const int repeats = argc > 4 ? positive_integer(argv[4]) : 1;
        if (n > 10000) throw std::runtime_error("Image count must be 1-10000");
        const char* cifar = argc > 2 ? argv[2] : "/home/ubuntu/test_batch.bin";
        const char* params = argc > 5 ? argv[5] : "/home/ubuntu/weights_biases_v2_int16.bin";
        const auto bytes = read_bytes(params, fixed_v2::parameter_words * 2, true);
        fixed_v2::Parameters p;
        p.words.resize(fixed_v2::parameter_words);
        for (size_t i = 0; i < p.words.size(); ++i)
            p.words[i] = static_cast<int16_t>(fixed_v2::signed_bits(
                uint16_t(bytes[2*i]) | (uint16_t(bytes[2*i+1]) << 8), 16));
        p.bind();
        const auto raw = read_bytes(cifar, size_t(n) * 3073, false);
        std::vector<fixed_v2::word> images(size_t(n) * 3072);
        const float mean[3] = {0.4914f,0.4822f,0.4465f};
        const float stddev[3] = {0.2470f,0.2435f,0.2616f};
        for (int i = 0; i < n; ++i) {
            if (raw[i * 3073] > 9) throw std::runtime_error("Invalid CIFAR label");
            for (int j = 0; j < 3072; ++j) {
                const float pixel = raw[i * 3073 + 1 + j] / 255.0f;
                images[i * 3072 + j] = fixed_v2::quantize((pixel - mean[j/1024]) / stddev[j/1024]);
            }
        }
        fixed_v2::Workspace scratch;
        std::vector<int> predictions(n);
        printf("ARITHMETIC FIXED_Q6_10\n");
        if (argc > 6) {
            fixed_v2::infer(images.data(), p, scratch);
            const char* names[9] = {"conv1","conv2","pool1","conv3","conv4","pool2","conv5","conv6","gap"};
            dump(argv[6], "input", images.data(), 3072);
            for (int i = 0; i < 9; ++i)
                dump(argv[6], names[i], scratch.stages[i].data(), scratch.stages[i].size());
            dump(argv[6], "logits", scratch.logits.data(), scratch.logits.size());
        }
        volatile int warmup_prediction = 0;
        for (int i = 0; i < warmup; ++i)
            warmup_prediction = fixed_v2::infer(images.data() + (i % n) * 3072, p, scratch);
        (void)warmup_prediction;
        for (int pass = 0; pass < repeats; ++pass) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < n; ++i)
                predictions[i] = fixed_v2::infer(images.data() + i * 3072, p, scratch);
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            printf("INFERENCE_TIME_MS %.6f\n", ms);
        }
        printf("PREDICTIONS");
        for (int prediction : predictions) printf(" %d", prediction);
        printf("\n");
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}

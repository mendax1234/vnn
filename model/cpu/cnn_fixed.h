#ifndef SOFTWARE_MODEL_FIXED_H
#define SOFTWARE_MODEL_FIXED_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace fixed_v2 {
using word = int16_t;
constexpr int frac_bits = 10;
constexpr int parameter_words = 288298;

// Decode two's complement explicitly: no out-of-range signed casts or
// signed-overflow assumptions. Unsigned shifts reproduce RTL bit selection.
inline int64_t signed_bits(uint64_t bits, unsigned width) {
    const uint64_t mask = (uint64_t(1) << width) - 1;
    bits &= mask;
    return (bits & (uint64_t(1) << (width - 1)))
        ? -static_cast<int64_t>((mask + 1) - bits)
        : static_cast<int64_t>(bits);
}
inline word conv_output(int64_t accumulator) {
    return static_cast<word>(signed_bits(
        static_cast<uint64_t>(accumulator) >> frac_bits, 16));
}
inline int64_t floor_shift(int64_t value, unsigned shift) {
    const int64_t scale = int64_t(1) << shift;
    return value >= 0 ? value / scale : -((-value + scale - 1) / scale);
}
inline word quantize(float value) {
    value = std::max(-32.0f, std::min(31.9990234375f, value));
    const float scaled = value * 1024.0f;
    const float lower = std::floor(scaled);
    const float fraction = scaled - lower;
    int result = static_cast<int>(lower);
    // NumPy round uses ties to even, unlike std::round.
    if (fraction > 0.5f || (fraction == 0.5f && result % 2 != 0)) ++result;
    return static_cast<word>(result);
}

struct Parameters {
    std::vector<word> words;
    const word* weights[7];
    const word* biases[7];
    void bind() {
        const int counts[7] = {864, 9216, 18432, 36864, 73728, 147456, 1280};
        const int channels[7] = {32, 32, 64, 64, 128, 128, 10};
        size_t offset = 0;
        // cnn_dataflow.v stream: all weights first, then all biases.
        for (int i = 0; i < 7; ++i) {
            weights[i] = words.data() + offset;
            offset += counts[i];
        }
        for (int i = 0; i < 7; ++i) {
            biases[i] = words.data() + offset;
            offset += channels[i];
        }
    }
};

inline void convolution(const word* input, std::vector<word>& output,
                        const word* weights, const word* biases,
                        int ic_count, int oc_count, int h, int w) {
    output.resize(oc_count * h * w);
    for (int oc = 0; oc < oc_count; ++oc)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                int64_t acc = int64_t(biases[oc]) * 1024;
                for (int ic = 0; ic < ic_count; ++ic)
                    for (int ky = 0; ky < 3; ++ky)
                        for (int kx = 0; kx < 3; ++kx) {
                            const int iy = y + ky - 1, ix = x + kx - 1;
                            if (iy >= 0 && iy < h && ix >= 0 && ix < w)
                                acc += int32_t(input[(ic * h + iy) * w + ix]) *
                                    int32_t(weights[((oc * ic_count + ic) * 3 + ky) * 3 + kx]);
                        }
                // Active fused Conv-ReLU RTL: select [25:10], then ReLU.
                // Its 32/40-bit accumulator wrapping cannot alter these bits.
                const word truncated = conv_output(acc);
                output[(oc * h + y) * w + x] = std::max<word>(0, truncated);
            }
}
inline void maxpool(const word* input, std::vector<word>& output,
                    int channels, int h, int w) {
    output.resize(channels * (h / 2) * (w / 2));
    for (int c = 0; c < channels; ++c)
        for (int y = 0; y < h / 2; ++y)
            for (int x = 0; x < w / 2; ++x) {
                const int pos = (c * h + 2 * y) * w + 2 * x;
                output[(c * (h / 2) + y) * (w / 2) + x] =
                    std::max(std::max(input[pos], input[pos + 1]),
                             std::max(input[pos + w], input[pos + w + 1]));
            }
}
inline void gap(const word* input, std::vector<word>& output) {
    output.resize(128);
    for (int c = 0; c < 128; ++c) {
        int32_t sum = 0; // A 22-bit accumulator holds every 64-term int16 sum.
        for (int i = 0; i < 64; ++i) sum += input[c * 64 + i];
        output[c] = static_cast<word>(floor_shift(sum, 6));
    }
}
inline int fully_connected(const word* input, const word* weights,
                           const word* biases, std::vector<int64_t>& logits) {
    logits.resize(10);
    for (int oc = 0; oc < 10; ++oc) {
        int64_t acc = int64_t(biases[oc]) * 1024;
        for (int ic = 0; ic < 128; ++ic)
            acc += int32_t(input[ic]) * int32_t(weights[oc * 128 + ic]);
        // Active fully_connected_ping_pong: full Q12.20 products,
        // signed 39-bit accumulation and bias, then argmax (no rescaling).
        logits[oc] = signed_bits(static_cast<uint64_t>(acc), 39);
    }
    return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

struct Workspace {
    std::vector<word> stages[9]; // conv1, conv2, pool1, conv3, conv4, pool2, conv5, conv6, GAP
    std::vector<int64_t> logits;
    Workspace() {
        const int sizes[9] = {32768,32768,8192,16384,16384,4096,8192,8192,128};
        for (int i = 0; i < 9; ++i) stages[i].resize(sizes[i]);
        logits.resize(10);
    }
};
inline int infer(const word* input, const Parameters& p, Workspace& s) {
    convolution(input, s.stages[0], p.weights[0], p.biases[0], 3,32,32,32);
    convolution(s.stages[0].data(), s.stages[1], p.weights[1], p.biases[1], 32,32,32,32);
    maxpool(s.stages[1].data(), s.stages[2], 32,32,32);
    convolution(s.stages[2].data(), s.stages[3], p.weights[2], p.biases[2], 32,64,16,16);
    convolution(s.stages[3].data(), s.stages[4], p.weights[3], p.biases[3], 64,64,16,16);
    maxpool(s.stages[4].data(), s.stages[5], 64,16,16);
    convolution(s.stages[5].data(), s.stages[6], p.weights[4], p.biases[4], 64,128,8,8);
    convolution(s.stages[6].data(), s.stages[7], p.weights[5], p.biases[5], 128,128,8,8);
    gap(s.stages[7].data(), s.stages[8]);
    return fully_connected(s.stages[8].data(), p.weights[6], p.biases[6], s.logits);
}
} // namespace fixed_v2
#endif

#include "cnn_accelerator.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>

static void push_raw(hls::stream<axis_word_t>& stream, ap_uint<16> raw) {
    axis_word_t word;
    word.data = raw;
    word.keep = -1;
    word.strb = -1;
    word.last = 0;
    stream.write(word);
}

static bool stream_mem(hls::stream<axis_word_t>& stream,
                       const std::string& path, int expected) {
    FILE* file = std::fopen(path.c_str(), "r");
    if (!file) {
        std::fprintf(stderr, "Cannot open parameter file: %s\n", path.c_str());
        return false;
    }
    unsigned int raw;
    int count = 0;
    while (std::fscanf(file, "%x", &raw) == 1) {
        push_raw(stream, ap_uint<16>(raw));
        ++count;
    }
    std::fclose(file);
    if (count != expected) {
        std::fprintf(stderr, "%s: expected %d words, read %d\n",
                     path.c_str(), expected, count);
        return false;
    }
    return true;
}

static bool stream_parameters(hls::stream<axis_word_t>& stream,
                              const std::string& directory) {
    struct Entry { const char* file; int words; };
    const Entry entries[] = {
        {"conv1_w.mem", 32 * 3 * 3 * 3},
        {"conv2_w.mem", 32 * 32 * 3 * 3},
        {"conv3_w.mem", 64 * 32 * 3 * 3},
        {"conv4_w.mem", 64 * 64 * 3 * 3},
        {"conv5_w.mem", 128 * 64 * 3 * 3},
        {"conv6_w.mem", 128 * 128 * 3 * 3},
        {"fc_w.mem", 10 * 128},
        {"conv1_b.mem", 32}, {"conv2_b.mem", 32},
        {"conv3_b.mem", 64}, {"conv4_b.mem", 64},
        {"conv5_b.mem", 128}, {"conv6_b.mem", 128},
        {"fc_b.mem", 10},
    };
    for (const Entry& entry : entries)
        if (!stream_mem(stream, directory + "/" + entry.file, entry.words))
            return false;
    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "Usage: %s MEM_DIR CIFAR_TEST_BIN NUM_IMAGES\n", argv[0]);
        return 2;
    }
    const std::string mem_dir = argv[1];
    const char* cifar_path = argv[2];
    const int requested = std::atoi(argv[3]);
    if (requested < 1 || requested > 10000) return 2;

    FILE* cifar = std::fopen(cifar_path, "rb");
    if (!cifar) {
        std::fprintf(stderr, "Cannot open CIFAR test batch: %s\n", cifar_path);
        return 2;
    }

    hls::stream<axis_word_t> input("input");
    hls::stream<axis_word_t> output("output");
    if (!stream_parameters(input, mem_dir)) return 2;

    const float mean[3] = {0.4914f, 0.4822f, 0.4465f};
    const float stddev[3] = {0.2470f, 0.2435f, 0.2616f};
    uint8_t record[3073];
    int labels[10000];
    int loaded = 0;

    for (; loaded < requested; ++loaded) {
        if (std::fread(record, 1, sizeof(record), cifar) != sizeof(record)) break;
        labels[loaded] = record[0];
        for (int c = 0; c < 3; ++c)
            for (int h = 0; h < 32; ++h)
                for (int w = 0; w < 32; ++w) {
                    const int offset = 1 + c * 1024 + h * 32 + w;
                    const float pixel = record[offset] / 255.0f;
                    data_t value = (pixel - mean[c]) / stddev[c];
                    push_raw(input, value.range(15, 0));
                }
    }
    std::fclose(cifar);
    if (loaded != requested) {
        std::fprintf(stderr, "Requested %d images, loaded %d\n", requested, loaded);
        return 2;
    }

    cnn_accelerator(input, output, loaded);

    int correct = 0;
    for (int i = 0; i < loaded; ++i) {
        const axis_word_t result = output.read();
        const int prediction = result.data.to_uint();
        correct += prediction == labels[i];
        std::printf("image=%d label=%d prediction=%d%s\n", i, labels[i],
                    prediction, prediction == labels[i] ? " correct" : "");
    }
    const float accuracy = 100.0f * correct / loaded;
    std::printf("Top-1 accuracy: %d/%d = %.2f%%\n", correct, loaded, accuracy);
    return 0;
}

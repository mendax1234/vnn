#include "cnn_accelerator.h"

// Optional standalone Conv1-Conv6 + ReLU vectors for RTL testbenches.
// These dumps are C-simulation-only because stdio is not synthesizable. Each
// input file follows its AXI wrapper's order: activations, weights, biases.

#define CNN_V2_DUMP_CONV1_RELU_VECTORS
#define CNN_V2_DUMP_CONV2_RELU_VECTORS
#define CNN_V2_DUMP_CONV3_RELU_VECTORS
#define CNN_V2_DUMP_CONV4_RELU_VECTORS
#define CNN_V2_DUMP_CONV5_RELU_VECTORS
#define CNN_V2_DUMP_CONV6_RELU_VECTORS
#define CNN_V2_DUMP_POST_LAYER_VECTORS

#if (defined(CNN_V2_DUMP_CONV1_RELU_VECTORS) \
  || defined(CNN_V2_DUMP_CONV2_RELU_VECTORS) \
  || defined(CNN_V2_DUMP_CONV3_RELU_VECTORS) \
  || defined(CNN_V2_DUMP_CONV4_RELU_VECTORS) \
  || defined(CNN_V2_DUMP_CONV5_RELU_VECTORS) \
  || defined(CNN_V2_DUMP_CONV6_RELU_VECTORS) \
  || defined(CNN_V2_DUMP_POST_LAYER_VECTORS)) && !defined(__SYNTHESIS__)
#include <cstdio>

static void dump_conv1_input_hex(const char* path,
                                 const data_t activation[3][32][32],
                                 const weight_t weights[32][3][3][3],
                                 const weight_t bias[32]) {
    FILE* file = std::fopen(path, "w");
    if (!file) {
        std::fprintf(stderr, "Cannot open %s for writing\n", path);
        return;
    }

    for (int c = 0; c < 3; ++c)
        for (int h = 0; h < 32; ++h)
            for (int w = 0; w < 32; ++w) {
                ap_uint<16> raw = activation[c][h][w].range(15, 0);
                std::fprintf(file, "%04X\n", raw.to_uint());
            }

    for (int oc = 0; oc < 32; ++oc)
        for (int ic = 0; ic < 3; ++ic)
            for (int kh = 0; kh < 3; ++kh)
                for (int kw = 0; kw < 3; ++kw) {
                    ap_uint<16> raw = weights[oc][ic][kh][kw].range(15, 0);
                    std::fprintf(file, "%04X\n", raw.to_uint());
                }

    for (int oc = 0; oc < 32; ++oc) {
        ap_uint<16> raw = bias[oc].range(15, 0);
        std::fprintf(file, "%04X\n", raw.to_uint());
    }

    std::fclose(file);
}

static void dump_conv1_output_hex(const char* path,
                                  const data_t activation[32][32][32]) {
    FILE* file = std::fopen(path, "w");
    if (!file) {
        std::fprintf(stderr, "Cannot open %s for writing\n", path);
        return;
    }

    for (int c = 0; c < 32; ++c)
        for (int h = 0; h < 32; ++h)
            for (int w = 0; w < 32; ++w) {
                ap_uint<16> raw = activation[c][h][w].range(15, 0);
                std::fprintf(file, "%04X\n", raw.to_uint());
            }

    std::fclose(file);
}

static void dump_conv2_input_hex(const char* path,
                                 const data_t activation[32][32][32],
                                 const weight_t weights[32][32][3][3],
                                 const weight_t bias[32]) {
    FILE* file = std::fopen(path, "w");
    if (!file) {
        std::fprintf(stderr, "Cannot open %s for writing\n", path);
        return;
    }

    for (int c = 0; c < 32; ++c)
        for (int h = 0; h < 32; ++h)
            for (int w = 0; w < 32; ++w) {
                ap_uint<16> raw = activation[c][h][w].range(15, 0);
                std::fprintf(file, "%04X\n", raw.to_uint());
            }

    for (int oc = 0; oc < 32; ++oc)
        for (int ic = 0; ic < 32; ++ic)
            for (int kh = 0; kh < 3; ++kh)
                for (int kw = 0; kw < 3; ++kw) {
                    ap_uint<16> raw = weights[oc][ic][kh][kw].range(15, 0);
                    std::fprintf(file, "%04X\n", raw.to_uint());
                }

    for (int oc = 0; oc < 32; ++oc) {
        ap_uint<16> raw = bias[oc].range(15, 0);
        std::fprintf(file, "%04X\n", raw.to_uint());
    }
    std::fclose(file);
}

static void dump_conv2_output_hex(const char* path,
                                  const data_t activation[32][32][32]) {
    FILE* file = std::fopen(path, "w");
    if (!file) {
        std::fprintf(stderr, "Cannot open %s for writing\n", path);
        return;
    }
    for (int c = 0; c < 32; ++c)
        for (int h = 0; h < 32; ++h)
            for (int w = 0; w < 32; ++w) {
                ap_uint<16> raw = activation[c][h][w].range(15, 0);
                std::fprintf(file, "%04X\n", raw.to_uint());
            }
    std::fclose(file);
}

static void dump_conv3_input_hex(const char* path,
                                 const data_t activation[32][16][16],
                                 const weight_t weights[64][32][3][3],
                                 const weight_t bias[64]) {
    FILE* file = std::fopen(path, "w");
    if (!file) { std::fprintf(stderr, "Cannot open %s for writing\n", path); return; }
    for (int c=0;c<32;++c) for (int h=0;h<16;++h) for (int w=0;w<16;++w) {
        ap_uint<16> raw=activation[c][h][w].range(15,0);
        std::fprintf(file,"%04X\n",raw.to_uint());
    }
    for (int oc=0;oc<64;++oc) for (int ic=0;ic<32;++ic)
        for (int kh=0;kh<3;++kh) for (int kw=0;kw<3;++kw) {
            ap_uint<16> raw=weights[oc][ic][kh][kw].range(15,0);
            std::fprintf(file,"%04X\n",raw.to_uint());
        }
    for (int oc=0;oc<64;++oc) {
        ap_uint<16> raw=bias[oc].range(15,0);
        std::fprintf(file,"%04X\n",raw.to_uint());
    }
    std::fclose(file);
}

static void dump_conv3_output_hex(const char* path,
                                  const data_t activation[64][16][16]) {
    FILE* file=std::fopen(path,"w");
    if (!file) { std::fprintf(stderr,"Cannot open %s for writing\n",path); return; }
    for (int c=0;c<64;++c) for (int h=0;h<16;++h) for (int w=0;w<16;++w) {
        ap_uint<16> raw=activation[c][h][w].range(15,0);
        std::fprintf(file,"%04X\n",raw.to_uint());
    }
    std::fclose(file);
}

static void dump_conv4_input_hex(const char* path,
                                 const data_t activation[64][16][16],
                                 const weight_t weights[64][64][3][3],
                                 const weight_t bias[64]) {
    FILE* file=std::fopen(path,"w");
    if (!file) { std::fprintf(stderr,"Cannot open %s for writing\n",path); return; }
    for (int c=0;c<64;++c) for (int h=0;h<16;++h) for (int w=0;w<16;++w) {
        ap_uint<16> raw=activation[c][h][w].range(15,0);
        std::fprintf(file,"%04X\n",raw.to_uint());
    }
    for (int oc=0;oc<64;++oc) for (int ic=0;ic<64;++ic)
        for (int kh=0;kh<3;++kh) for (int kw=0;kw<3;++kw) {
            ap_uint<16> raw=weights[oc][ic][kh][kw].range(15,0);
            std::fprintf(file,"%04X\n",raw.to_uint());
        }
    for (int oc=0;oc<64;++oc) {
        ap_uint<16> raw=bias[oc].range(15,0);
        std::fprintf(file,"%04X\n",raw.to_uint());
    }
    std::fclose(file);
}

static void dump_conv4_output_hex(const char* path,
                                  const data_t activation[64][16][16]) {
    FILE* file=std::fopen(path,"w");
    if (!file) { std::fprintf(stderr,"Cannot open %s for writing\n",path); return; }
    for (int c=0;c<64;++c) for (int h=0;h<16;++h) for (int w=0;w<16;++w) {
        ap_uint<16> raw=activation[c][h][w].range(15,0);
        std::fprintf(file,"%04X\n",raw.to_uint());
    }
    std::fclose(file);
}

static void dump_conv5_input_hex(const char* path,
                                 const data_t activation[64][8][8],
                                 const weight_t weights[128][64][3][3],
                                 const weight_t bias[128]) {
    FILE* file = std::fopen(path, "w");
    if (!file) {
        std::fprintf(stderr, "Cannot open %s for writing\n", path);
        return;
    }

    for (int c = 0; c < 64; ++c)
        for (int h = 0; h < 8; ++h)
            for (int w = 0; w < 8; ++w) {
                ap_uint<16> raw = activation[c][h][w].range(15, 0);
                std::fprintf(file, "%04X\n", raw.to_uint());
            }

    for (int oc = 0; oc < 128; ++oc)
        for (int ic = 0; ic < 64; ++ic)
            for (int kh = 0; kh < 3; ++kh)
                for (int kw = 0; kw < 3; ++kw) {
                    ap_uint<16> raw = weights[oc][ic][kh][kw].range(15, 0);
                    std::fprintf(file, "%04X\n", raw.to_uint());
                }

    for (int oc = 0; oc < 128; ++oc) {
        ap_uint<16> raw = bias[oc].range(15, 0);
        std::fprintf(file, "%04X\n", raw.to_uint());
    }

    std::fclose(file);
}

static void dump_conv5_output_hex(const char* path,
                                  const data_t activation[128][8][8]) {
    FILE* file = std::fopen(path, "w");
    if (!file) {
        std::fprintf(stderr, "Cannot open %s for writing\n", path);
        return;
    }

    for (int c = 0; c < 128; ++c)
        for (int h = 0; h < 8; ++h)
            for (int w = 0; w < 8; ++w) {
                ap_uint<16> raw = activation[c][h][w].range(15, 0);
                std::fprintf(file, "%04X\n", raw.to_uint());
            }

    std::fclose(file);
}

static void dump_conv6_input_hex(const char* path,
                                 const data_t activation[128][8][8],
                                 const weight_t weights[128][128][3][3],
                                 const weight_t bias[128]) {
    FILE* file = std::fopen(path, "w");
    if (!file) {
        std::fprintf(stderr, "Cannot open %s for writing\n", path);
        return;
    }

    for (int c = 0; c < 128; ++c)
        for (int h = 0; h < 8; ++h)
            for (int w = 0; w < 8; ++w) {
                ap_uint<16> raw = activation[c][h][w].range(15, 0);
                std::fprintf(file, "%04X\n", raw.to_uint());
            }

    for (int oc = 0; oc < 128; ++oc)
        for (int ic = 0; ic < 128; ++ic)
            for (int kh = 0; kh < 3; ++kh)
                for (int kw = 0; kw < 3; ++kw) {
                    ap_uint<16> raw = weights[oc][ic][kh][kw].range(15, 0);
                    std::fprintf(file, "%04X\n", raw.to_uint());
                }

    for (int oc = 0; oc < 128; ++oc) {
        ap_uint<16> raw = bias[oc].range(15, 0);
        std::fprintf(file, "%04X\n", raw.to_uint());
    }

    std::fclose(file);
}

static void dump_conv6_output_hex(const char* path,
                                  const data_t activation[128][8][8]) {
    FILE* file = std::fopen(path, "w");
    if (!file) {
        std::fprintf(stderr, "Cannot open %s for writing\n", path);
        return;
    }

    for (int c = 0; c < 128; ++c)
        for (int h = 0; h < 8; ++h)
            for (int w = 0; w < 8; ++w) {
                ap_uint<16> raw = activation[c][h][w].range(15, 0);
                std::fprintf(file, "%04X\n", raw.to_uint());
            }

    std::fclose(file);
}

static void dump_maxpool1_output_hex(const char* path,
                                     const data_t activation[32][16][16]) {
    FILE* file = std::fopen(path, "w");
    if (!file) return;

    for (int c = 0; c < 32; ++c)
        for (int h = 0; h < 16; ++h)
            for (int w = 0; w < 16; ++w) {
                ap_uint<16> raw = activation[c][h][w].range(15, 0);
                std::fprintf(file, "%04X\n", raw.to_uint());
            }

    std::fclose(file);
}

static void dump_maxpool2_output_hex(const char* path,
                                     const data_t activation[64][8][8]) {
    FILE* file = std::fopen(path, "w");
    if (!file) return;

    for (int c = 0; c < 64; ++c)
        for (int h = 0; h < 8; ++h)
            for (int w = 0; w < 8; ++w) {
                ap_uint<16> raw = activation[c][h][w].range(15, 0);
                std::fprintf(file, "%04X\n", raw.to_uint());
            }

    std::fclose(file);
}

static void dump_gap_output_hex(const char* path,
                                const data_t gap[128]) {
    FILE* file = std::fopen(path, "w");
    if (!file) return;

    for (int c = 0; c < 128; ++c) {
        ap_uint<16> raw = gap[c].range(15, 0);
        std::fprintf(file, "%04X\n", raw.to_uint());
    }

    std::fclose(file);
}

static void dump_fc_output_hex(const char* path,
                               ap_uint<16> prediction) {
    FILE* file = std::fopen(path, "w");
    if (!file) return;

    std::fprintf(file, "%04X\n", prediction.to_uint());
    std::fclose(file);
}
#endif

void cnn_accelerator(hls::stream<axis_word_t>& input,
                        hls::stream<axis_word_t>& output,
                        int num_images) {
#pragma HLS INTERFACE axis port=input
#pragma HLS INTERFACE axis port=output
#pragma HLS INTERFACE s_axilite port=num_images bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static weight_t w1[32][3][3][3];
    static weight_t w2[32][32][3][3];
    static weight_t w3[64][32][3][3];
    static weight_t w4[64][64][3][3];
    static weight_t w5[128][64][3][3];
    static weight_t w6[128][128][3][3];
    static weight_t wf[10][128];
    static weight_t b1[32], b2[32], b3[64], b4[64], b5[128], b6[128], bf[10];

#pragma HLS BIND_STORAGE variable=w1 type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=w2 type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=w3 type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=w4 type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=w5 type=ram_1p impl=uram
#pragma HLS BIND_STORAGE variable=w6 type=ram_1p impl=uram
#pragma HLS BIND_STORAGE variable=wf type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=b1 type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=b2 type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=b3 type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=b4 type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=b5 type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=b6 type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=bf type=ram_1p impl=lutram

    // Stream protocol: W1..W6, WF, B1..B6, BF, then CHW images.
    load_conv_weights<32, 3>(input, w1);
    load_conv_weights<32, 32>(input, w2);
    load_conv_weights<64, 32>(input, w3);
    load_conv_weights<64, 64>(input, w4);
    load_conv_weights<128, 64>(input, w5);
    load_conv_weights<128, 128>(input, w6);
    load_linear_weights<10, 128>(input, wf);
    load_vector<32>(input, b1);
    load_vector<32>(input, b2);
    load_vector<64>(input, b3);
    load_vector<64>(input, b4);
    load_vector<128>(input, b5);
    load_vector<128>(input, b6);
    load_vector<10>(input, bf);

    for (int image = 0; image < num_images; ++image) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=10000 avg=100
        data_t image_data[3][32][32];
        data_t x1[32][32][32], x2[32][32][32], p1[32][16][16];
        data_t x3[64][16][16], x4[64][16][16], p2[64][8][8];
        data_t x5[128][8][8], x6[128][8][8], gap[128];
        logit_t logits[10];

#pragma HLS BIND_STORAGE variable=image_data type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=p1 type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=p2 type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=x5 type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=x6 type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=gap type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=logits type=ram_1p impl=lutram

        for (int c = 0; c < 3; ++c)
            for (int h = 0; h < 32; ++h)
                for (int w = 0; w < 32; ++w) {
#pragma HLS PIPELINE II=1
                    axis_word_t word = input.read();
                    image_data[c][h][w].range(15, 0) = word.data;
                }

#if defined(CNN_V2_DUMP_CONV1_RELU_VECTORS) && !defined(__SYNTHESIS__)
        if (image == 0)
            dump_conv1_input_hex("test_input_conv1_relu.mem",
                                 image_data, w1, b1);
#endif
        conv3x3_relu<3, 32, 32, 32, 2>(image_data, x1, w1, b1);
#if defined(CNN_V2_DUMP_CONV1_RELU_VECTORS) && !defined(__SYNTHESIS__)
        if (image == 0)
            dump_conv1_output_hex("test_result_expected_conv1_relu.mem", x1);
#endif
#if defined(CNN_V2_DUMP_CONV2_RELU_VECTORS) && !defined(__SYNTHESIS__)
        if (image == 0)
            dump_conv2_input_hex("test_input_conv2_relu.mem", x1, w2, b2);
#endif
        conv3x3_relu<32, 32, 32, 32, 4>(x1, x2, w2, b2);
#if defined(CNN_V2_DUMP_CONV2_RELU_VECTORS) && !defined(__SYNTHESIS__)
        if (image == 0)
            dump_conv2_output_hex("test_result_expected_conv2_relu.mem", x2);
#endif
        maxpool2x2<32, 32, 32>(x2, p1);
#if defined(CNN_V2_DUMP_POST_LAYER_VECTORS) && !defined(__SYNTHESIS__)
        if (image == 0) {
            dump_maxpool1_output_hex("test_result_expected_maxpool1_v2.mem", p1);
            dump_maxpool1_output_hex("Project/HDL/Memory/Model-Version2-Post/test_result_expected_maxpool1_v2.mem", p1);
        }
#endif
#if defined(CNN_V2_DUMP_CONV3_RELU_VECTORS) && !defined(__SYNTHESIS__)
        if (image == 0)
            dump_conv3_input_hex("test_input_conv3_relu.mem", p1, w3, b3);
#endif
        conv3x3_relu<32, 64, 16, 16, 4>(p1, x3, w3, b3);
#if defined(CNN_V2_DUMP_CONV3_RELU_VECTORS) && !defined(__SYNTHESIS__)
        if (image == 0)
            dump_conv3_output_hex("test_result_expected_conv3_relu.mem", x3);
#endif
#if defined(CNN_V2_DUMP_CONV4_RELU_VECTORS) && !defined(__SYNTHESIS__)
        if (image == 0)
            dump_conv4_input_hex("test_input_conv4_relu.mem", x3, w4, b4);
#endif
        conv3x3_relu<64, 64, 16, 16, 8>(x3, x4, w4, b4);
#if defined(CNN_V2_DUMP_CONV4_RELU_VECTORS) && !defined(__SYNTHESIS__)
        if (image == 0)
            dump_conv4_output_hex("test_result_expected_conv4_relu.mem", x4);
#endif
        maxpool2x2<64, 16, 16>(x4, p2);
#if defined(CNN_V2_DUMP_POST_LAYER_VECTORS) && !defined(__SYNTHESIS__)
        if (image == 0) {
            dump_maxpool2_output_hex("test_result_expected_maxpool2_v2.mem", p2);
            dump_maxpool2_output_hex("Project/HDL/Memory/Model-Version2-Post/test_result_expected_maxpool2_v2.mem", p2);
        }
#endif
#if defined(CNN_V2_DUMP_CONV5_RELU_VECTORS) && !defined(__SYNTHESIS__)
        if (image == 0)
            dump_conv5_input_hex("test_input_conv5_relu.mem", p2, w5, b5);
#endif
        // PIC=8 is the HLS partial-unroll setting. The standalone RTL Conv5
        // datapath instead processes all 64 input channels in parallel.
        conv3x3_relu<64, 128, 8, 8, 8>(p2, x5, w5, b5);
#if defined(CNN_V2_DUMP_CONV5_RELU_VECTORS) && !defined(__SYNTHESIS__)
        if (image == 0)
            dump_conv5_output_hex("test_result_expected_conv5_relu.mem", x5);
#endif
#if defined(CNN_V2_DUMP_CONV6_RELU_VECTORS) && !defined(__SYNTHESIS__)
        if (image == 0)
            dump_conv6_input_hex("test_input_conv6_relu.mem", x5, w6, b6);
#endif
        // PIC=8 is the HLS partial-unroll setting. The standalone RTL Conv6
        // uses 64 lanes and processes the 128 input channels in two groups.
        conv3x3_relu<128, 128, 8, 8, 8>(x5, x6, w6, b6);
#if defined(CNN_V2_DUMP_CONV6_RELU_VECTORS) && !defined(__SYNTHESIS__)
        if (image == 0)
            dump_conv6_output_hex("test_result_expected_conv6_relu.mem", x6);
#endif
        global_average_pool<128, 8, 8>(x6, gap);
#if defined(CNN_V2_DUMP_POST_LAYER_VECTORS) && !defined(__SYNTHESIS__)
        if (image == 0) {
            dump_gap_output_hex("test_result_expected_gap_v2.mem", gap);
            dump_gap_output_hex("Project/HDL/Memory/Model-Version2-Post/test_result_expected_gap_v2.mem", gap);
        }
#endif
        linear<128, 10>(gap, logits, wf, bf);

        ap_uint<16> prediction = 0;
        logit_t maximum = logits[0];
        for (int i = 1; i < 10; ++i)
            if (logits[i] > maximum) {
                maximum = logits[i];
                prediction = i;
            }
#if defined(CNN_V2_DUMP_POST_LAYER_VECTORS) && !defined(__SYNTHESIS__)
        if (image == 0) {
            dump_fc_output_hex("test_result_expected_fc_v2.mem", prediction);
            dump_fc_output_hex("Project/HDL/Memory/Model-Version2-Post/test_result_expected_fc_v2.mem", prediction);
        }
#endif

        axis_word_t result;
        result.data = prediction;
        result.keep = -1;
        result.strb = -1;
        result.last = (image == num_images - 1);
        output.write(result);
    }
}

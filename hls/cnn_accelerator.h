#ifndef CNN_ACCELERATOR_H
#define CNN_ACCELERATOR_H

#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <hls_stream.h>

typedef ap_axiu<16, 0, 0, 0> axis_word_t;
typedef ap_fixed<16, 6> data_t;
typedef ap_fixed<16, 6> weight_t;
typedef ap_fixed<40, 18> accum_t;
typedef ap_fixed<24, 12> logit_t;

static const int CNN_V2_PARAMETER_WORDS = 288298;
static const int CNN_V2_IMAGE_WORDS = 3 * 32 * 32;

template<int OC, int IC>
void load_conv_weights(hls::stream<axis_word_t>& input,
                       weight_t weights[OC][IC][3][3]) {
    for (int oc = 0; oc < OC; ++oc)
        for (int ic = 0; ic < IC; ++ic)
            for (int kh = 0; kh < 3; ++kh)
                for (int kw = 0; kw < 3; ++kw) {
#pragma HLS PIPELINE II=1
                    axis_word_t word = input.read();
                    weights[oc][ic][kh][kw].range(15, 0) = word.data;
                }
}

template<int N>
void load_vector(hls::stream<axis_word_t>& input, weight_t values[N]) {
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        axis_word_t word = input.read();
        values[i].range(15, 0) = word.data;
    }
}

template<int OUT, int IN>
void load_linear_weights(hls::stream<axis_word_t>& input,
                         weight_t weights[OUT][IN]) {
    for (int o = 0; o < OUT; ++o)
        for (int i = 0; i < IN; ++i) {
#pragma HLS PIPELINE II=1
            axis_word_t word = input.read();
            weights[o][i].range(15, 0) = word.data;
        }
}

template<int IC, int OC, int H, int W, int PIC>
void conv3x3_relu(const data_t input[IC][H][W],
                  data_t output[OC][H][W],
                  const weight_t weights[OC][IC][3][3],
                  const weight_t bias[OC]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_RESHAPE variable=input cyclic factor=PIC dim=1
#pragma HLS ARRAY_RESHAPE variable=weights cyclic factor=PIC dim=2

    for (int oc = 0; oc < OC; ++oc)
        for (int oh = 0; oh < H; ++oh)
            for (int ow = 0; ow < W; ++ow) {
                accum_t sum = bias[oc];

                for (int ic = 0; ic < IC; ic += PIC) {
#pragma HLS PIPELINE II=1
                    accum_t partial[PIC];
#pragma HLS ARRAY_PARTITION variable=partial complete dim=1

                    for (int p = 0; p < PIC; ++p) {
#pragma HLS UNROLL
                        partial[p] = 0;
                    }

                    for (int p = 0; p < PIC; ++p) {
#pragma HLS UNROLL
                        if (ic + p < IC) {
                            for (int kh = 0; kh < 3; ++kh) {
                                for (int kw = 0; kw < 3; ++kw) {
                                    const int ih = oh + kh - 1;
                                    const int iw = ow + kw - 1;
                                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                        partial[p] += input[ic + p][ih][iw] *
                                                      weights[oc][ic + p][kh][kw];
                                }
                            }
                        }
                    }

                    for (int p = 0; p < PIC; ++p) {
#pragma HLS UNROLL
                        sum += partial[p];
                    }
                }
                output[oc][oh][ow] = sum > 0 ? data_t(sum) : data_t(0);
            }
}

template<int C, int H, int W>
void maxpool2x2(const data_t input[C][H][W],
                data_t output[C][H / 2][W / 2]) {
    for (int c = 0; c < C; ++c)
        for (int oh = 0; oh < H / 2; ++oh)
            for (int ow = 0; ow < W / 2; ++ow) {
#pragma HLS PIPELINE II=1
                const int ih = oh * 2;
                const int iw = ow * 2;
                data_t maximum = input[c][ih][iw];
                if (input[c][ih][iw + 1] > maximum) maximum = input[c][ih][iw + 1];
                if (input[c][ih + 1][iw] > maximum) maximum = input[c][ih + 1][iw];
                if (input[c][ih + 1][iw + 1] > maximum) maximum = input[c][ih + 1][iw + 1];
                output[c][oh][ow] = maximum;
            }
}

template<int C, int H, int W>
void global_average_pool(const data_t input[C][H][W], data_t output[C]) {
    for (int c = 0; c < C; ++c) {
        accum_t sum = 0;
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
#pragma HLS PIPELINE II=1
                sum += input[c][h][w];
            }
        output[c] = data_t(sum / (H * W));
    }
}

template<int IN, int OUT>
void linear(const data_t input[IN], logit_t output[OUT],
            const weight_t weights[OUT][IN], const weight_t bias[OUT]) {
    for (int o = 0; o < OUT; ++o) {
        accum_t sum = bias[o];
        for (int i = 0; i < IN; ++i) {
#pragma HLS PIPELINE II=1
            sum += accum_t(input[i]) * accum_t(weights[o][i]);
        }
        output[o] = logit_t(sum);
    }
}

void cnn_accelerator(hls::stream<axis_word_t>& input,
                        hls::stream<axis_word_t>& output,
                        int num_images);

#endif

`timescale 1ns / 1ps

module conv1_weight_agu #(
        parameter ADDR_WIDTH = 9,
        parameter OC_DIM     = 16,
        parameter K          = 3,
        parameter OUTPUT_UNROLL = 4
    )(
        input [$clog2(OC_DIM/OUTPUT_UNROLL)-1:0] oc,
        input [$clog2(K)-1:0]      k_row,
        input [$clog2(K)-1:0]      k_col,

        output [ADDR_WIDTH-1:0]    B_read_address
    );

    // Weights per output channel mapped in 1 BRAM = 3x3 = 9 words
    localparam WEIGHTS_PER_KERNEL = K * K;

    wire [ADDR_WIDTH-1:0] oc_offset     = oc * WEIGHTS_PER_KERNEL;
    wire [ADDR_WIDTH-1:0] kernel_offset = (k_row * K) + k_col;

    assign B_read_address = oc_offset + kernel_offset;

endmodule

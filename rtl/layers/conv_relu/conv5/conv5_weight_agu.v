`timescale 1ns / 1ps

// Address generator for each Conv5 weight RAM with one output lane.
module conv5_weight_agu #(
        parameter ADDR_WIDTH = 11,
        parameter OC_DIM     = 128,
        parameter OUTPUT_UNROLL = 1,
        parameter K          = 3
    ) (
        input  [$clog2(OC_DIM/OUTPUT_UNROLL)-1:0] oc,
        input  [$clog2(K)-1:0]      k_row,
        input  [$clog2(K)-1:0]      k_col,

        output [ADDR_WIDTH-1:0] B_read_address
    );

    localparam WEIGHTS_PER_KERNEL = K * K;

    wire [ADDR_WIDTH-1:0] output_channel_offset =
         oc * WEIGHTS_PER_KERNEL;
    wire [ADDR_WIDTH-1:0] kernel_offset = (k_row * K) + k_col;

    assign B_read_address = output_channel_offset + kernel_offset;

endmodule

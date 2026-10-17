`timescale 1ns / 1ps

// Conv6 weight AGU for two output lanes and two 64-channel input groups.
module conv6_weight_agu #(
        parameter ADDR_WIDTH = 11,
        parameter OC_DIM     = 128,
        parameter OUTPUT_UNROLL = 2,
        parameter K          = 3,
        parameter IC_GROUPS  = 2
    ) (
        input  [$clog2(OC_DIM/OUTPUT_UNROLL)-1:0] oc,
        input  [$clog2(IC_GROUPS)-1:0] ic_group,
        input  [$clog2(K)-1:0]         k_row,
        input  [$clog2(K)-1:0]         k_col,

        output [ADDR_WIDTH-1:0] B_read_address
    );

    localparam WEIGHTS_PER_KERNEL = K * K;

    wire [ADDR_WIDTH-1:0] channel_group = oc * IC_GROUPS + ic_group;
    wire [ADDR_WIDTH-1:0] group_offset =
         channel_group * WEIGHTS_PER_KERNEL;
    wire [ADDR_WIDTH-1:0] kernel_offset = k_row * K + k_col;

    assign B_read_address = group_offset + kernel_offset;

endmodule

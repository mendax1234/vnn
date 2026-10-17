`timescale 1ns / 1ps

module conv2_res_agu #(
        parameter ADDR_WIDTH = 13,
        parameter OC_DIM     = 32,
        parameter IN_H       = 16,
        parameter IN_W       = 16,
        parameter OUTPUT_UNROLL = 4
    )(
        input  [$clog2(OC_DIM/OUTPUT_UNROLL)-1:0] oc,
        input  [$clog2(IN_H)-1:0]     row,
        input  [$clog2(IN_W)-1:0]     col,

        output [ADDR_WIDTH-1:0]       RES_write_address_even,
        output [ADDR_WIDTH-1:0]       RES_write_address_odd,
        output [ADDR_WIDTH-1:0]       RES_write_address_0,
        output [ADDR_WIDTH-1:0]       RES_write_address_1,
        output [ADDR_WIDTH-1:0]       RES_write_address_2,
        output [ADDR_WIDTH-1:0]       RES_write_address_3
    );

    // Calculate the 2D spatial offset for the current pixel
    wire [ADDR_WIDTH-1:0] spatial_offset = (row * IN_W) + col;

    wire [ADDR_WIDTH-1:0] ch_0_offset = ((oc * OUTPUT_UNROLL) + 0) * (IN_H * IN_W);
    wire [ADDR_WIDTH-1:0] ch_1_offset = ((oc * OUTPUT_UNROLL) + 1) * (IN_H * IN_W);
    wire [ADDR_WIDTH-1:0] ch_2_offset = ((oc * OUTPUT_UNROLL) + 2) * (IN_H * IN_W);
    wire [ADDR_WIDTH-1:0] ch_3_offset = ((oc * OUTPUT_UNROLL) + 3) * (IN_H * IN_W);

    // Final target addresses
    assign RES_write_address_0 = ch_0_offset + spatial_offset;
    assign RES_write_address_1 = ch_1_offset + spatial_offset;
    assign RES_write_address_2 = ch_2_offset + spatial_offset;
    assign RES_write_address_3 = ch_3_offset + spatial_offset;

    // Backward-compatible names used by the original output-unroll-2 Conv2.
    assign RES_write_address_even = RES_write_address_0;
    assign RES_write_address_odd  = RES_write_address_1;

endmodule

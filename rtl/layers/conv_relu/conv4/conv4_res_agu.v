`timescale 1ns / 1ps

module conv4_res_agu #(
        parameter ADDR_WIDTH=14, parameter OC_DIM=64,
        parameter IN_H=16, parameter IN_W=16,
        parameter OUTPUT_UNROLL=4
    )(
        input [$clog2(OC_DIM/OUTPUT_UNROLL)-1:0] oc,
        input [$clog2(IN_H)-1:0] row,
        input [$clog2(IN_W)-1:0] col,
        output [ADDR_WIDTH-1:0] RES_write_address_0,
        output [ADDR_WIDTH-1:0] RES_write_address_1
    );
    wire [ADDR_WIDTH-1:0] spatial_offset=(row*IN_W)+col;
    assign RES_write_address_0=(((oc*OUTPUT_UNROLL)+0)*(IN_H*IN_W))+spatial_offset;
    assign RES_write_address_1=(((oc*OUTPUT_UNROLL)+1)*(IN_H*IN_W))+spatial_offset;
endmodule

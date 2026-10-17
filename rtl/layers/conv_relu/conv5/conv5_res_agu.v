`timescale 1ns / 1ps

// Conv5 channel-major result address for one output channel per cycle.
module conv5_res_agu #(
        parameter ADDR_WIDTH=13,parameter OC_DIM=128,
        parameter OUTPUT_UNROLL=1,parameter IN_H=8,parameter IN_W=8
    )(
        input[$clog2(OC_DIM/OUTPUT_UNROLL)-1:0]oc,
        input[$clog2(IN_H)-1:0]row,input[$clog2(IN_W)-1:0]col,
        output[ADDR_WIDTH-1:0]RES_write_address
    );
    wire[ADDR_WIDTH-1:0]pixel=row*IN_W+col;
    assign RES_write_address=(oc*OUTPUT_UNROLL)*IN_H*IN_W+pixel;
endmodule

`timescale 1ns / 1ps

module maxpool_agu #(
        parameter ADDR_WIDTH = 14,
        parameter C_DIM      = 16,
        parameter IN_H       = 32,
        parameter IN_W       = 32
    )(
        input  [$clog2(C_DIM)-1:0]    c,
        input  [$clog2(IN_H/2)-1:0]   oh,
        input  [$clog2(IN_W/2)-1:0]   ow,
        input                         kh, // Only needs 1 bit (0 to 1)
        input                         kw, // Only needs 1 bit (0 to 1)

        output [ADDR_WIDTH-1:0]       read_address
    );

    // Calculate actual input coordinates based on stride 2
    wire [$clog2(IN_H)-1:0] ih = (oh << 1) + kh;
    wire [$clog2(IN_W)-1:0] iw = (ow << 1) + kw;

    // Flatten 3D coordinates to 1D memory address
    assign read_address = (c * IN_H * IN_W) + (ih * IN_W) + iw;

endmodule

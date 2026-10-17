`timescale 1ns / 1ps

// Address generator for Conv5's 3x3, stride-1, padding-1 input window.
// Each of the 64 input-channel RAMs stores one 8x8 feature map and shares
// this address because all input channels are processed in parallel.
module conv5_input_agu #(
        parameter ADDR_WIDTH = 6,  // $clog2(8 * 8)
        parameter IN_H       = 8,
        parameter IN_W       = 8,
        parameter K          = 3,
        parameter PAD        = 1
    ) (
        input  [$clog2(IN_H)-1:0] row,
        input  [$clog2(IN_W)-1:0] col,
        input  [$clog2(K)-1:0]    k_row,
        input  [$clog2(K)-1:0]    k_col,

        output [ADDR_WIDTH-1:0] A_read_address,
        output                  out_of_bounds
    );

    localparam ROW_W = $clog2(IN_H) + 2;
    localparam COL_W = $clog2(IN_W) + 2;

    wire signed [ROW_W-1:0] input_row =
        $signed({1'b0, row}) + $signed({1'b0, k_row}) - PAD;
    wire signed [COL_W-1:0] input_col =
        $signed({1'b0, col}) + $signed({1'b0, k_col}) - PAD;

    assign out_of_bounds = (input_row < 0) || (input_row >= IN_H)
                        || (input_col < 0) || (input_col >= IN_W);

    wire [ADDR_WIDTH-1:0] row_offset = input_row * IN_W;
    wire [ADDR_WIDTH-1:0] col_offset = input_col;

    assign A_read_address = out_of_bounds
                          ? {ADDR_WIDTH{1'b0}}
                          : row_offset + col_offset;

endmodule

`timescale 1ns / 1ps

// Conv6 input AGU for a 64-lane implementation of 128 input channels.
// Each lane RAM stores two 8x8 maps: channel lane, then channel lane+64.
module conv6_input_agu #(
        parameter ADDR_WIDTH = 7,  // $clog2(2 * 8 * 8)
        parameter IN_H       = 8,
        parameter IN_W       = 8,
        parameter K          = 3,
        parameter PAD        = 1,
        parameter IC_GROUPS  = 2
    ) (
        input  [$clog2(IC_GROUPS)-1:0] ic_group,
        input  [$clog2(IN_H)-1:0]      row,
        input  [$clog2(IN_W)-1:0]      col,
        input  [$clog2(K)-1:0]         k_row,
        input  [$clog2(K)-1:0]         k_col,

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

    wire [ADDR_WIDTH-1:0] group_offset = ic_group * (IN_H * IN_W);
    wire [ADDR_WIDTH-1:0] row_offset = input_row * IN_W;
    wire [ADDR_WIDTH-1:0] col_offset = input_col;

    assign A_read_address = out_of_bounds
                          ? {ADDR_WIDTH{1'b0}}
                          : group_offset + row_offset + col_offset;

endmodule

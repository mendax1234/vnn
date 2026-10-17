`timescale 1ns / 1ps

module conv2_input_agu #(
        parameter ADDR_WIDTH = 8,  // $clog2(16x16)
        parameter IN_H       = 16,
        parameter IN_W       = 16,
        parameter K          = 3,
        parameter PAD        = 1
    )(
        input [$clog2(IN_H)-1:0]  row,
        input [$clog2(IN_W)-1:0]  col,
        input [$clog2(K)-1:0]     k_row,
        input [$clog2(K)-1:0]     k_col,

        output [ADDR_WIDTH-1:0]   A_read_address,
        output                    out_of_bounds
    );

    localparam ROW_W = $clog2(IN_H) + 2;
    localparam COL_W = $clog2(IN_W) + 2;

    wire signed [ROW_W-1:0] v_row = $signed({1'b0, row}) + $signed({1'b0, k_row}) - PAD;
    wire signed [COL_W-1:0] v_col = $signed({1'b0, col}) + $signed({1'b0, k_col}) - PAD;

    assign out_of_bounds = (v_row < 0) || (v_row >= IN_H) || (v_col < 0) || (v_col >= IN_W);

    wire [ADDR_WIDTH-1:0] row_offset = v_row * IN_W;
    wire [ADDR_WIDTH-1:0] col_offset = v_col;

    assign A_read_address = out_of_bounds ? {ADDR_WIDTH{1'b0}} : (row_offset + col_offset);

endmodule

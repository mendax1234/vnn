`timescale 1ns / 1ps

// Four logical 16-bit banks packed into one true-dual-port BRAM36.
//
//   Port A, lower address half: lane 0 (bits 15:0), lane 1 (bits 31:16)
//   Port B, upper address half: lane 2 (bits 15:0), lane 3 (bits 31:16)
//
// During compute both 32-bit ports read simultaneously, producing four weights
// per BRAM per cycle. The registered outputs have a three-cycle latency.
module memory_bram_tdp_packed4 #(
        parameter width = 16,
        parameter depth_bits = 8
    )(
        input clk,
        input [3:0] write_en,
        input [depth_bits-1:0] write_address,
        input [width-1:0] write_data_in,
        input read_en,
        input [depth_bits-1:0] read_address,
        output [(4*width)-1:0] read_data_out
    );

    localparam PORT_WIDTH = 2*width;
    (* ram_style = "block" *)
    reg [PORT_WIDTH-1:0] RAM [0:(2**(depth_bits+1))-1];

    wire port_a_write = |write_en[1:0];
    wire port_b_write = |write_en[3:2];
    wire [depth_bits:0] port_a_address =
        port_a_write ? {1'b0, write_address} : {1'b0, read_address};
    wire [depth_bits:0] port_b_address =
        port_b_write ? {1'b1, write_address} : {1'b1, read_address};

    reg [PORT_WIDTH-1:0] port_a_data_reg1, port_a_data_reg2, port_a_data_out;
    reg [PORT_WIDTH-1:0] port_b_data_reg1, port_b_data_reg2, port_b_data_out;

    // TDP port A: logical lanes 0 and 1.
    always @(posedge clk) begin
        if (port_a_write || read_en) begin
            if (write_en[0])
                RAM[port_a_address][width-1:0] <= write_data_in;
            if (write_en[1])
                RAM[port_a_address][2*width-1:width] <= write_data_in;
            port_a_data_reg1 <= RAM[port_a_address];
        end
    end

    always @(posedge clk) begin
        port_a_data_reg2 <= port_a_data_reg1;
        port_a_data_out <= port_a_data_reg2;
    end

    // TDP port B: logical lanes 2 and 3 in the upper address half.
    always @(posedge clk) begin
        if (port_b_write || read_en) begin
            if (write_en[2])
                RAM[port_b_address][width-1:0] <= write_data_in;
            if (write_en[3])
                RAM[port_b_address][2*width-1:width] <= write_data_in;
            port_b_data_reg1 <= RAM[port_b_address];
        end
    end

    always @(posedge clk) begin
        port_b_data_reg2 <= port_b_data_reg1;
        port_b_data_out <= port_b_data_reg2;
    end

    assign read_data_out = {port_b_data_out, port_a_data_out};

endmodule

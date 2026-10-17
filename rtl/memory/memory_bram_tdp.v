`timescale 1ns / 1ps

module memory_bram_tdp #(
        parameter width = 16,
        parameter depth_bits = 14
    )(
        input clk,

        // Port A
        input port_a_enable,
        input port_a_write_en,
        input [depth_bits-1:0] port_a_address,
        input [width-1:0] port_a_data_in,
        output reg [width-1:0] port_a_data_out,

        // Port B
        input port_b_enable,
        input port_b_write_en,
        input [depth_bits-1:0] port_b_address,
        input [width-1:0] port_b_data_in,
        output reg [width-1:0] port_b_data_out
    );

    (* ram_style = "block" *)
    reg [width-1:0] RAM [0:(2**depth_bits)-1];

    // Optional but highly recommended for Vivado simulation
    // to prevent 'X' states on unwritten addresses.
    integer i;
    initial begin
        for (i = 0; i < 2**depth_bits; i = i + 1)
            RAM[i] = 0;
    end

    // Pipeline registers for 3-cycle latency
    reg [width-1:0] port_a_data_reg1;
    reg [width-1:0] port_a_data_reg2;
    
    reg [width-1:0] port_b_data_reg1;
    reg [width-1:0] port_b_data_reg2;

    // Port A Block (Exact AMD Pattern)
    always @(posedge clk) begin
        if (port_a_enable) begin
            if (port_a_write_en) begin
                RAM[port_a_address] <= port_a_data_in;
            end
            // Read happens regardless of write (Read-First behavior)
            port_a_data_reg1 <= RAM[port_a_address];
        end
    end

    always @(posedge clk) begin
        port_a_data_reg2 <= port_a_data_reg1;
        port_a_data_out  <= port_a_data_reg2;
    end

    // Port B Block (Exact AMD Pattern)
    always @(posedge clk) begin
        if (port_b_enable) begin
            if (port_b_write_en) begin
                RAM[port_b_address] <= port_b_data_in;
            end
            // Read happens regardless of write (Read-First behavior)
            port_b_data_reg1 <= RAM[port_b_address];
        end
    end

    always @(posedge clk) begin
        port_b_data_reg2 <= port_b_data_reg1;
        port_b_data_out  <= port_b_data_reg2;
    end

endmodule

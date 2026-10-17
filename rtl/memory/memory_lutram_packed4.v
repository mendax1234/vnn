`timescale 1ns / 1ps

// Four independently writable logical banks implemented as distributed RAM.
// All banks share the compute read address and return one weight per bank each
// cycle.  Reads are asynchronous; the Conv1 core delays the shared read address
// until its three-cycle input BRAM data reaches the multiply boundary.
module memory_lutram_packed4 #(
        parameter width = 16,
        parameter depth_bits = 7
    )(
        input clk,
        input [3:0] write_en,
        input [depth_bits-1:0] write_address,
        input [width-1:0] write_data_in,
        input read_en,
        input [depth_bits-1:0] read_address,
        output [(4*width)-1:0] read_data_out
    );

    // Separate arrays retain independent write enables while allowing all four
    // weights to be read in parallel from the common convolution address.
    (* ram_style = "distributed" *)
    reg [width-1:0] RAM0 [0:(2**depth_bits)-1];
    (* ram_style = "distributed" *)
    reg [width-1:0] RAM1 [0:(2**depth_bits)-1];
    (* ram_style = "distributed" *)
    reg [width-1:0] RAM2 [0:(2**depth_bits)-1];
    (* ram_style = "distributed" *)
    reg [width-1:0] RAM3 [0:(2**depth_bits)-1];

    // Synchronous Write
    always @(posedge clk) begin
        if (write_en[0])
            RAM0[write_address] <= write_data_in;
        if (write_en[1])
            RAM1[write_address] <= write_data_in;
        if (write_en[2])
            RAM2[write_address] <= write_data_in;
        if (write_en[3])
            RAM3[write_address] <= write_data_in;
    end

    // Asynchronous Read
    assign read_data_out = read_en ? {
               RAM3[read_address], RAM2[read_address],
               RAM1[read_address], RAM0[read_address]
           } : {(4*width){1'b0}};

endmodule

`timescale 1ns / 1ps

module memory_lutram
    #(
         parameter width = 8,                   // width is the number of bits per location
         parameter depth_bits = 2               // depth is the number of locations (2^number of address bits)
     )
     (
         input clk,
         input write_en,
         input [depth_bits-1:0] write_address,
         input [width-1:0] write_data_in,

         input read_en,
         input [depth_bits-1:0] read_address,
         output [width-1:0] read_data_out       // Changed to wire for async continuous assignment
     );

    (* ram_style = "distributed" *)
    reg [width-1:0] RAM [0:(2**depth_bits)-1];

    // Synchronous Write
    always @(posedge clk) begin
        if (write_en) begin
            RAM[write_address] <= write_data_in;
        end
    end

    // Asynchronous Read
    assign read_data_out = read_en ? RAM[read_address] : {width{1'b0}};

endmodule

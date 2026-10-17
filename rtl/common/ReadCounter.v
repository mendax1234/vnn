`timescale 1ns / 1ps

module Counter #(
        parameter ADDR_WIDTH = 9, 
        parameter CNT_INIT = 0,
        parameter MAX_VAL = 255
    ) (
        input                       CLK,
        input                       RESET,
        input                       ENABLE,
        (* max_fanout = "32" *) output reg [ADDR_WIDTH-1:0] CNT
    );

    initial begin
        CNT <= CNT_INIT;
    end

    always @(posedge CLK) begin
        if (RESET) begin
            CNT <= CNT_INIT;
        end
        else if (ENABLE) begin
            if (CNT == MAX_VAL)
                CNT <= 0;
            else
                CNT <= CNT + 1;
        end
    end
endmodule
`timescale 1ns / 1ps

module ping_pong_buffer_tdp #(
        parameter DATA_WIDTH = 16,
        parameter DEPTH_BITS = 14
    )(
        input clk,
        input rst,

        // Producer Interface (Writes 2 words via Dual Ports)
        input  producer_write_en_even,
        input  [DEPTH_BITS-1:0] producer_write_addr_even,
        input  [DATA_WIDTH-1:0] producer_write_data_even,

        input  producer_write_en_odd,
        input  [DEPTH_BITS-1:0] producer_write_addr_odd,
        input  [DATA_WIDTH-1:0] producer_write_data_odd,

        input  producer_frame_done,
        output producer_ready,

        // Consumer Interface (Reads 1 word via Port A)
        input  consumer_read_en,
        input  [DEPTH_BITS-1:0] consumer_read_addr,
        output [DATA_WIDTH-1:0] consumer_read_data,

        input  consumer_frame_done,
        output consumer_ready
    );

    // Internal State Flags (Proper Handshake)
    reg bank0_full, bank1_full;
    reg write_bank_sel, read_bank_sel;

    assign producer_ready = (write_bank_sel == 1'b0) ? ~bank0_full : ~bank1_full;
    assign consumer_ready = (read_bank_sel  == 1'b0) ? bank0_full  : bank1_full;

    always @(posedge clk) begin
        if (rst) begin
            bank0_full     <= 1'b0;
            bank1_full     <= 1'b0;
            write_bank_sel <= 1'b0;
            read_bank_sel  <= 1'b0;
        end
        else begin
            if (producer_frame_done && producer_ready) begin
                if (write_bank_sel == 1'b0)
                    bank0_full <= 1'b1;
                else
                    bank1_full <= 1'b1;
                write_bank_sel <= ~write_bank_sel;
            end
            if (consumer_frame_done && consumer_ready) begin
                if (read_bank_sel == 1'b0)
                    bank0_full <= 1'b0;
                else
                    bank1_full <= 1'b0;
                read_bank_sel <= ~read_bank_sel;
            end
        end
    end

    wire [DATA_WIDTH-1:0] data_out_0_A, data_out_1_A;
    wire bank0_producer_active = (write_bank_sel == 1'b0) && ~bank0_full;
    wire bank1_producer_active = (write_bank_sel == 1'b1) && ~bank1_full;
    wire bank0_consumer_active = (read_bank_sel == 1'b0) && bank0_full;
    wire bank1_consumer_active = (read_bank_sel == 1'b1) && bank1_full;

    // Bank 0
    memory_bram_tdp #(
                        .width(DATA_WIDTH),
                        .depth_bits(DEPTH_BITS)
                    ) bank_0 (
                        .clk(clk),
                        // Port A: Producer Write Even OR Consumer Read
                        .port_a_enable   (bank0_producer_active ? producer_write_en_even :
                                          bank0_consumer_active ? consumer_read_en : 1'b0),
                        .port_a_write_en (bank0_producer_active ? producer_write_en_even : 1'b0),
                        .port_a_address  (bank0_producer_active ? producer_write_addr_even : consumer_read_addr),
                        .port_a_data_in  (producer_write_data_even),
                        .port_a_data_out (data_out_0_A),

                        // Port B: Producer Write Odd ONLY
                        .port_b_enable   (bank0_producer_active ? producer_write_en_odd : 1'b0),
                        .port_b_write_en (bank0_producer_active ? producer_write_en_odd : 1'b0),
                        .port_b_address  (producer_write_addr_odd),
                        .port_b_data_in  (producer_write_data_odd),
                        .port_b_data_out () // Unused
                    );

    // Bank 1
    memory_bram_tdp #(
                        .width(DATA_WIDTH),
                        .depth_bits(DEPTH_BITS)
                    ) bank_1 (
                        .clk(clk),
                        // Port A: Producer Write Even OR Consumer Read
                        .port_a_enable   (bank1_producer_active ? producer_write_en_even :
                                          bank1_consumer_active ? consumer_read_en : 1'b0),
                        .port_a_write_en (bank1_producer_active ? producer_write_en_even : 1'b0),
                        .port_a_address  (bank1_producer_active ? producer_write_addr_even : consumer_read_addr),
                        .port_a_data_in  (producer_write_data_even),
                        .port_a_data_out (data_out_1_A),

                        // Port B: Producer Write Odd ONLY
                        .port_b_enable   (bank1_producer_active ? producer_write_en_odd : 1'b0),
                        .port_b_write_en (bank1_producer_active ? producer_write_en_odd : 1'b0),
                        .port_b_address  (producer_write_addr_odd),
                        .port_b_data_in  (producer_write_data_odd),
                        .port_b_data_out () // Unused
                    );

    // Output Mux
    assign consumer_read_data = (read_bank_sel == 1'b1) ? data_out_1_A : data_out_0_A;

endmodule

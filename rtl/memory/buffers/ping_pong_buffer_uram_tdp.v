`timescale 1ns / 1ps

// Two physical TDP UltraRAM frame memories. Conv2 consumes both ports of the
// write frame while Pool1 reads one port of the other frame concurrently.
module ping_pong_buffer_uram_tdp #(
        parameter DATA_WIDTH = 16,
        parameter DEPTH_BITS = 15
    )(
        input clk,
        input rst,

        input producer_write_en_even,
        input [DEPTH_BITS-1:0] producer_write_addr_even,
        input [DATA_WIDTH-1:0] producer_write_data_even,
        input producer_write_en_odd,
        input [DEPTH_BITS-1:0] producer_write_addr_odd,
        input [DATA_WIDTH-1:0] producer_write_data_odd,
        input producer_frame_done,
        output producer_ready,

        input consumer_read_en,
        input [DEPTH_BITS-1:0] consumer_read_addr,
        output [DATA_WIDTH-1:0] consumer_read_data,
        input consumer_frame_done,
        output consumer_ready
    );

    reg bank0_full, bank1_full;
    reg write_bank_sel, read_bank_sel;

    assign producer_ready = write_bank_sel ? ~bank1_full : ~bank0_full;
    assign consumer_ready = read_bank_sel ? bank1_full : bank0_full;

    always @(posedge clk) begin
        if (rst) begin
            bank0_full <= 1'b0;
            bank1_full <= 1'b0;
            write_bank_sel <= 1'b0;
            read_bank_sel <= 1'b0;
        end
        else begin
            if (producer_frame_done && producer_ready) begin
                if (write_bank_sel)
                    bank1_full <= 1'b1;
                else
                    bank0_full <= 1'b1;
                write_bank_sel <= ~write_bank_sel;
            end
            if (consumer_frame_done && consumer_ready) begin
                if (read_bank_sel)
                    bank1_full <= 1'b0;
                else
                    bank0_full <= 1'b0;
                read_bank_sel <= ~read_bank_sel;
            end
        end
    end

    wire [DATA_WIDTH-1:0] bank0_read_data;
    wire [DATA_WIDTH-1:0] bank1_read_data;
    wire bank0_producer_active = (write_bank_sel == 1'b0) && ~bank0_full;
    wire bank1_producer_active = (write_bank_sel == 1'b1) && ~bank1_full;
    wire bank0_consumer_active = (read_bank_sel == 1'b0) && bank0_full;
    wire bank1_consumer_active = (read_bank_sel == 1'b1) && bank1_full;

    memory_uram_tdp #(
                        .width(DATA_WIDTH), .depth_bits(DEPTH_BITS)
                    ) bank_0 (
                        .clk(clk),
                        .port_a_enable(bank0_producer_active ? producer_write_en_even :
                                       bank0_consumer_active ? consumer_read_en : 1'b0),
                        .port_a_write_en(bank0_producer_active ? producer_write_en_even : 1'b0),
                        .port_a_address(bank0_producer_active ? producer_write_addr_even : consumer_read_addr),
                        .port_a_data_in(producer_write_data_even),
                        .port_a_data_out(bank0_read_data),
                        .port_b_enable(bank0_producer_active ? producer_write_en_odd : 1'b0),
                        .port_b_write_en(bank0_producer_active ? producer_write_en_odd : 1'b0),
                        .port_b_address(producer_write_addr_odd),
                        .port_b_data_in(producer_write_data_odd),
                        .port_b_data_out()
                    );

    memory_uram_tdp #(
                        .width(DATA_WIDTH), .depth_bits(DEPTH_BITS)
                    ) bank_1 (
                        .clk(clk),
                        .port_a_enable(bank1_producer_active ? producer_write_en_even :
                                       bank1_consumer_active ? consumer_read_en : 1'b0),
                        .port_a_write_en(bank1_producer_active ? producer_write_en_even : 1'b0),
                        .port_a_address(bank1_producer_active ? producer_write_addr_even : consumer_read_addr),
                        .port_a_data_in(producer_write_data_even),
                        .port_a_data_out(bank1_read_data),
                        .port_b_enable(bank1_producer_active ? producer_write_en_odd : 1'b0),
                        .port_b_write_en(bank1_producer_active ? producer_write_en_odd : 1'b0),
                        .port_b_address(producer_write_addr_odd),
                        .port_b_data_in(producer_write_data_odd),
                        .port_b_data_out()
                    );

    assign consumer_read_data = read_bank_sel ? bank1_read_data : bank0_read_data;

endmodule

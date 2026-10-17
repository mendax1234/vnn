`timescale 1ns / 1ps

// Two-write channel-partitioned ping-pong buffer with one TDP RAMB18 per
// channel. Ping and pong occupy the two halves of the same physical memory.
// Port A accepts one decoded producer write for each channel, while port B
// reads the other frame with a fixed three-cycle latency.
module ping_pong_buffer_array_bram_tdp_2w #(
        parameter DATA_WIDTH = 16,
        parameter DEPTH_BITS = 8,
        parameter NUM_CHANNELS = 64,
        parameter CHANNEL_BITS = $clog2(NUM_CHANNELS)
    )(
        input clk, input rst,
        input producer_write_en_0,
        input [CHANNEL_BITS-1:0] producer_write_channel_0,
        input [DEPTH_BITS-1:0] producer_write_addr_0,
        input [DATA_WIDTH-1:0] producer_write_data_0,
        input producer_write_en_1,
        input [CHANNEL_BITS-1:0] producer_write_channel_1,
        input [DEPTH_BITS-1:0] producer_write_addr_1,
        input [DATA_WIDTH-1:0] producer_write_data_1,
        input producer_frame_done, output producer_ready,
        input consumer_read_en,
        input [DEPTH_BITS-1:0] consumer_read_addr,
        output [NUM_CHANNELS*DATA_WIDTH-1:0] consumer_read_data_bus,
        input consumer_frame_done, output consumer_ready
    );

    reg bank0_full, bank1_full;
    reg write_bank_sel, read_bank_sel;
    assign producer_ready = write_bank_sel ? ~bank1_full : ~bank0_full;
    assign consumer_ready = read_bank_sel ? bank1_full : bank0_full;

    always @(posedge clk) begin
        if (rst) begin
            bank0_full <= 0;
            bank1_full <= 0;
            write_bank_sel <= 0;
            read_bank_sel <= 0;
        end
        else begin
            if (producer_frame_done && producer_ready) begin
                if (write_bank_sel)
                    bank1_full <= 1;
                else
                    bank0_full <= 1;
                write_bank_sel <= ~write_bank_sel;
            end
            if (consumer_frame_done && consumer_ready) begin
                if (read_bank_sel)
                    bank1_full <= 0;
                else
                    bank0_full <= 0;
                read_bank_sel <= ~read_bank_sel;
            end
        end
    end

    // The two Conv3 lanes always target adjacent, therefore distinct, channels.
    // synthesis translate_off
    always @(posedge clk) begin
        if (producer_ready && producer_write_en_0 && producer_write_en_1 &&
            (producer_write_channel_0 == producer_write_channel_1))
            $error("Combined-frame BRAM received two writes for one channel");
    end
    // synthesis translate_on

    wire [DEPTH_BITS:0] physical_read_addr = {
         read_bank_sel, consumer_read_addr
    };
    genvar channel;
    generate
        for (channel = 0; channel < NUM_CHANNELS; channel = channel + 1) begin : CHANNEL_MEMORIES
            wire select_0 = producer_write_en_0 &&
                            (producer_write_channel_0 == channel);
            wire select_1 = producer_write_en_1 &&
                            (producer_write_channel_1 == channel);
            wire write_en = producer_ready && (select_0 || select_1);
            wire [DEPTH_BITS-1:0] write_addr = select_0 ?
                 producer_write_addr_0 : producer_write_addr_1;
            wire [DATA_WIDTH-1:0] write_data = select_0 ?
                 producer_write_data_0 : producer_write_data_1;
            wire [DEPTH_BITS:0] physical_write_addr = {
                 write_bank_sel, write_addr
            };
            wire [DATA_WIDTH-1:0] read_data;

            memory_bram_tdp #(
                                .width(DATA_WIDTH),
                                .depth_bits(DEPTH_BITS + 1)
                            ) channel_ram (
                                .clk(clk),
                                .port_a_enable(write_en),
                                .port_a_write_en(write_en),
                                .port_a_address(physical_write_addr),
                                .port_a_data_in(write_data),
                                .port_a_data_out(),
                                .port_b_enable(consumer_read_en && consumer_ready),
                                .port_b_write_en(1'b0),
                                .port_b_address(physical_read_addr),
                                .port_b_data_in({DATA_WIDTH{1'b0}}),
                                .port_b_data_out(read_data)
                            );
            assign consumer_read_data_bus[channel*DATA_WIDTH +: DATA_WIDTH] = read_data;
        end
    endgenerate
endmodule

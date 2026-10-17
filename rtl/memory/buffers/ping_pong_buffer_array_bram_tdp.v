`timescale 1ns / 1ps

// Channel-partitioned ping-pong buffer implemented with one TDP BRAM per
// channel.  The ping/pong selector is stored as the most-significant address
// bit, so two logical frames share one physical memory.  Port A is dedicated
// to producer writes and port B is dedicated to consumer reads; consequently
// the producer and consumer can access different frames concurrently.
//
// memory_bram_tdp has a three-cycle synchronous read latency.  The consumer
// must therefore pipeline its address/control by the same three cycles.
module ping_pong_buffer_array_bram_tdp #(
        parameter DATA_WIDTH   = 16,
        parameter DEPTH_BITS   = 10,
        parameter NUM_CHANNELS = 3
    ) (
        input  wire                           clk,
        input  wire                           rst,

        input  wire [NUM_CHANNELS-1:0]        producer_write_en_bus,
        input  wire [DEPTH_BITS-1:0]          producer_write_addr,
        input  wire [DATA_WIDTH-1:0]          producer_write_data,
        input  wire                           producer_frame_done,
        output wire                           producer_ready,

        input  wire                           consumer_read_en,
        input  wire [DEPTH_BITS-1:0]          consumer_read_addr,
        output wire [NUM_CHANNELS*DATA_WIDTH-1:0] consumer_read_data_bus,
        input  wire                           consumer_frame_done,
        output wire                           consumer_ready
    );

    reg bank0_full;
    reg bank1_full;
    reg write_bank_sel;
    reg read_bank_sel;

    assign producer_ready = write_bank_sel ? ~bank1_full : ~bank0_full;
    assign consumer_ready = read_bank_sel  ?  bank1_full :  bank0_full;

    always @(posedge clk) begin
        if (rst) begin
            bank0_full     <= 1'b0;
            bank1_full     <= 1'b0;
            write_bank_sel <= 1'b0;
            read_bank_sel  <= 1'b0;
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

    wire [DEPTH_BITS:0] physical_write_addr = {
        write_bank_sel, producer_write_addr
    };
    wire [DEPTH_BITS:0] physical_read_addr = {
        read_bank_sel, consumer_read_addr
    };

    genvar channel;
    generate
        for (channel = 0; channel < NUM_CHANNELS; channel = channel + 1) begin : CHANNEL_MEMORIES
            wire [DATA_WIDTH-1:0] read_data;

            memory_bram_tdp #(
                .width      (DATA_WIDTH),
                .depth_bits (DEPTH_BITS + 1)
            ) channel_ram (
                .clk             (clk),

                .port_a_enable   (producer_write_en_bus[channel] && producer_ready),
                .port_a_write_en (producer_write_en_bus[channel] && producer_ready),
                .port_a_address  (physical_write_addr),
                .port_a_data_in  (producer_write_data),
                .port_a_data_out (),

                .port_b_enable   (consumer_read_en && consumer_ready),
                .port_b_write_en (1'b0),
                .port_b_address  (physical_read_addr),
                .port_b_data_in  ({DATA_WIDTH{1'b0}}),
                .port_b_data_out (read_data)
            );

            assign consumer_read_data_bus[channel*DATA_WIDTH +: DATA_WIDTH] = read_data;
        end
    endgenerate

endmodule

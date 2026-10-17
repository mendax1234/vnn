`timescale 1ns / 1ps

// Channel-parallel ping-pong buffer with eight channels per UltraRAM.
// Ping and pong frames use separate devices because reading all eight packed
// channels occupies both UltraRAM ports.  Conv1 may write two channels while
// Conv2 reads the other frame concurrently.
module ping_pong_buffer_array_uram_packed8_2w #(
        parameter DATA_WIDTH = 16,
        parameter DEPTH_BITS = 10,
        parameter NUM_CHANNELS = 32,
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

    localparam NUM_GROUPS = NUM_CHANNELS / 8;
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

    wire [NUM_CHANNELS*DATA_WIDTH-1:0] read_data_0_bus;
    wire [NUM_CHANNELS*DATA_WIDTH-1:0] read_data_1_bus;
    genvar group;
    generate
        for (group = 0; group<NUM_GROUPS; group = group + 1) begin : GROUPS
            wire select_0 = (producer_write_channel_0[CHANNEL_BITS-1:3] == group);
            wire select_1 = (producer_write_channel_1[CHANNEL_BITS-1:3] == group);
            wire frame0_write = !write_bank_sel && producer_ready;
            wire frame1_write =  write_bank_sel && producer_ready;
            wire frame0_read = !read_bank_sel && consumer_ready;
            wire frame1_read =  read_bank_sel && consumer_ready;
            wire [8*DATA_WIDTH-1:0] group_data_0;
            wire [8*DATA_WIDTH-1:0] group_data_1;

            memory_uram_packed8_frame_2w #(
                                             .width(DATA_WIDTH), .depth_bits(DEPTH_BITS)
                                         ) frame0 (
                                             .clk(clk),
                                             .write_en_0(frame0_write && producer_write_en_0 && select_0),
                                             .write_channel_0(producer_write_channel_0[2:0]),
                                             .write_address_0(producer_write_addr_0),
                                             .write_data_0(producer_write_data_0),
                                             .write_en_1(frame0_write && producer_write_en_1 && select_1),
                                             .write_channel_1(producer_write_channel_1[2:0]),
                                             .write_address_1(producer_write_addr_1),
                                             .write_data_1(producer_write_data_1),
                                             .read_en(frame0_read && consumer_read_en),
                                             .read_address(consumer_read_addr),
                                             .read_data_out(group_data_0)
                                         );
            memory_uram_packed8_frame_2w #(
                                             .width(DATA_WIDTH), .depth_bits(DEPTH_BITS)
                                         ) frame1 (
                                             .clk(clk),
                                             .write_en_0(frame1_write && producer_write_en_0 && select_0),
                                             .write_channel_0(producer_write_channel_0[2:0]),
                                             .write_address_0(producer_write_addr_0),
                                             .write_data_0(producer_write_data_0),
                                             .write_en_1(frame1_write && producer_write_en_1 && select_1),
                                             .write_channel_1(producer_write_channel_1[2:0]),
                                             .write_address_1(producer_write_addr_1),
                                             .write_data_1(producer_write_data_1),
                                             .read_en(frame1_read && consumer_read_en),
                                             .read_address(consumer_read_addr),
                                             .read_data_out(group_data_1)
                                         );
            assign read_data_0_bus[group*8*DATA_WIDTH +: 8*DATA_WIDTH] = group_data_0;
            assign read_data_1_bus[group*8*DATA_WIDTH +: 8*DATA_WIDTH] = group_data_1;
        end
    endgenerate

    assign consumer_read_data_bus = read_bank_sel ? read_data_1_bus : read_data_0_bus;
endmodule

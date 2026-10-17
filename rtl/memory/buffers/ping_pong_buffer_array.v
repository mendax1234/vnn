`timescale 1ns / 1ps

module ping_pong_buffer_array #(
        parameter DATA_WIDTH = 16,
        parameter DEPTH_BITS = 8,
        parameter NUM_CHANNELS = 16,
        parameter MEM_TYPE = "LUTRAM" // Accepts "BRAM", or "LUTRAM" (no URAM)
    )(
        input wire clk,
        input wire rst,

        // Producer Interface
        input  wire [NUM_CHANNELS-1:0]               producer_write_en_bus,
        input  wire [DEPTH_BITS-1:0]                 producer_write_addr,
        input  wire [(NUM_CHANNELS*DATA_WIDTH)-1:0]  producer_write_data_bus,
        input  wire                                  producer_frame_done,
        output wire                                  producer_ready,

        // Consumer Interface
        input  wire                                  consumer_read_en,
        input  wire [DEPTH_BITS-1:0]                 consumer_read_addr,
        output reg  [(NUM_CHANNELS*DATA_WIDTH)-1:0]  consumer_read_data_bus,
        input  wire                                  consumer_frame_done,
        output wire                                  consumer_ready
    );

    // Internal State Flags
    reg bank0_full, bank1_full;
    reg write_bank_sel, read_bank_sel;

    // Handshake Control Logic
    assign producer_ready = (write_bank_sel == 1'b0) ? ~bank0_full : ~bank1_full;
    assign consumer_ready = (read_bank_sel  == 1'b0) ? bank0_full  : bank1_full;

    always @(posedge clk) begin
        if (rst) begin
            bank0_full <= 0;
            bank1_full <= 0;
            write_bank_sel <= 0;
            read_bank_sel <= 0;
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

    // Generate the array of memory banks
    wire [(NUM_CHANNELS*DATA_WIDTH)-1:0] read_data_0_bus;
    wire [(NUM_CHANNELS*DATA_WIDTH)-1:0] read_data_1_bus;

    genvar i;
    generate
        for (i = 0; i < NUM_CHANNELS; i = i + 1) begin : channel_gen
            wire we_0 = producer_write_en_bus[i] && producer_ready && (write_bank_sel == 1'b0);
            wire we_1 = producer_write_en_bus[i] && producer_ready && (write_bank_sel == 1'b1);

            wire [DATA_WIDTH-1:0] w_data = producer_write_data_bus[i*DATA_WIDTH +: DATA_WIDTH];
            wire [DATA_WIDTH-1:0] r_data_0;
            wire [DATA_WIDTH-1:0] r_data_1;

            if (MEM_TYPE == "LUTRAM") begin : gen_lut
                memory_lutram #(
                                  .width      (DATA_WIDTH),
                                  .depth_bits (DEPTH_BITS)
                              ) bank0 (
                                  .clk            (clk),
                                  .write_en       (we_0),
                                  .write_address  (producer_write_addr),
                                  .write_data_in  (w_data),
                                  .read_en        (consumer_read_en),
                                  .read_address   (consumer_read_addr),
                                  .read_data_out  (r_data_0)
                              );
                memory_lutram #(
                                  .width      (DATA_WIDTH),
                                  .depth_bits (DEPTH_BITS)
                              ) bank1 (
                                  .clk            (clk),
                                  .write_en       (we_1),
                                  .write_address  (producer_write_addr),
                                  .write_data_in  (w_data),
                                  .read_en        (consumer_read_en),
                                  .read_address   (consumer_read_addr),
                                  .read_data_out  (r_data_1)
                              );
            end
            else begin : gen_bram // Defaults to BRAM if not LUTRAM
                memory_RAM #(
                               .width      (DATA_WIDTH),
                               .depth_bits (DEPTH_BITS)
                           ) bank0 (
                               .clk            (clk),
                               .write_en       (we_0),
                               .write_address  (producer_write_addr),
                               .write_data_in  (w_data),
                               .read_en        (consumer_read_en),
                               .read_address   (consumer_read_addr),
                               .read_data_out  (r_data_0)
                           );
                memory_RAM #(
                               .width      (DATA_WIDTH),
                               .depth_bits (DEPTH_BITS)
                           ) bank1 (
                               .clk            (clk),
                               .write_en       (we_1),
                               .write_address  (producer_write_addr),
                               .write_data_in  (w_data),
                               .read_en        (consumer_read_en),
                               .read_address   (consumer_read_addr),
                               .read_data_out  (r_data_1)
                           );
            end

            assign read_data_0_bus[i*DATA_WIDTH +: DATA_WIDTH] = r_data_0;
            assign read_data_1_bus[i*DATA_WIDTH +: DATA_WIDTH] = r_data_1;
        end
    endgenerate

    // Output Mux
    always @(*) begin
        if (read_bank_sel == 1'b0)
            consumer_read_data_bus = read_data_0_bus;
        else
            consumer_read_data_bus = read_data_1_bus;
    end

endmodule

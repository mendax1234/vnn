`timescale 1ns / 1ps

module ping_pong_buffer #(
        parameter DATA_WIDTH = 16,
        parameter DEPTH_BITS = 10,
        parameter MEM_TYPE   = "BRAM" // Accepts "BRAM", "URAM", or "LUTRAM"
    )(
        input wire clk,
        input wire rst,

        // Producer Interface
        input  wire                  producer_write_en,
        input  wire [DEPTH_BITS-1:0] producer_write_addr,
        input  wire [DATA_WIDTH-1:0] producer_write_data,
        input  wire                  producer_frame_done,
        output wire                  producer_ready,

        // Consumer Interface
        input  wire                  consumer_read_en,
        input  wire [DEPTH_BITS-1:0] consumer_read_addr,
        output reg  [DATA_WIDTH-1:0] consumer_read_data,
        input  wire                  consumer_frame_done,
        output wire                  consumer_ready
    );

    // Internal State Flags
    reg bank0_full, bank1_full;
    reg write_bank_sel, read_bank_sel;

    // Handshake Control Logic
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

    // Memory Instantiations
    wire we_0 = producer_write_en && producer_ready && (write_bank_sel == 1'b0);
    wire we_1 = producer_write_en && producer_ready && (write_bank_sel == 1'b1);

    wire [DATA_WIDTH-1:0] read_data_0_w;
    wire [DATA_WIDTH-1:0] read_data_1_w;

    generate
        if (MEM_TYPE == "URAM") begin : gen_uram
            memory_URAM #(
                            .width      (DATA_WIDTH),
                            .depth_bits (DEPTH_BITS)
                        ) bank0 (
                            .clk            (clk),
                            .write_en       (we_0),
                            .write_address  (producer_write_addr),
                            .write_data_in  (producer_write_data),
                            .read_en        (consumer_read_en),
                            .read_address   (consumer_read_addr),
                            .read_data_out  (read_data_0_w)
                        );
            memory_URAM #(
                            .width      (DATA_WIDTH),
                            .depth_bits (DEPTH_BITS)
                        ) bank1 (
                            .clk            (clk),
                            .write_en       (we_1),
                            .write_address  (producer_write_addr),
                            .write_data_in  (producer_write_data),
                            .read_en        (consumer_read_en),
                            .read_address   (consumer_read_addr),
                            .read_data_out  (read_data_1_w)
                        );
        end
        else if (MEM_TYPE == "LUTRAM") begin : gen_lutram
            memory_lutram #(
                              .width      (DATA_WIDTH),
                              .depth_bits (DEPTH_BITS)
                          ) bank0 (
                              .clk            (clk),
                              .write_en       (we_0),
                              .write_address  (producer_write_addr),
                              .write_data_in  (producer_write_data),
                              .read_en        (consumer_read_en),
                              .read_address   (consumer_read_addr),
                              .read_data_out  (read_data_0_w)
                          );
            memory_lutram #(
                              .width      (DATA_WIDTH),
                              .depth_bits (DEPTH_BITS)
                          ) bank1 (
                              .clk            (clk),
                              .write_en       (we_1),
                              .write_address  (producer_write_addr),
                              .write_data_in  (producer_write_data),
                              .read_en        (consumer_read_en),
                              .read_address   (consumer_read_addr),
                              .read_data_out  (read_data_1_w)
                          );
        end
        else begin : gen_bram
            memory_RAM #(
                           .width      (DATA_WIDTH),
                           .depth_bits (DEPTH_BITS)
                       ) bank0 (
                           .clk            (clk),
                           .write_en       (we_0),
                           .write_address  (producer_write_addr),
                           .write_data_in  (producer_write_data),
                           .read_en        (consumer_read_en),
                           .read_address   (consumer_read_addr),
                           .read_data_out  (read_data_0_w)
                       );
            memory_RAM #(
                           .width      (DATA_WIDTH),
                           .depth_bits (DEPTH_BITS)
                       ) bank1 (
                           .clk            (clk),
                           .write_en       (we_1),
                           .write_address  (producer_write_addr),
                           .write_data_in  (producer_write_data),
                           .read_en        (consumer_read_en),
                           .read_address   (consumer_read_addr),
                           .read_data_out  (read_data_1_w)
                       );
        end
    endgenerate

    // Output Mux
    always @(*) begin
        if (read_bank_sel == 1'b0)
            consumer_read_data = read_data_0_w;
        else
            consumer_read_data = read_data_1_w;
    end

endmodule

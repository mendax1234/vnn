`timescale 1ns / 1ps

// Eight 16-bit channel memories packed into one UltraRAM frame memory.
// Channels 0-3 use port A in the lower address half and channels 4-7 use
// port B in the upper address half.  Two values at one spatial address can be
// written in a cycle using UltraRAM's independent 9-bit byte write enables.
module memory_uram_packed8_frame_2w #(
        parameter width = 16,
        parameter depth_bits = 10
    )(
        input clk,
        input write_en_0,
        input [2:0] write_channel_0,
        input [depth_bits-1:0] write_address_0,
        input [width-1:0] write_data_0,
        input write_en_1,
        input [2:0] write_channel_1,
        input [depth_bits-1:0] write_address_1,
        input [width-1:0] write_data_1,
        input read_en,
        input [depth_bits-1:0] read_address,
        output [(8*width)-1:0] read_data_out
    );

    localparam SLOT_WIDTH = 18;
    localparam PORT_WIDTH = 72;
    localparam ADDR_WIDTH = depth_bits + 1;
    localparam MEMORY_SIZE = PORT_WIDTH * (2**ADDR_WIDTH);

    wire write_0_a = write_en_0 && !write_channel_0[2];
    wire write_1_a = write_en_1 && !write_channel_1[2];
    wire write_0_b = write_en_0 &&  write_channel_0[2];
    wire write_1_b = write_en_1 &&  write_channel_1[2];
    wire port_a_write = write_0_a || write_1_a;
    wire port_b_write = write_0_b || write_1_b;

    wire [3:0] lane_write_en_a;
    wire [3:0] lane_write_en_b;
    wire [width-1:0] lane_write_data_a [0:3];
    wire [width-1:0] lane_write_data_b [0:3];
    genvar lane;
    generate
        for (lane = 0; lane<4; lane = lane + 1) begin : WRITE_LANES
            assign lane_write_en_a[lane] =
                   (write_0_a && (write_channel_0[1:0] == lane)) ||
                   (write_1_a && (write_channel_1[1:0] == lane));
            assign lane_write_en_b[lane] =
                   (write_0_b && (write_channel_0[1:0] == lane)) ||
                   (write_1_b && (write_channel_1[1:0] == lane));
            assign lane_write_data_a[lane] =
                   (write_0_a && (write_channel_0[1:0] == lane)) ? write_data_0 : write_data_1;
            assign lane_write_data_b[lane] =
                   (write_0_b && (write_channel_0[1:0] == lane)) ? write_data_0 : write_data_1;
        end
    endgenerate

    wire [7:0] port_a_write_enable = {
             {2{lane_write_en_a[3]}}, {2{lane_write_en_a[2]}},
             {2{lane_write_en_a[1]}}, {2{lane_write_en_a[0]}}
         };
    wire [7:0] port_b_write_enable = {
             {2{lane_write_en_b[3]}}, {2{lane_write_en_b[2]}},
             {2{lane_write_en_b[1]}}, {2{lane_write_en_b[0]}}
         };
    wire [PORT_WIDTH-1:0] port_a_write_data = {
             2'b0, lane_write_data_a[3], 2'b0, lane_write_data_a[2],
             2'b0, lane_write_data_a[1], 2'b0, lane_write_data_a[0]
         };
    wire [PORT_WIDTH-1:0] port_b_write_data = {
             2'b0, lane_write_data_b[3], 2'b0, lane_write_data_b[2],
             2'b0, lane_write_data_b[1], 2'b0, lane_write_data_b[0]
         };
    wire [ADDR_WIDTH-1:0] port_a_address = port_a_write ?
         {1'b0, write_0_a ? write_address_0 : write_address_1} : {1'b0, read_address};
    wire [ADDR_WIDTH-1:0] port_b_address = port_b_write ?
         {1'b1, write_0_b ? write_address_0 : write_address_1} : {1'b1, read_address};
    wire [PORT_WIDTH-1:0] port_a_data_out;
    wire [PORT_WIDTH-1:0] port_b_data_out;

    // Both Conv1 results in a pair always have the same spatial address.
    // synthesis translate_off
    always @(posedge clk) begin
        if (write_0_a && write_1_a && (write_address_0 != write_address_1))
            $error("Packed UltraRAM port A received different write addresses");
        if (write_0_b && write_1_b && (write_address_0 != write_address_1))
            $error("Packed UltraRAM port B received different write addresses");
    end
    // synthesis translate_on

    xpm_memory_tdpram #(
                          .ADDR_WIDTH_A(ADDR_WIDTH), .ADDR_WIDTH_B(ADDR_WIDTH),
                          .AUTO_SLEEP_TIME(0), .BYTE_WRITE_WIDTH_A(9), .BYTE_WRITE_WIDTH_B(9),
                          .CASCADE_HEIGHT(0), .CLOCKING_MODE("common_clock"), .ECC_MODE("no_ecc"),
                          .MEMORY_INIT_FILE("none"), .MEMORY_INIT_PARAM("0"),
                          .MEMORY_OPTIMIZATION("true"), .MEMORY_PRIMITIVE("ultra"),
                          .MEMORY_SIZE(MEMORY_SIZE), .MESSAGE_CONTROL(0),
                          .READ_DATA_WIDTH_A(PORT_WIDTH), .READ_DATA_WIDTH_B(PORT_WIDTH),
                          .READ_LATENCY_A(3), .READ_LATENCY_B(3),
                          .READ_RESET_VALUE_A("0"), .READ_RESET_VALUE_B("0"),
                          .RST_MODE_A("SYNC"), .RST_MODE_B("SYNC"), .SIM_ASSERT_CHK(0),
                          .USE_EMBEDDED_CONSTRAINT(0), .USE_MEM_INIT(0), .WAKEUP_TIME("disable_sleep"),
                          .WRITE_DATA_WIDTH_A(PORT_WIDTH), .WRITE_DATA_WIDTH_B(PORT_WIDTH),
                          .WRITE_MODE_A("no_change"), .WRITE_MODE_B("no_change")
                      ) uram (
                          .clka(clk), .clkb(clk),
                          .ena(port_a_write || read_en), .enb(port_b_write || read_en),
                          .wea(port_a_write_enable), .web(port_b_write_enable),
                          .addra(port_a_address), .addrb(port_b_address),
                          .dina(port_a_write_data), .dinb(port_b_write_data),
                          .douta(port_a_data_out), .doutb(port_b_data_out),
                          .regcea(1'b1), .regceb(1'b1), .rsta(1'b0), .rstb(1'b0), .sleep(1'b0),
                          .injectdbiterra(1'b0), .injectdbiterrb(1'b0),
                          .injectsbiterra(1'b0), .injectsbiterrb(1'b0),
                          .dbiterra(), .dbiterrb(), .sbiterra(), .sbiterrb()
                      );

    generate
        for (lane = 0; lane<4; lane = lane + 1) begin : READ_LANES
            assign read_data_out[lane*width +: width] =
                   port_a_data_out[lane*SLOT_WIDTH +: width];
            assign read_data_out[(lane+4)*width +: width] =
                   port_b_data_out[lane*SLOT_WIDTH +: width];
        end
    endgenerate
endmodule

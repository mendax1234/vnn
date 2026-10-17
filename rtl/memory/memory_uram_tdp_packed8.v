`timescale 1ns / 1ps

// Eight logical 16-bit banks packed into one true-dual-port UltraRAM.
// Each physical port is 72 bits wide, divided into eight native 9-bit byte
// lanes. A logical 16-bit value occupies two byte lanes (18 physical bits).
// An XPM is used because RTL inference does not recognize independent 18-bit
// part-select writes as valid UltraRAM byte writes.
module memory_uram_tdp_packed8 #(
        parameter width = 16,
        parameter depth_bits = 10
    )(
        input clk,
        input [7:0] write_en,
        input [depth_bits-1:0] write_address,
        input [width-1:0] write_data_in,
        input read_en,
        input [depth_bits-1:0] read_address,
        output [(8*width)-1:0] read_data_out
    );

    localparam SLOT_WIDTH = 18;
    localparam PORT_WIDTH = 72;
    localparam ADDR_WIDTH = depth_bits + 1;
    localparam MEMORY_SIZE = PORT_WIDTH * (2**ADDR_WIDTH);

    wire port_a_write = |write_en[3:0];
    wire port_b_write = |write_en[7:4];
    wire [ADDR_WIDTH-1:0] port_a_address =
        port_a_write ? {1'b0, write_address} : {1'b0, read_address};
    wire [ADDR_WIDTH-1:0] port_b_address =
        port_b_write ? {1'b1, write_address} : {1'b1, read_address};

    // A 16-bit lane occupies two 9-bit UltraRAM byte-write lanes.
    wire [7:0] port_a_write_enable = {
        {2{write_en[3]}}, {2{write_en[2]}},
        {2{write_en[1]}}, {2{write_en[0]}}
    };
    wire [7:0] port_b_write_enable = {
        {2{write_en[7]}}, {2{write_en[6]}},
        {2{write_en[5]}}, {2{write_en[4]}}
    };
    wire [PORT_WIDTH-1:0] write_data = {
        2'b0, write_data_in, 2'b0, write_data_in,
        2'b0, write_data_in, 2'b0, write_data_in
    };
    wire [PORT_WIDTH-1:0] port_a_data_out;
    wire [PORT_WIDTH-1:0] port_b_data_out;

    xpm_memory_tdpram #(
        .ADDR_WIDTH_A(ADDR_WIDTH),
        .ADDR_WIDTH_B(ADDR_WIDTH),
        .AUTO_SLEEP_TIME(0),
        .BYTE_WRITE_WIDTH_A(9),
        .BYTE_WRITE_WIDTH_B(9),
        .CASCADE_HEIGHT(0),
        .CLOCKING_MODE("common_clock"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("none"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("ultra"),
        .MEMORY_SIZE(MEMORY_SIZE),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_A(PORT_WIDTH),
        .READ_DATA_WIDTH_B(PORT_WIDTH),
        .READ_LATENCY_A(3),
        .READ_LATENCY_B(3),
        .READ_RESET_VALUE_A("0"),
        .READ_RESET_VALUE_B("0"),
        .RST_MODE_A("SYNC"),
        .RST_MODE_B("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_EMBEDDED_CONSTRAINT(0),
        .USE_MEM_INIT(0),
        .WAKEUP_TIME("disable_sleep"),
        .WRITE_DATA_WIDTH_A(PORT_WIDTH),
        .WRITE_DATA_WIDTH_B(PORT_WIDTH),
        .WRITE_MODE_A("no_change"),
        .WRITE_MODE_B("no_change")
    ) uram (
        .clka(clk),
        .clkb(clk),
        .ena(port_a_write || read_en),
        .enb(port_b_write || read_en),
        .wea(port_a_write_enable),
        .web(port_b_write_enable),
        .addra(port_a_address),
        .addrb(port_b_address),
        .dina(write_data),
        .dinb(write_data),
        .douta(port_a_data_out),
        .doutb(port_b_data_out),
        .regcea(1'b1),
        .regceb(1'b1),
        .rsta(1'b0),
        .rstb(1'b0),
        .sleep(1'b0),
        .injectdbiterra(1'b0),
        .injectdbiterrb(1'b0),
        .injectsbiterra(1'b0),
        .injectsbiterrb(1'b0),
        .dbiterra(),
        .dbiterrb(),
        .sbiterra(),
        .sbiterrb()
    );

    assign read_data_out[0*width +: width] = port_a_data_out[0*SLOT_WIDTH +: width];
    assign read_data_out[1*width +: width] = port_a_data_out[1*SLOT_WIDTH +: width];
    assign read_data_out[2*width +: width] = port_a_data_out[2*SLOT_WIDTH +: width];
    assign read_data_out[3*width +: width] = port_a_data_out[3*SLOT_WIDTH +: width];
    assign read_data_out[4*width +: width] = port_b_data_out[0*SLOT_WIDTH +: width];
    assign read_data_out[5*width +: width] = port_b_data_out[1*SLOT_WIDTH +: width];
    assign read_data_out[6*width +: width] = port_b_data_out[2*SLOT_WIDTH +: width];
    assign read_data_out[7*width +: width] = port_b_data_out[3*SLOT_WIDTH +: width];

endmodule

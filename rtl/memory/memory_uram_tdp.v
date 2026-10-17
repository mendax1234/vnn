`timescale 1ns / 1ps

// True-dual-port UltraRAM with a fixed three-cycle external read latency.
// Use an XPM instead of RTL inference so Vivado can place the read pipeline in
// and immediately around the UltraRAM primitive. Both ports retain no-change
// write behavior. The surrounding ping-pong buffer guarantees that its producer
// and consumer access different physical frame memories.
module memory_uram_tdp #(
        parameter width = 16,
        parameter depth_bits = 14
    )(
        input clk,

        input port_a_enable,
        input port_a_write_en,
        input [depth_bits-1:0] port_a_address,
        input [width-1:0] port_a_data_in,
        output [width-1:0] port_a_data_out,

        input port_b_enable,
        input port_b_write_en,
        input [depth_bits-1:0] port_b_address,
        input [width-1:0] port_b_data_in,
        output [width-1:0] port_b_data_out
    );

    localparam MEMORY_SIZE = width * (2**depth_bits);

    // READ_LATENCY=3 preserves the existing contract used by Pool1 while
    // allowing Vivado to use the UltraRAM output register for higher Fmax.
    // Whole-word writes give each port a one-bit XPM write-enable bus.
    xpm_memory_tdpram #(
                          .ADDR_WIDTH_A(depth_bits),
                          .ADDR_WIDTH_B(depth_bits),
                          .AUTO_SLEEP_TIME(0),
                          .BYTE_WRITE_WIDTH_A(width),
                          .BYTE_WRITE_WIDTH_B(width),
                          // Keep the 32Kx16 frame store from becoming one
                          // timing-limiting eight-URAM cascade. Vivado builds
                          // two shorter columns and preserves the XPM's
                          // external three-cycle read latency.
                          .CASCADE_HEIGHT(4),
                          .CLOCKING_MODE("common_clock"),
                          .ECC_MODE("no_ecc"),
                          .MEMORY_INIT_FILE("none"),
                          .MEMORY_INIT_PARAM("0"),
                          .MEMORY_OPTIMIZATION("true"),
                          .MEMORY_PRIMITIVE("ultra"),
                          .MEMORY_SIZE(MEMORY_SIZE),
                          .MESSAGE_CONTROL(0),
                          .READ_DATA_WIDTH_A(width),
                          .READ_DATA_WIDTH_B(width),
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
                          .WRITE_DATA_WIDTH_A(width),
                          .WRITE_DATA_WIDTH_B(width),
                          .WRITE_MODE_A("no_change"),
                          .WRITE_MODE_B("no_change")
                      ) uram (
                          .clka(clk),
                          .clkb(clk),
                          .ena(port_a_enable),
                          .enb(port_b_enable),
                          .wea(port_a_write_en),
                          .web(port_b_write_en),
                          .addra(port_a_address),
                          .addrb(port_b_address),
                          .dina(port_a_data_in),
                          .dinb(port_b_data_in),
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

endmodule

`timescale 1ns / 1ps

module axi_image_receiver (
        input  wire        clk,
        input  wire        rst_n,
        // AXI-Stream Interface
        output wire        s_axis_tready,
        input  wire [15:0] s_axis_tdata,
        input  wire        s_axis_tvalid,
        // Downstream Ping-Pong Buffer Interface (Producer Side)
        input  wire        buffer_ready,  // High if buffers have an empty bank
        output wire        write_en_a0,
        output wire        write_en_a1,
        output wire        write_en_a2,
        output wire [9:0]  write_addr,    // 0 to 1023
        output wire [15:0] write_data,
        output wire        frame_done     // Pulses high when image is fully loaded
    );

    reg [11:0] pixel_count;

    // A valid transaction occurs when master sends data and the coprocessor is ready
    wire handshake = s_axis_tvalid && s_axis_tready;

    // The coprocessor is ready to accept data as long as the downstream buffers have space
    assign s_axis_tready = buffer_ready;

    // Pass data directly through
    assign write_data = s_axis_tdata;

    // Address relies on the lower 10 bits of the counter.
    // This naturally wraps back to 0 at 1024 and 2048.
    assign write_addr = pixel_count[9:0];

    // Combinational routing (Demux) based on the counter
    assign write_en_a0 = handshake && (pixel_count < 12'd1024);
    assign write_en_a1 = handshake && (pixel_count >= 12'd1024) && (pixel_count < 12'd2048);
    assign write_en_a2 = handshake && (pixel_count >= 12'd2048);

    // Pulse frame_done on the exact cycle the 3072nd pixel is written
    assign frame_done = handshake && (pixel_count == 12'd3071);

    // Sequential counter logic
    always @(posedge clk) begin
        if (!rst_n) begin
            pixel_count <= 12'd0;
        end
        else if (handshake) begin
            if (pixel_count == 12'd3071)
                pixel_count <= 12'd0; // Reset for the next image
            else
                pixel_count <= pixel_count + 1'b1;
        end
    end

endmodule

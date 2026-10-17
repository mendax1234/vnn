`timescale 1ns / 1ps

module axi_transmitter #(
        parameter NUM_OUTPUT_WORDS = 10,
        parameter NUM_IMAGES = 100,      // Batch size
        // Set for an asynchronous LUTRAM source.  The default preserves the
        // original transmitter contract used by the Version 1 design.
        parameter LUTRAM_INPUT = 0
    )(
        input  wire        clk,
        input  wire        rst_n,
        // Upstream Ping-Pong Buffer Interface (Consumer Side)
        input  wire        buffer_ready,  // HIGH when buffer has a full set of FC1 results
        output wire        read_en,
        output wire [3:0]  read_addr,
        input  wire [15:0] read_data,
        output wire        frame_done,    // Pulses high when words are sent
        output wire        batch_done,

        // AXI-Stream Interface (Master)
        input  wire        m_axis_tready,
        output wire [15:0] m_axis_tdata,
        output wire        m_axis_tvalid,
        output wire        m_axis_tlast
    );

    // FSM States
    localparam IDLE   = 3'b100;
    localparam STREAM = 3'b010;
    localparam DONE   = 3'b001;

    reg [2:0] state = IDLE, n_state = IDLE;

    /*-- FSM Control Variables --*/
    reg en_word_count, clr_word_count;
    reg en_image_count, clr_image_count;

    /*-- Datapath Registers --*/
    reg [3:0]  word_count;
    reg [31:0] image_count;

    // -------------------------------------------------------------------------
    // State Memory (Sequential)
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
        end
        else begin
            state <= n_state;
        end
    end

    // -------------------------------------------------------------------------
    // Next-State & Output Logic (Combinational)
    // -------------------------------------------------------------------------
    always @(*) begin
        // Default assignments
        n_state         = state;
        en_word_count   = 1'b0;
        clr_word_count  = 1'b0;
        en_image_count  = 1'b0;
        clr_image_count = 1'b0;

        case (state)
            IDLE: begin
                clr_word_count = 1'b1; // Keep word count at 0 until we start
                if (buffer_ready) begin
                    n_state = STREAM;
                end
            end

            STREAM: begin
                // Handshake condition
                if (m_axis_tvalid && m_axis_tready) begin
                    en_word_count = 1'b1;
                    if (word_count == NUM_OUTPUT_WORDS - 1) begin
                        n_state = DONE;
                    end
                end
            end

            DONE: begin
                // Manage the batch image count at the end of a frame
                if (image_count == NUM_IMAGES - 1) begin
                    clr_image_count = 1'b1; // Reset for the next batch
                end
                else begin
                    en_image_count = 1'b1;
                end
                // Unconditionally return to IDLE after 1 clock cycle
                n_state = IDLE;
            end

            default:
                n_state = IDLE;
        endcase
    end

    // Continuous Assignments for standard outputs
    // An asynchronous LUTRAM must remain enabled for the entire STREAM state
    // so TDATA stays stable while TVALID is high and TREADY is low.  Other
    // memories retain the original ready-qualified read-enable behaviour.
    assign read_en       = (state == STREAM) &&
                           (LUTRAM_INPUT || m_axis_tready);
    assign read_addr     = word_count;
    assign m_axis_tdata  = read_data;
    assign m_axis_tvalid = (state == STREAM);

    // Assert TLAST only on the last word of the LAST image in the batch
    assign m_axis_tlast  = (state == STREAM) &&
           (word_count == NUM_OUTPUT_WORDS - 1) &&
           (image_count == NUM_IMAGES - 1);

    // Handshake indicators to upstream buffer / top module
    assign frame_done = (state == DONE);
    assign batch_done = (state == DONE) && (image_count == NUM_IMAGES - 1);

    // -------------------------------------------------------------------------
    // Datapath: Counters
    // -------------------------------------------------------------------------
    // Word Counter
    always @(posedge clk) begin
        if (!rst_n || clr_word_count) begin
            word_count <= 0;
        end
        else if (en_word_count) begin
            word_count <= word_count + 1;
        end
    end

    // Image (Batch) Counter
    always @(posedge clk) begin
        if (!rst_n || clr_image_count) begin
            image_count <= 0;
        end
        else if (en_image_count) begin
            image_count <= image_count + 1;
        end
    end

endmodule

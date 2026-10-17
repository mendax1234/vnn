`timescale 1ns / 1ps

module avg_pool_ping_pong #(
        // Model_version2 default: global_average_pool<128, 8, 8>.
        parameter width          = 16,
        parameter C_DIM          = 128,
        parameter IN_W           = 8,
        parameter IN_H           = 8,
        parameter A_depth_bits   = $clog2(C_DIM * IN_H * IN_W),
        parameter RES_depth_bits = $clog2(C_DIM)
    )(
        input      clk,
        input      rst,
        input      consumer_ready,      // HIGH when upstream Input Buffer is full
        input      producer_ready,      // HIGH when downstream Output Buffer has an empty bank
        output reg consumer_frame_done, // Pulses HIGH to tell upstream we finished reading
        output reg producer_frame_done, // Pulses HIGH to tell downstream we finished writing

        // RAM A (Input)
        output reg                    A_read_en,
        output     [A_depth_bits-1:0] A_read_address,
        input signed [     width-1:0] A_read_data_out,

        // RAM RES (Output)
        output                        RES_write_en,
        output   [RES_depth_bits-1:0] RES_write_address,
        output signed [    width-1:0] RES_write_data_in
    );

    localparam NUMBER_OF_OUTPUT_WORDS = C_DIM;
    localparam SHIFT_AMOUNT = $clog2(IN_H * IN_W);
    localparam ACC_WIDTH    = width + SHIFT_AMOUNT;

    // Define the states of state machine
    localparam IDLE    = 4'b1000;
    localparam COMPUTE = 4'b0100;
    localparam FLUSH   = 4'b0010;
    localparam FINISH  = 4'b0001;

    // State Variables
    reg [3:0] state = IDLE;
    reg [3:0] n_state = IDLE;

    // Coordinate Registers
    wire [$clog2(C_DIM)-1:0] c;
    wire [$clog2(IN_H)-1:0]  h;
    wire [$clog2(IN_W)-1:0]  w;

    // Counter Enables
    reg en_w, en_h, en_c;

    // Stage 1: Read1 Signals
    reg counter_clear_Read1;
    reg reset_acc_Read1;
    reg RES_write_en_Read1;

    // Stage 1.1: Read2 Signals
    reg reset_acc_Read2;
    reg RES_write_en_Read2;

    // Stage 1.2: Read3 Signals
    reg reset_acc_Read3;
    reg RES_write_en_Read3;

    // Stage 2: Accumulate
    reg reset_acc_Acc;
    reg RES_write_en_Acc;
    reg [ACC_WIDTH-1:0] res_Acc;

    // Stage 3: Output
    reg RES_write_en_Out;
    reg [ACC_WIDTH-1:0] res_Out;
    wire [ACC_WIDTH-1:0] res_shifted_Out;

    // -------------------------------------------------------------------------
    // FSM Control Logic
    // -------------------------------------------------------------------------
    always @(posedge clk) state <= n_state;

    always @(*) begin
        n_state = state;
        A_read_en = 0;
        counter_clear_Read1 = 0;
        consumer_frame_done = 1'b0;
        producer_frame_done = 1'b0;

        en_w = 0;
        en_h = 0;
        en_c = 0;
        reset_acc_Read1 = 0;
        RES_write_en_Read1 = 0;

        case (state)
            IDLE: begin
                counter_clear_Read1 = 1'b1;
                if (consumer_ready && producer_ready)
                    n_state = COMPUTE;
            end
            COMPUTE: begin
                A_read_en = 1'b1;

                // GAP accumulates across the entire HxW spatial map.
                // Reset accumulator on the very first pixel of a new channel
                reset_acc_Read1 = (h == 0) && (w == 0);

                // Write the result when we hit the very last pixel of the channel
                RES_write_en_Read1 = (h == IN_H - 1) && (w == IN_W - 1);

                // Cascading Counters
                en_w = 1'b1; // Always ticking in COMPUTE
                en_h = en_w && (w == IN_W - 1);
                en_c = en_h && (h == IN_H - 1);

                // Finish condition based on writes
                if (en_c && (c == C_DIM - 1)) begin
                    n_state = FLUSH;
                end
            end
            FLUSH: begin
                if (RES_write_en && (RES_write_address == NUMBER_OF_OUTPUT_WORDS - 1)) begin
                    n_state = FINISH;
                end
            end
            FINISH: begin
                consumer_frame_done = 1'b1; // Tell Input Ping-Pong to flip its read bank
                producer_frame_done = 1'b1; // Tell Output Ping-Pong to flip its write bank
                n_state = IDLE;
            end
        endcase
    end

    // Datapath
    // Stage 1: Read Stage

    // Pipeline Register 1: Read1/Read2
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            reset_acc_Read2 <= 0;
            RES_write_en_Read2 <= 0;
        end
        else begin
            reset_acc_Read2 <= reset_acc_Read1;
            RES_write_en_Read2 <= RES_write_en_Read1;
        end
    end

    // Pipeline Register 1.1: Read2/Read3
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            reset_acc_Read3 <= 0;
            RES_write_en_Read3 <= 0;
        end
        else begin
            reset_acc_Read3 <= reset_acc_Read2;
            RES_write_en_Read3 <= RES_write_en_Read2;
        end
    end

    // Pipeline Register 1.2: Read3/Accumulate
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            reset_acc_Acc <= 0;
            RES_write_en_Acc <= 0;
        end
        else begin
            reset_acc_Acc <= reset_acc_Read3;
            RES_write_en_Acc <= RES_write_en_Read3;
        end
    end

    // Stage 2: Accumulate Stage
    always @(*) begin
        if (reset_acc_Acc) begin
            // Overwrite accumulator with new pixel
            res_Acc = $signed(A_read_data_out);
        end
        else begin
            // Accumulate.
            res_Acc = $signed(res_Out) + $signed(A_read_data_out);
        end
    end

    // Pipeline Register 2: Accumulate/Output
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Out <= 0;
            res_Out <= 0;
        end
        else begin
            RES_write_en_Out <= RES_write_en_Acc;
            res_Out <= res_Acc;
        end
    end

    // Stage 3: Output Stage
    // Comb Logic
    // Reset Mux
    assign res_shifted_Out = $signed(res_Out) >>> SHIFT_AMOUNT;

    assign RES_write_en = RES_write_en_Out;
    // Truncate the 22-bit shifted value back down to the 16-bit interface
    assign RES_write_data_in = res_shifted_Out[width-1:0];

    // Instantiations: Counters & AGUs
    // Width Counter (Inner)
    Counter #(
                .ADDR_WIDTH($clog2(IN_W)),
                .CNT_INIT(0),
                .MAX_VAL(IN_W-1)
            ) w_cnt (
                .CLK(clk),
                .RESET(!rst || counter_clear_Read1),
                .ENABLE(en_w),
                .CNT(w)
            );

    // Height Counter (Middle)
    Counter #(
                .ADDR_WIDTH($clog2(IN_H)),
                .CNT_INIT(0),
                .MAX_VAL(IN_H-1)
            ) h_cnt (
                .CLK(clk),
                .RESET(!rst || counter_clear_Read1),
                .ENABLE(en_h),
                .CNT(h)
            );

    // Channel Counter (Outer)
    Counter #(
                .ADDR_WIDTH($clog2(C_DIM)),
                .CNT_INIT(0),
                .MAX_VAL(C_DIM-1)
            ) c_cnt (
                .CLK(clk),
                .RESET(!rst || counter_clear_Read1),
                .ENABLE(en_c),
                .CNT(c)
            );

    // Output Address Counter (ticks only when we write a completed average)
    Counter #(
                .ADDR_WIDTH(RES_depth_bits),
                .CNT_INIT(0),
                .MAX_VAL(NUMBER_OF_OUTPUT_WORDS-1)
            ) res_cnt (
                .CLK(clk),
                .RESET(!rst || counter_clear_Read1),
                .ENABLE(RES_write_en),
                .CNT(RES_write_address)
            );

    // Input Address Generation (Direct wire logic since GAP reads sequentially)
    // Channel-major address = (c * IN_H * IN_W) + (h * IN_W) + w
    assign A_read_address = (c * IN_H * IN_W) + (h * IN_W) + w;

endmodule

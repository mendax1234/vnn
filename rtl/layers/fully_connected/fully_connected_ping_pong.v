`timescale 1ns / 1ps

/* ----------------------------------------------------------------------------------
--	(c) Zhu Wenbo NUS
--  RTL Implementation of a Fully Connected (Linear) + Argmax Layer
----------------------------------------------------------------------------------
*/

module fully_connected_ping_pong #(
        parameter width          = 16,
        parameter FRAC_BITS      = 10,
        // Model_version2 dimensions: 128 GAP features -> 10 classes.
        parameter IN_F           = 128,
        parameter OUT_F          = 10,
        parameter A_depth_bits   = $clog2(IN_F),
        parameter B_depth_bits   = $clog2(IN_F * OUT_F),
        parameter C_depth_bits   = $clog2(OUT_F),
        parameter RES_depth_bits = 1,
        // Full Q12.20 product plus guard bits for an IN_F-term sum.
        parameter ACC_WIDTH      = (2 * width) + $clog2(IN_F)
    ) (
        input      clk,
        input      rst,
        input      consumer_ready,      // HIGH when upstream Input Buffer is full
        input      producer_ready,      // HIGH when downstream Output Buffer has an empty bank
        output reg consumer_frame_done, // Pulses HIGH to tell upstream we finished reading
        output reg producer_frame_done, // Pulses HIGH to tell downstream we finished writing

        // RAM A (Input Features - from GAP)
        output reg                    A_read_en,
        output     [A_depth_bits-1:0] A_read_address,
        input      [       width-1:0] A_read_data_out,

        // RAM B (Weights)
        output reg                    B_read_en,
        output     [B_depth_bits-1:0] B_read_address,
        input      [       width-1:0] B_read_data_out,

        // RAM C (Bias)
        output reg                    C_read_en,
        output     [C_depth_bits-1:0] C_read_address,
        input      [       width-1:0] C_read_data_out,

        // RAM RES (Output Logits)
        output                          RES_write_en,
        output     [RES_depth_bits-1:0] RES_write_address,
        output     [         width-1:0] RES_write_data_in
    );

    // Total number of output data
    localparam NUMBER_OF_OUTPUT_WORDS = 1; // As we combine the Argmax with Fully connected

    // Define the states of state machine
    localparam IDLE    = 4'b1000;
    localparam COMPUTE = 4'b0100;
    localparam FLUSH   = 4'b0010;
    localparam FINISH  = 4'b0001;

    // State Variables
    reg [3:0] state = IDLE;
    reg [3:0] n_state = IDLE;

    /*-- FSM Control Variable --*/
    // Coordinate Registers (Only 2 needed for FC)
    wire [$clog2(IN_F)-1:0]  in_f;
    wire [$clog2(OUT_F)-1:0] out_f;

    // Coordinate counter enable registers
    reg en_in_f, en_out_f;

    // Stage 1: Read1 Signals
    reg RES_write_en_Read1;
    reg counter_clear_Read1;
    reg compare_en_Read1;
    reg reset_acc_Read1;
    reg lutram_read_en_Read1;

    // Stage 1.1: Read2 Signals
    reg RES_write_en_Read2;
    reg compare_en_Read2;
    reg reset_acc_Read2;
    reg lutram_read_en_Read2;
    reg [$clog2(IN_F)-1:0] in_f_Read2;
    reg [$clog2(OUT_F)-1:0] out_f_Read2;

    // Stage 1.2: Read3 Signals
    reg RES_write_en_Read3;
    reg compare_en_Read3;
    reg reset_acc_Read3;
    reg lutram_read_en_Read3;
    reg [$clog2(IN_F)-1:0] in_f_Read3;
    reg [$clog2(OUT_F)-1:0] out_f_Read3;

    // Stage 2: Multiply Signals
    reg RES_write_en_Mul;
    wire [2*width-1:0] mul_Mul;
    reg compare_en_Mul;
    reg lutram_read_en_Mul;
    reg [$clog2(IN_F)-1:0] in_f_Mul;
    reg [$clog2(OUT_F)-1:0] out_f_Mul;
    reg reset_acc_Mul;

    // Stage 3: Accumulate Signals
    reg [2*width-1:0] mul_Acc;
    reg [width-1:0] bias_Acc;
    reg RES_write_en_Acc;
    wire [ACC_WIDTH-1:0] res_Acc;
    reg compare_en_Acc;
    reg [$clog2(OUT_F)-1:0] out_f_Acc;
    reg reset_acc_Acc;

    // Stage 4: Output Signals
    reg RES_write_en_Out;
    reg [ACC_WIDTH-1:0] res_Out;
    reg [width-1:0] bias_Out;
    wire [ACC_WIDTH-1:0] res_biased_Out;
    wire [ACC_WIDTH-1:0] aligned_bias;
    reg compare_en_Out;
    reg [$clog2(OUT_F)-1:0] out_f_Out;
    reg [ACC_WIDTH-1:0] best_val_Out;
    reg [$clog2(OUT_F)-1:0] best_idx_Out;

    // Stage 5: Compare Signals
    reg RES_write_en_Cmp;
    reg [ACC_WIDTH-1:0] best_val_Cmp;
    reg [$clog2(OUT_F)-1:0] best_idx_Cmp;

    // Mealy FSM as the control logic
    // State Memory
    always @(posedge clk) begin
        state <= n_state;
    end

    // Next State & Output Logic
    always @(*) begin
        n_state = state; // Hold the current state unless explicitly transits

        // Input and bias LUTRAM reads are issued at the Multiply boundary,
        // after their addresses have followed the BRAM control pipeline.
        A_read_en = lutram_read_en_Mul;
        B_read_en = 0;
        C_read_en = lutram_read_en_Mul;
        RES_write_en_Read1 = 0;
        counter_clear_Read1 = 0;
        consumer_frame_done = 1'b0;
        producer_frame_done = 1'b0;
        compare_en_Read1 = 0;
        reset_acc_Read1 = 0;
        lutram_read_en_Read1 = 0;

        // Default Counter Enables
        en_in_f  = 0;
        en_out_f = 0;

        case (state)
            IDLE: begin
                counter_clear_Read1 = 1'b1;
                if (consumer_ready && producer_ready) begin
                    n_state = COMPUTE;
                end
            end
            COMPUTE: begin
                B_read_en  = 1'b1;
                lutram_read_en_Read1 = 1'b1;

                reset_acc_Read1 = (in_f == 0);
                RES_write_en_Read1 = (in_f == IN_F - 1) && (out_f == OUT_F - 1);
                // Write triggers at the end of the inner feature loop
                compare_en_Read1 = (in_f == IN_F - 1);

                // Cascading Enables Logic inside FSM
                en_in_f  = 1'b1; // Inner loop always ticking in COMPUTE
                en_out_f = en_in_f && (in_f == IN_F - 1);

                // Finish condition based on write addresses
                if (en_out_f && (out_f == OUT_F - 1)) begin
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

    // DATAPATH
    // Stage 1: Read Stage

    // Pipeline 1 Register: Read1/Read2
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Read2 <= 0;
            compare_en_Read2 <= 0;
            lutram_read_en_Read2 <= 0;
            in_f_Read2 <= 0;
            out_f_Read2 <= 0;
            reset_acc_Read2 <= 0;
        end
        else begin
            RES_write_en_Read2 <= RES_write_en_Read1;
            compare_en_Read2 <= compare_en_Read1;
            lutram_read_en_Read2 <= lutram_read_en_Read1;
            in_f_Read2 <= in_f;
            out_f_Read2 <= out_f;
            reset_acc_Read2 <= reset_acc_Read1;
        end
    end

    // Pipeline 1.1 Register: Read2/Read3
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Read3 <= 0;
            compare_en_Read3 <= 0;
            lutram_read_en_Read3 <= 0;
            in_f_Read3 <= 0;
            out_f_Read3 <= 0;
            reset_acc_Read3 <= 0;
        end
        else begin
            RES_write_en_Read3 <= RES_write_en_Read2;
            compare_en_Read3 <= compare_en_Read2;
            lutram_read_en_Read3 <= lutram_read_en_Read2;
            in_f_Read3 <= in_f_Read2;
            out_f_Read3 <= out_f_Read2;
            reset_acc_Read3 <= reset_acc_Read2;
        end
    end

    // Pipeline 1.1 Register: Read3/Mul
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Mul <= 0;
            compare_en_Mul <= 0;
            lutram_read_en_Mul <= 0;
            in_f_Mul <= 0;
            out_f_Mul <= 0;
            reset_acc_Mul <= 0;
        end
        else begin
            RES_write_en_Mul <= RES_write_en_Read3;
            compare_en_Mul <= compare_en_Read3;
            lutram_read_en_Mul <= lutram_read_en_Read3;
            in_f_Mul <= in_f_Read3;
            out_f_Mul <= out_f_Read3;
            reset_acc_Mul <= reset_acc_Read3;
        end
    end

    // Stage 2: Multiply
    // Comb Logic
    assign mul_Mul = $signed(A_read_data_out) * $signed(B_read_data_out);

    // Pipeline 2 Register: Multiply/Accumulate
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            mul_Acc <= 0;
            RES_write_en_Acc <= 0;
            bias_Acc <= 0;
            compare_en_Acc <= 0;
            out_f_Acc <= 0;
            reset_acc_Acc <= 0;
        end
        else begin
            // Keep the complete Q12.20 product. HLS accumulates before
            // converting its logits, so per-product truncation is not equal.
            mul_Acc <= mul_Mul;
            RES_write_en_Acc <= RES_write_en_Mul;
            // A and C are asynchronous LUTRAM outputs addressed by the
            // registered Multiply-stage coordinates below.
            bias_Acc <= C_read_data_out;
            compare_en_Acc <= compare_en_Mul;
            out_f_Acc <= out_f_Mul;
            reset_acc_Acc <= reset_acc_Mul;
        end
    end

    // Stage 3: Accumulate
    // Comb Logic
    assign res_Acc = reset_acc_Acc
           ? {{(ACC_WIDTH - 2*width){mul_Acc[2*width-1]}}, mul_Acc}
           : $signed(res_Out)
           + {{(ACC_WIDTH - 2*width){mul_Acc[2*width-1]}}, mul_Acc};

    // Pipeline 3 Register: Accumulate/Output
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            res_Out <= 0;
            RES_write_en_Out <= 0;
            bias_Out <= 0;
            compare_en_Out <= 0;
            out_f_Out <= 0;
        end
        else begin
            res_Out <= res_Acc;
            RES_write_en_Out <= RES_write_en_Acc;
            bias_Out <= bias_Acc;
            compare_en_Out <= compare_en_Acc;
            out_f_Out <= out_f_Acc;
        end
    end

    // Stage 4: Output
    // Comb Logic
    // Convert the Q6.10 bias to the accumulator's Q12.20 scale.
    assign aligned_bias = $signed({{(ACC_WIDTH - width){bias_Out[width-1]}}, bias_Out}) <<< FRAC_BITS;
    assign res_biased_Out = $signed(res_Out) + aligned_bias;

    // Compare Mux
    always @(*) begin
        if (compare_en_Out && ((out_f_Out == 0) || ($signed(res_biased_Out) > $signed(best_val_Cmp)))) begin
            best_val_Out = res_biased_Out;
            best_idx_Out = out_f_Out;
        end
        else begin
            best_val_Out = best_val_Cmp;
            best_idx_Out = best_idx_Cmp;
        end
    end

    // Stage 5: Compare
    // Pipeline 4 Register: Output/Compare
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            best_val_Cmp <= 0;
            best_idx_Cmp <= 0;
            RES_write_en_Cmp <= 0;
        end
        else begin
            best_val_Cmp <= best_val_Out;
            best_idx_Cmp <= best_idx_Out;
            RES_write_en_Cmp <= RES_write_en_Out;
        end
    end

    // Truncate Q12.20 back to Q6.10
    assign RES_write_data_in = {{(width - $clog2(OUT_F)){1'b0}}, best_idx_Cmp};
    assign RES_write_en = RES_write_en_Cmp;

    // COUNTERS & AGU WIRES
    // Input Feature Counter (Inner Loop: 0 to 63)
    Counter #(
                .ADDR_WIDTH($clog2(IN_F)),
                .CNT_INIT(0),
                .MAX_VAL(IN_F - 1)
            ) in_f_counter (
                .CLK(clk),
                .RESET(!rst || counter_clear_Read1),
                .ENABLE(en_in_f),
                .CNT(in_f)
            );

    // Output Feature Counter (Outer Loop: 0 to 9)
    Counter #(
                .ADDR_WIDTH($clog2(OUT_F)),
                .CNT_INIT(0),
                .MAX_VAL(OUT_F - 1)
            ) out_f_counter (
                .CLK(clk),
                .RESET(!rst || counter_clear_Read1),
                .ENABLE(en_out_f),
                .CNT(out_f)
            );

    // Output RAM Address generator (Ticks only when we write)
    Counter #(
                .ADDR_WIDTH(RES_depth_bits),
                .CNT_INIT(0),
                .MAX_VAL(NUMBER_OF_OUTPUT_WORDS - 1)
            ) RES_Counter (
                .CLK(clk),
                .RESET(!rst || counter_clear_Read1),
                .ENABLE(RES_write_en),
                .CNT(RES_write_address)
            );

    // Direct Address Wiring

    // A (Input): delay the LUTRAM address to align with the weight BRAM output.
    assign A_read_address = in_f_Mul;

    // B (Weights): Flattened 2D index (out_f * IN_F + in_f)
    assign B_read_address = (out_f * IN_F) + in_f;

    // C (Bias): use the matching delayed output-channel coordinate.
    assign C_read_address = out_f_Mul;

endmodule

`timescale 1ns / 1ps

/* ----------------------------------------------------------------------------------
--  RTL Implementation of Fused Convolution Layer 1 + ReLU
--  Input-channel unroll factor:  3 (all input channels processed in parallel)
--  Output-channel unroll factor: 4 (four output channels processed in parallel)
--  The 3x3 kernel positions are processed temporally, one position per cycle.
--  The original Counter/AGU/pipeline-register structure is retained.
--  Four results are serialized as two channel pairs into the downstream buffer.
----------------------------------------------------------------------------------
*/

module conv1_relu_unrolled_ping_pong #(
        parameter width          = 16,
        parameter FRAC_BITS      = 10,
        parameter A_depth_bits   = 10,
        parameter B_depth_bits   = 7,
        parameter C_depth_bits   = 3,
        parameter RES_depth_bits = 15,
        parameter OC_DIM         = 32,
        parameter OUT_UNROLL     = 4,
        parameter IN_H           = 32,
        parameter IN_W           = 32,
        parameter K              = 3
    ) (
        input clk, input rst, input consumer_ready, input producer_ready,
        output reg consumer_frame_done, output reg producer_frame_done,
        output reg A_read_en, output reg [A_depth_bits-1:0] A_read_address,
        output reg B_read_en, output reg [B_depth_bits-1:0] B_read_address,

        // Input feature-map data: three input channels
        input [width-1:0] A_read_data_out_0, A_read_data_out_1, A_read_data_out_2,
        // Weight data: four output-channel lanes, three input channels each
        input [width-1:0] B_read_data_out_0_oc0, B_read_data_out_1_oc0, B_read_data_out_2_oc0,
        input [width-1:0] B_read_data_out_0_oc1, B_read_data_out_1_oc1, B_read_data_out_2_oc1,
        input [width-1:0] B_read_data_out_0_oc2, B_read_data_out_1_oc2, B_read_data_out_2_oc2,
        input [width-1:0] B_read_data_out_0_oc3, B_read_data_out_1_oc3, B_read_data_out_2_oc3,

        // RAM C (Bias)
        (* keep = "true" *) output reg C_read_en,
        output reg [C_depth_bits-1:0] C_read_address,
        input [width-1:0] C_read_data_out_oc0,
        input [width-1:0] C_read_data_out_oc1,
        input [width-1:0] C_read_data_out_oc2,
        input [width-1:0] C_read_data_out_oc3,

        // RAM RES (Output - EVEN and ODD via TDP RAM)
        output RES_write_en_even,
        output [RES_depth_bits-1:0] RES_write_address_even,
        output [width-1:0] RES_write_data_in_even,
        output RES_write_en_odd,
        output [RES_depth_bits-1:0] RES_write_address_odd,
        output [width-1:0] RES_write_data_in_odd
    );

    localparam NUMBER_OF_OUTPUT_WORDS = OC_DIM * IN_H * IN_W;
    localparam IDLE = 4'b1000, COMPUTE = 4'b0100, FLUSH = 4'b0010, FINISH = 4'b0001;
    reg [3:0] state = IDLE, n_state = IDLE;

    /*-- FSM Control Variables --*/
    wire [$clog2(OC_DIM/OUT_UNROLL)-1:0] oc;
    wire [$clog2(IN_H)-1:0] row;
    wire [$clog2(IN_W)-1:0] col;
    wire [$clog2(K)-1:0] k_row, k_col;
    reg en_k_col, en_k_row, en_col, en_row, en_oc;

    // Stage 1: Read1 Signals (combinational AGU outputs)
    wire [A_depth_bits-1:0] A_read_address_Read1;
    wire [B_depth_bits-1:0] B_read_address_Read1;
    wire [C_depth_bits-1:0] C_read_address_Read1;
    wire [RES_depth_bits-1:0] calc_addr_0_Read1, calc_addr_1_Read1, calc_addr_2_Read1, calc_addr_3_Read1;
    wire out_of_bounds_Read1;
    reg A_read_en_Read1, B_read_en_Read1, C_read_en_Read1;
    reg RES_write_en_Read1, counter_clear_Read1, reset_acc_Read1;

    // Stage 1.1: Read2 Signals
    reg RES_write_en_Read2, reset_acc_Read2, out_of_bounds_Read2;
    reg B_read_en_Read2, C_read_en_Read2;
    reg [B_depth_bits-1:0] B_read_address_Read2;
    reg [C_depth_bits-1:0] C_read_address_Read2;
    reg [RES_depth_bits-1:0] calc_addr_0_Read2, calc_addr_1_Read2, calc_addr_2_Read2, calc_addr_3_Read2;

    // Stage 1.2: Read3 Signals
    reg RES_write_en_Read3, reset_acc_Read3, out_of_bounds_Read3;
    reg B_read_en_Read3, C_read_en_Read3;
    reg [B_depth_bits-1:0] B_read_address_Read3;
    reg [C_depth_bits-1:0] C_read_address_Read3;
    reg [RES_depth_bits-1:0] calc_addr_0_Read3, calc_addr_1_Read3, calc_addr_2_Read3, calc_addr_3_Read3;

    // Stage 1.3: Read4 Signals
    reg RES_write_en_Read4, reset_acc_Read4, out_of_bounds_Read4;
    reg B_read_en_Read4, C_read_en_Read4;
    reg [B_depth_bits-1:0] B_read_address_Read4;
    reg [C_depth_bits-1:0] C_read_address_Read4;
    reg [RES_depth_bits-1:0] calc_addr_0_Read4, calc_addr_1_Read4, calc_addr_2_Read4, calc_addr_3_Read4;

    // Stage 1.4: Read5 Signals
    reg RES_write_en_Read5, reset_acc_Read5, out_of_bounds_Read5;
    reg [RES_depth_bits-1:0] calc_addr_0_Read5, calc_addr_1_Read5, calc_addr_2_Read5, calc_addr_3_Read5;

    wire [width-1:0] padded_A_data_0_Read5, padded_A_data_1_Read5, padded_A_data_2_Read5;

    // Stage 2: Multiply Signals
    reg RES_write_en_Mul, reset_acc_Mul;
    reg [RES_depth_bits-1:0] res_addr_0_Mul, res_addr_1_Mul, res_addr_2_Mul, res_addr_3_Mul;
    reg signed [width-1:0] B_read_data_Mul_0_oc0, B_read_data_Mul_1_oc0, B_read_data_Mul_2_oc0;
    reg signed [width-1:0] B_read_data_Mul_0_oc1, B_read_data_Mul_1_oc1, B_read_data_Mul_2_oc1;
    reg signed [width-1:0] B_read_data_Mul_0_oc2, B_read_data_Mul_1_oc2, B_read_data_Mul_2_oc2;
    reg signed [width-1:0] B_read_data_Mul_0_oc3, B_read_data_Mul_1_oc3, B_read_data_Mul_2_oc3;
    reg signed [width-1:0] C_read_data_Mul_oc0, C_read_data_Mul_oc1, C_read_data_Mul_oc2, C_read_data_Mul_oc3;
    reg signed [width-1:0] padded_A_data_0_Mul, padded_A_data_1_Mul, padded_A_data_2_Mul;
    wire [2*width-1:0] mul_0_oc0_Mul, mul_1_oc0_Mul, mul_2_oc0_Mul;
    wire [2*width-1:0] mul_0_oc1_Mul, mul_1_oc1_Mul, mul_2_oc1_Mul;
    wire [2*width-1:0] mul_0_oc2_Mul, mul_1_oc2_Mul, mul_2_oc2_Mul;
    wire [2*width-1:0] mul_0_oc3_Mul, mul_1_oc3_Mul, mul_2_oc3_Mul;

    // Stage 3: Accumulate Signals
    reg [2*width-1:0] mul_0_oc0_Acc, mul_1_oc0_Acc, mul_2_oc0_Acc;
    reg [2*width-1:0] mul_0_oc1_Acc, mul_1_oc1_Acc, mul_2_oc1_Acc;
    reg [2*width-1:0] mul_0_oc2_Acc, mul_1_oc2_Acc, mul_2_oc2_Acc;
    reg [2*width-1:0] mul_0_oc3_Acc, mul_1_oc3_Acc, mul_2_oc3_Acc;
    reg [width-1:0] bias_oc0_Acc, bias_oc1_Acc, bias_oc2_Acc, bias_oc3_Acc;
    reg RES_write_en_Acc, reset_acc_Acc;
    wire [2*width-1:0] spatial_sum_oc0, spatial_sum_oc1, spatial_sum_oc2, spatial_sum_oc3;
    wire [2*width-1:0] res_Acc_oc0, res_Acc_oc1, res_Acc_oc2, res_Acc_oc3;
    reg [RES_depth_bits-1:0] res_addr_0_Acc, res_addr_1_Acc, res_addr_2_Acc, res_addr_3_Acc;

    // Stage 4: Bias Signals
    reg [2*width-1:0] accumulator_oc0_Bias, accumulator_oc1_Bias, accumulator_oc2_Bias, accumulator_oc3_Bias;
    reg [width-1:0] bias_oc0_Bias, bias_oc1_Bias, bias_oc2_Bias, bias_oc3_Bias;
    reg RES_write_en_Bias;
    reg [RES_depth_bits-1:0] addr_0_Bias, addr_1_Bias, addr_2_Bias, addr_3_Bias;
    wire [2*width-1:0] aligned_bias_oc0, aligned_bias_oc1, aligned_bias_oc2, aligned_bias_oc3;
    wire [2*width-1:0] res_biased_oc0, res_biased_oc1, res_biased_oc2, res_biased_oc3;
    wire [width-1:0] trunc_oc0, trunc_oc1, trunc_oc2, trunc_oc3;

    // Stage 5: ReLU input registers and combinational outputs
    reg RES_write_en_Relu;
    reg [width-1:0] data_0_Relu, data_1_Relu, data_2_Relu, data_3_Relu;
    reg [RES_depth_bits-1:0] addr_0_Relu, addr_1_Relu, addr_2_Relu, addr_3_Relu;
    wire [width-1:0] relu_oc0, relu_oc1, relu_oc2, relu_oc3;

    // Stages 6-7: Two output-channel pairs for the producer write ports
    reg RES_write_en_Out1, RES_write_en_Out2;
    reg [width-1:0] data_0_Out1, data_1_Out1, data_2_Out1, data_3_Out1;
    reg [RES_depth_bits-1:0] addr_0_Out1, addr_1_Out1, addr_2_Out1, addr_3_Out1;
    reg [width-1:0] data_2_Out2, data_3_Out2;
    reg [RES_depth_bits-1:0] addr_2_Out2, addr_3_Out2;

    // State Memory
    always @(posedge clk) state <= n_state;

    // Next State & Output Logic
    always @(*) begin
        n_state = state;
        A_read_en_Read1 = 0;
        B_read_en_Read1 = 0;
        C_read_en_Read1 = 0;
        RES_write_en_Read1 = 0;
        counter_clear_Read1 = 0;
        reset_acc_Read1 = 0;
        consumer_frame_done = 0;
        producer_frame_done = 0;
        en_k_col = 0;
        en_k_row = 0;
        en_col = 0;
        en_row = 0;
        en_oc = 0;
        case (state)
            IDLE: begin
                counter_clear_Read1 = 1;
                if (consumer_ready && producer_ready)
                    n_state = COMPUTE;
            end
            COMPUTE: begin
                A_read_en_Read1 = !out_of_bounds_Read1;
                B_read_en_Read1 = 1;
                C_read_en_Read1 = 1;
                reset_acc_Read1 = (k_row == 0) && (k_col == 0);
                RES_write_en_Read1 = (k_row == K-1) && (k_col == K-1);
                en_k_col = 1;
                en_k_row = en_k_col && (k_col == K-1);
                en_col = en_k_row && (k_row == K-1);
                en_row = en_col && (col == IN_W-1);
                en_oc = en_row && (row == IN_H-1);
                if (en_oc && (oc == (OC_DIM/OUT_UNROLL)-1))
                    n_state = FLUSH;
            end
            FLUSH:
                if (RES_write_en_Out2 && (addr_3_Out2 == NUMBER_OF_OUTPUT_WORDS-1))
                    n_state = FINISH;
            FINISH: begin
                consumer_frame_done = 1;
                producer_frame_done = 1;
                n_state = IDLE;
            end
        endcase
    end

    // Datapath
    // Stage 1: Read
    // Pipeline Register 1: Read1/Read2
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            A_read_en <= 0;
            A_read_address <= 0;
            B_read_en_Read2 <= 0;
            C_read_en_Read2 <= 0;
            B_read_address_Read2 <= 0;
            C_read_address_Read2 <= 0;
            RES_write_en_Read2 <= 0;
            out_of_bounds_Read2 <= 0;
            reset_acc_Read2 <= 0;
            calc_addr_0_Read2 <= 0;
            calc_addr_1_Read2 <= 0;
            calc_addr_2_Read2 <= 0;
            calc_addr_3_Read2 <= 0;
        end
        else begin
            A_read_en <= A_read_en_Read1;
            A_read_address <= A_read_address_Read1;
            B_read_en_Read2 <= B_read_en_Read1;
            C_read_en_Read2 <= C_read_en_Read1;
            B_read_address_Read2 <= B_read_address_Read1;
            C_read_address_Read2 <= C_read_address_Read1;
            RES_write_en_Read2 <= RES_write_en_Read1;
            out_of_bounds_Read2 <= out_of_bounds_Read1;
            reset_acc_Read2 <= reset_acc_Read1;
            calc_addr_0_Read2 <= calc_addr_0_Read1;
            calc_addr_1_Read2 <= calc_addr_1_Read1;
            calc_addr_2_Read2 <= calc_addr_2_Read1;
            calc_addr_3_Read2 <= calc_addr_3_Read1;
        end
    end

    // Pipeline Register 2: Read2/Read3
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Read3 <= 0;
            out_of_bounds_Read3 <= 0;
            reset_acc_Read3 <= 0;
            B_read_en_Read3 <= 0;
            C_read_en_Read3 <= 0;
            B_read_address_Read3 <= 0;
            C_read_address_Read3 <= 0;
            calc_addr_0_Read3 <= 0;
            calc_addr_1_Read3 <= 0;
            calc_addr_2_Read3 <= 0;
            calc_addr_3_Read3 <= 0;
        end
        else begin
            RES_write_en_Read3 <= RES_write_en_Read2;
            out_of_bounds_Read3 <= out_of_bounds_Read2;
            reset_acc_Read3 <= reset_acc_Read2;
            B_read_en_Read3 <= B_read_en_Read2;
            C_read_en_Read3 <= C_read_en_Read2;
            B_read_address_Read3 <= B_read_address_Read2;
            C_read_address_Read3 <= C_read_address_Read2;
            calc_addr_0_Read3 <= calc_addr_0_Read2;
            calc_addr_1_Read3 <= calc_addr_1_Read2;
            calc_addr_2_Read3 <= calc_addr_2_Read2;
            calc_addr_3_Read3 <= calc_addr_3_Read2;
        end
    end

    // Pipeline Register 3: Read3/Read4
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Read4 <= 0;
            out_of_bounds_Read4 <= 0;
            reset_acc_Read4 <= 0;
            B_read_en_Read4 <= 0;
            C_read_en_Read4 <= 0;
            B_read_address_Read4 <= 0;
            C_read_address_Read4 <= 0;
            calc_addr_0_Read4 <= 0;
            calc_addr_1_Read4 <= 0;
            calc_addr_2_Read4 <= 0;
            calc_addr_3_Read4 <= 0;
        end
        else begin
            RES_write_en_Read4 <= RES_write_en_Read3;
            out_of_bounds_Read4 <= out_of_bounds_Read3;
            reset_acc_Read4 <= reset_acc_Read3;
            B_read_en_Read4 <= B_read_en_Read3;
            C_read_en_Read4 <= C_read_en_Read3;
            B_read_address_Read4 <= B_read_address_Read3;
            C_read_address_Read4 <= C_read_address_Read3;
            calc_addr_0_Read4 <= calc_addr_0_Read3;
            calc_addr_1_Read4 <= calc_addr_1_Read3;
            calc_addr_2_Read4 <= calc_addr_2_Read3;
            calc_addr_3_Read4 <= calc_addr_3_Read3;
        end
    end

    // Pipeline Register 4: Read4/Read5
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Read5 <= 0;
            out_of_bounds_Read5 <= 0;
            reset_acc_Read5 <= 0;
            B_read_en <= 0;
            C_read_en <= 0;
            B_read_address <= 0;
            C_read_address <= 0;
            calc_addr_0_Read5 <= 0;
            calc_addr_1_Read5 <= 0;
            calc_addr_2_Read5 <= 0;
            calc_addr_3_Read5 <= 0;
        end
        else begin
            RES_write_en_Read5 <= RES_write_en_Read4;
            out_of_bounds_Read5 <= out_of_bounds_Read4;
            reset_acc_Read5 <= reset_acc_Read4;
            // Delay the asynchronous weight/bias LUTRAM addresses until the
            // matching activation emerges from the input BRAM pipeline.
            B_read_en <= B_read_en_Read4;
            C_read_en <= C_read_en_Read4;
            B_read_address <= B_read_address_Read4;
            C_read_address <= C_read_address_Read4;
            calc_addr_0_Read5 <= calc_addr_0_Read4;
            calc_addr_1_Read5 <= calc_addr_1_Read4;
            calc_addr_2_Read5 <= calc_addr_2_Read4;
            calc_addr_3_Read5 <= calc_addr_3_Read4;
        end
    end

    // Stage 1: Read -- combinational padding for out-of-bounds pixels
    assign padded_A_data_0_Read5 = out_of_bounds_Read5 ? {width{1'b0}} : A_read_data_out_0;
    assign padded_A_data_1_Read5 = out_of_bounds_Read5 ? {width{1'b0}} : A_read_data_out_1;
    assign padded_A_data_2_Read5 = out_of_bounds_Read5 ? {width{1'b0}} : A_read_data_out_2;

    // Pipeline Register 4: Read5/Multiply
    // The activation arrives from BRAM while the delayed addresses select the
    // matching asynchronous LUTRAM weights and biases.  Capture all operands
    // together at this aligned boundary.
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Mul <= 0;
            reset_acc_Mul <= 0;
            res_addr_0_Mul <= 0;
            res_addr_1_Mul <= 0;
            res_addr_2_Mul <= 0;
            res_addr_3_Mul <= 0;
            padded_A_data_0_Mul <= 0;
            padded_A_data_1_Mul <= 0;
            padded_A_data_2_Mul <= 0;
            B_read_data_Mul_0_oc0 <= 0;
            B_read_data_Mul_1_oc0 <= 0;
            B_read_data_Mul_2_oc0 <= 0;
            B_read_data_Mul_0_oc1 <= 0;
            B_read_data_Mul_1_oc1 <= 0;
            B_read_data_Mul_2_oc1 <= 0;
            B_read_data_Mul_0_oc2 <= 0;
            B_read_data_Mul_1_oc2 <= 0;
            B_read_data_Mul_2_oc2 <= 0;
            B_read_data_Mul_0_oc3 <= 0;
            B_read_data_Mul_1_oc3 <= 0;
            B_read_data_Mul_2_oc3 <= 0;
            C_read_data_Mul_oc0 <= 0;
            C_read_data_Mul_oc1 <= 0;
            C_read_data_Mul_oc2 <= 0;
            C_read_data_Mul_oc3 <= 0;
        end
        else begin
            RES_write_en_Mul <= RES_write_en_Read5;
            reset_acc_Mul <= reset_acc_Read5;
            res_addr_0_Mul <= calc_addr_0_Read5;
            res_addr_1_Mul <= calc_addr_1_Read5;
            res_addr_2_Mul <= calc_addr_2_Read5;
            res_addr_3_Mul <= calc_addr_3_Read5;
            padded_A_data_0_Mul <= padded_A_data_0_Read5;
            padded_A_data_1_Mul <= padded_A_data_1_Read5;
            padded_A_data_2_Mul <= padded_A_data_2_Read5;
            B_read_data_Mul_0_oc0 <= B_read_data_out_0_oc0;
            B_read_data_Mul_1_oc0 <= B_read_data_out_1_oc0;
            B_read_data_Mul_2_oc0 <= B_read_data_out_2_oc0;
            B_read_data_Mul_0_oc1 <= B_read_data_out_0_oc1;
            B_read_data_Mul_1_oc1 <= B_read_data_out_1_oc1;
            B_read_data_Mul_2_oc1 <= B_read_data_out_2_oc1;
            B_read_data_Mul_0_oc2 <= B_read_data_out_0_oc2;
            B_read_data_Mul_1_oc2 <= B_read_data_out_1_oc2;
            B_read_data_Mul_2_oc2 <= B_read_data_out_2_oc2;
            B_read_data_Mul_0_oc3 <= B_read_data_out_0_oc3;
            B_read_data_Mul_1_oc3 <= B_read_data_out_1_oc3;
            B_read_data_Mul_2_oc3 <= B_read_data_out_2_oc3;
            C_read_data_Mul_oc0 <= C_read_data_out_oc0;
            C_read_data_Mul_oc1 <= C_read_data_out_oc1;
            C_read_data_Mul_oc2 <= C_read_data_out_oc2;
            C_read_data_Mul_oc3 <= C_read_data_out_oc3;
        end
    end

    // Stage 2: Multiply
    // Comb Logic
    assign mul_0_oc0_Mul = $signed(padded_A_data_0_Mul) * $signed(B_read_data_Mul_0_oc0);
    assign mul_1_oc0_Mul = $signed(padded_A_data_1_Mul) * $signed(B_read_data_Mul_1_oc0);
    assign mul_2_oc0_Mul = $signed(padded_A_data_2_Mul) * $signed(B_read_data_Mul_2_oc0);
    assign mul_0_oc1_Mul = $signed(padded_A_data_0_Mul) * $signed(B_read_data_Mul_0_oc1);
    assign mul_1_oc1_Mul = $signed(padded_A_data_1_Mul) * $signed(B_read_data_Mul_1_oc1);
    assign mul_2_oc1_Mul = $signed(padded_A_data_2_Mul) * $signed(B_read_data_Mul_2_oc1);
    assign mul_0_oc2_Mul = $signed(padded_A_data_0_Mul) * $signed(B_read_data_Mul_0_oc2);
    assign mul_1_oc2_Mul = $signed(padded_A_data_1_Mul) * $signed(B_read_data_Mul_1_oc2);
    assign mul_2_oc2_Mul = $signed(padded_A_data_2_Mul) * $signed(B_read_data_Mul_2_oc2);
    assign mul_0_oc3_Mul = $signed(padded_A_data_0_Mul) * $signed(B_read_data_Mul_0_oc3);
    assign mul_1_oc3_Mul = $signed(padded_A_data_1_Mul) * $signed(B_read_data_Mul_1_oc3);
    assign mul_2_oc3_Mul = $signed(padded_A_data_2_Mul) * $signed(B_read_data_Mul_2_oc3);

    // Pipeline Register 2: Multiply/Accumulate
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            mul_0_oc0_Acc <= 0;
            mul_1_oc0_Acc <= 0;
            mul_2_oc0_Acc <= 0;
            mul_0_oc1_Acc <= 0;
            mul_1_oc1_Acc <= 0;
            mul_2_oc1_Acc <= 0;
            mul_0_oc2_Acc <= 0;
            mul_1_oc2_Acc <= 0;
            mul_2_oc2_Acc <= 0;
            mul_0_oc3_Acc <= 0;
            mul_1_oc3_Acc <= 0;
            mul_2_oc3_Acc <= 0;
            RES_write_en_Acc <= 0;
            reset_acc_Acc <= 0;
            bias_oc0_Acc <= 0;
            res_addr_0_Acc <= 0;
            bias_oc1_Acc <= 0;
            res_addr_1_Acc <= 0;
            bias_oc2_Acc <= 0;
            res_addr_2_Acc <= 0;
            bias_oc3_Acc <= 0;
            res_addr_3_Acc <= 0;
        end
        else begin
            mul_0_oc0_Acc <= mul_0_oc0_Mul;
            mul_1_oc0_Acc <= mul_1_oc0_Mul;
            mul_2_oc0_Acc <= mul_2_oc0_Mul;
            mul_0_oc1_Acc <= mul_0_oc1_Mul;
            mul_1_oc1_Acc <= mul_1_oc1_Mul;
            mul_2_oc1_Acc <= mul_2_oc1_Mul;
            mul_0_oc2_Acc <= mul_0_oc2_Mul;
            mul_1_oc2_Acc <= mul_1_oc2_Mul;
            mul_2_oc2_Acc <= mul_2_oc2_Mul;
            mul_0_oc3_Acc <= mul_0_oc3_Mul;
            mul_1_oc3_Acc <= mul_1_oc3_Mul;
            mul_2_oc3_Acc <= mul_2_oc3_Mul;
            RES_write_en_Acc <= RES_write_en_Mul;
            reset_acc_Acc <= reset_acc_Mul;
            bias_oc0_Acc <= C_read_data_Mul_oc0;
            res_addr_0_Acc <= res_addr_0_Mul;
            bias_oc1_Acc <= C_read_data_Mul_oc1;
            res_addr_1_Acc <= res_addr_1_Mul;
            bias_oc2_Acc <= C_read_data_Mul_oc2;
            res_addr_2_Acc <= res_addr_2_Mul;
            bias_oc3_Acc <= C_read_data_Mul_oc3;
            res_addr_3_Acc <= res_addr_3_Mul;
        end
    end

    // Stage 3: Accumulate
    // Comb Logic: adder trees and temporal accumulators
    assign spatial_sum_oc0 = $signed(mul_0_oc0_Acc) + $signed(mul_1_oc0_Acc) + $signed(mul_2_oc0_Acc);
    assign spatial_sum_oc1 = $signed(mul_0_oc1_Acc) + $signed(mul_1_oc1_Acc) + $signed(mul_2_oc1_Acc);
    assign spatial_sum_oc2 = $signed(mul_0_oc2_Acc) + $signed(mul_1_oc2_Acc) + $signed(mul_2_oc2_Acc);
    assign spatial_sum_oc3 = $signed(mul_0_oc3_Acc) + $signed(mul_1_oc3_Acc) + $signed(mul_2_oc3_Acc);
    assign res_Acc_oc0 = reset_acc_Acc ? spatial_sum_oc0 : $signed(spatial_sum_oc0) + $signed(accumulator_oc0_Bias);
    assign res_Acc_oc1 = reset_acc_Acc ? spatial_sum_oc1 : $signed(spatial_sum_oc1) + $signed(accumulator_oc1_Bias);
    assign res_Acc_oc2 = reset_acc_Acc ? spatial_sum_oc2 : $signed(spatial_sum_oc2) + $signed(accumulator_oc2_Bias);
    assign res_Acc_oc3 = reset_acc_Acc ? spatial_sum_oc3 : $signed(spatial_sum_oc3) + $signed(accumulator_oc3_Bias);

    // Pipeline Register 3: Accumulate/Bias
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            accumulator_oc0_Bias <= 0;
            bias_oc0_Bias <= 0;
            addr_0_Bias <= 0;
            accumulator_oc1_Bias <= 0;
            bias_oc1_Bias <= 0;
            addr_1_Bias <= 0;
            accumulator_oc2_Bias <= 0;
            bias_oc2_Bias <= 0;
            addr_2_Bias <= 0;
            accumulator_oc3_Bias <= 0;
            bias_oc3_Bias <= 0;
            addr_3_Bias <= 0;
            RES_write_en_Bias <= 0;
        end
        else begin
            accumulator_oc0_Bias <= res_Acc_oc0;
            bias_oc0_Bias <= bias_oc0_Acc;
            addr_0_Bias <= res_addr_0_Acc;
            accumulator_oc1_Bias <= res_Acc_oc1;
            bias_oc1_Bias <= bias_oc1_Acc;
            addr_1_Bias <= res_addr_1_Acc;
            accumulator_oc2_Bias <= res_Acc_oc2;
            bias_oc2_Bias <= bias_oc2_Acc;
            addr_2_Bias <= res_addr_2_Acc;
            accumulator_oc3_Bias <= res_Acc_oc3;
            bias_oc3_Bias <= bias_oc3_Acc;
            addr_3_Bias <= res_addr_3_Acc;
            RES_write_en_Bias <= RES_write_en_Acc;
        end
    end

    // Stage 4: Bias
    // Comb Logic: align the Q6.10 bias, add it to the accumulator, then truncate back to Q6.10.
    assign aligned_bias_oc0 = $signed(bias_oc0_Bias) << FRAC_BITS;
    assign aligned_bias_oc1 = $signed(bias_oc1_Bias) << FRAC_BITS;
    assign aligned_bias_oc2 = $signed(bias_oc2_Bias) << FRAC_BITS;
    assign aligned_bias_oc3 = $signed(bias_oc3_Bias) << FRAC_BITS;
    assign res_biased_oc0 = $signed(accumulator_oc0_Bias) + aligned_bias_oc0;
    assign res_biased_oc1 = $signed(accumulator_oc1_Bias) + aligned_bias_oc1;
    assign res_biased_oc2 = $signed(accumulator_oc2_Bias) + aligned_bias_oc2;
    assign res_biased_oc3 = $signed(accumulator_oc3_Bias) + aligned_bias_oc3;
    assign trunc_oc0 = res_biased_oc0[width+FRAC_BITS-1:FRAC_BITS];
    assign trunc_oc1 = res_biased_oc1[width+FRAC_BITS-1:FRAC_BITS];
    assign trunc_oc2 = res_biased_oc2[width+FRAC_BITS-1:FRAC_BITS];
    assign trunc_oc3 = res_biased_oc3[width+FRAC_BITS-1:FRAC_BITS];

    // Pipeline Register 4: Bias/ReLU
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Relu <= 0;
            data_0_Relu <= 0;
            addr_0_Relu <= 0;
            data_1_Relu <= 0;
            addr_1_Relu <= 0;
            data_2_Relu <= 0;
            addr_2_Relu <= 0;
            data_3_Relu <= 0;
            addr_3_Relu <= 0;
        end
        else begin
            RES_write_en_Relu <= RES_write_en_Bias;
            data_0_Relu <= trunc_oc0;
            addr_0_Relu <= addr_0_Bias;
            data_1_Relu <= trunc_oc1;
            addr_1_Relu <= addr_1_Bias;
            data_2_Relu <= trunc_oc2;
            addr_2_Relu <= addr_2_Bias;
            data_3_Relu <= trunc_oc3;
            addr_3_Relu <= addr_3_Bias;
        end
    end

    // Stage 5: ReLU
    // Comb Logic: the sign bit selects the registered value or zero.
    assign relu_oc0 = data_0_Relu[width-1] ? {width{1'b0}} : data_0_Relu;
    assign relu_oc1 = data_1_Relu[width-1] ? {width{1'b0}} : data_1_Relu;
    assign relu_oc2 = data_2_Relu[width-1] ? {width{1'b0}} : data_2_Relu;
    assign relu_oc3 = data_3_Relu[width-1] ? {width{1'b0}} : data_3_Relu;

    // Pipeline Register 5: ReLU/Output1 (pair 0/1)
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Out1 <= 0;
            data_0_Out1 <= 0;
            addr_0_Out1 <= 0;
            data_1_Out1 <= 0;
            addr_1_Out1 <= 0;
            data_2_Out1 <= 0;
            addr_2_Out1 <= 0;
            data_3_Out1 <= 0;
            addr_3_Out1 <= 0;
        end
        else begin
            RES_write_en_Out1 <= RES_write_en_Relu;
            data_0_Out1 <= relu_oc0;
            addr_0_Out1 <= addr_0_Relu;
            data_1_Out1 <= relu_oc1;
            addr_1_Out1 <= addr_1_Relu;
            data_2_Out1 <= relu_oc2;
            addr_2_Out1 <= addr_2_Relu;
            data_3_Out1 <= relu_oc3;
            addr_3_Out1 <= addr_3_Relu;
        end
    end

    // Pipeline Register 6: Output1/Output2 (pair 2/3)
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Out2 <= 0;
            data_2_Out2 <= 0;
            addr_2_Out2 <= 0;
            data_3_Out2 <= 0;
            addr_3_Out2 <= 0;
        end
        else begin
            RES_write_en_Out2 <= RES_write_en_Out1;
            data_2_Out2 <= data_2_Out1;
            addr_2_Out2 <= addr_2_Out1;
            data_3_Out2 <= data_3_Out1;
            addr_3_Out2 <= addr_3_Out1;
        end
    end

    // Output Memory muxes
    // Serialize two output-channel pairs into the two producer write ports.
    assign RES_write_en_even = RES_write_en_Out1 | RES_write_en_Out2;
    assign RES_write_en_odd = RES_write_en_even;
    assign RES_write_address_even = RES_write_en_Out1 ? addr_0_Out1 : addr_2_Out2;
    assign RES_write_address_odd = RES_write_en_Out1 ? addr_1_Out1 : addr_3_Out2;
    assign RES_write_data_in_even = RES_write_en_Out1 ? data_0_Out1 : data_2_Out2;
    assign RES_write_data_in_odd = RES_write_en_Out1 ? data_1_Out1 : data_3_Out2;

    // Coordinate & Loop Counters
    // Kernel Column Counter
    Counter #(
                .ADDR_WIDTH($clog2(K)),
                .CNT_INIT(0),
                .MAX_VAL(K - 1)
            ) k_col_counter (
                .CLK(clk),
                .RESET(!rst || counter_clear_Read1),
                .ENABLE(en_k_col),
                .CNT(k_col)
            );

    // Kernel Row Counter
    Counter #(
                .ADDR_WIDTH($clog2(K)),
                .CNT_INIT(0),
                .MAX_VAL(K - 1)
            ) k_row_counter (
                .CLK(clk),
                .RESET(!rst || counter_clear_Read1),
                .ENABLE(en_k_row),
                .CNT(k_row)
            );

    // Image Column (Width) Counter
    Counter #(
                .ADDR_WIDTH($clog2(IN_W)),
                .CNT_INIT(0),
                .MAX_VAL(IN_W - 1)
            ) col_counter (
                .CLK(clk),
                .RESET(!rst || counter_clear_Read1),
                .ENABLE(en_col),
                .CNT(col)
            );

    // Image Row (Height) Counter
    Counter #(
                .ADDR_WIDTH($clog2(IN_H)),
                .CNT_INIT(0),
                .MAX_VAL(IN_H - 1)
            ) row_counter (
                .CLK(clk),
                .RESET(!rst || counter_clear_Read1),
                .ENABLE(en_row),
                .CNT(row)
            );

    // Output Channel Group Counter (Outermost)
    Counter #(
                .ADDR_WIDTH($clog2(OC_DIM / OUT_UNROLL)),
                .CNT_INIT(0),
                .MAX_VAL((OC_DIM / OUT_UNROLL) - 1)
            ) oc_counter (
                .CLK(clk),
                .RESET(!rst || counter_clear_Read1),
                .ENABLE(en_oc),
                .CNT(oc)
            );

    // Address Generation Units (AGUs)
    // Input memory address generator
    conv1_input_agu #(
                        .ADDR_WIDTH(A_depth_bits),
                        .IN_H(IN_H),
                        .IN_W(IN_W),
                        .K(K),
                        .PAD(1)
                    ) image_address_gen (
                        .row(row),
                        .col(col),
                        .k_row(k_row),
                        .k_col(k_col),
                        .A_read_address(A_read_address_Read1),
                        .out_of_bounds(out_of_bounds_Read1)
                    );


    // Weights memory address generator
    conv1_weight_agu #(
                         .ADDR_WIDTH(B_depth_bits),
                         .OC_DIM(OC_DIM),
                         .K(K),
                         .OUTPUT_UNROLL(OUT_UNROLL)
                     ) weight_address_gen (
                         .oc(oc),
                         .k_row(k_row),
                         .k_col(k_col),
                         .B_read_address(B_read_address_Read1)
                     );


    // Output RAM Address generator (Unrolled for 4 Output Channels)
    conv1_res_agu #(
                      .ADDR_WIDTH(RES_depth_bits),
                      .OC_DIM(OC_DIM),
                      .IN_H(IN_H),
                      .IN_W(IN_W),
                      .OUTPUT_UNROLL(OUT_UNROLL)
                  ) res_address_gen (
                      .oc(oc),
                      .row(row),
                      .col(col),
                      .RES_write_address_0(calc_addr_0_Read1),
                      .RES_write_address_1(calc_addr_1_Read1),
                      .RES_write_address_2(calc_addr_2_Read1),
                      .RES_write_address_3(calc_addr_3_Read1)
                  );

    assign C_read_address_Read1 = oc;

endmodule

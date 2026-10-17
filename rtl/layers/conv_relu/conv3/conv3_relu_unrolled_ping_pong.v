`timescale 1ns / 1ps

/* ----------------------------------------------------------------------------------
--  RTL Implementation of Fused Convolution Layer 3 + ReLU
--  Input-channel unroll factor: 32 (all input channels processed in parallel)
--  Output-channel unroll factor: 2 (two output channels processed in parallel)
--  The 3x3 kernel positions are processed temporally, one position per cycle.
--  ReLU is applied after bias addition and truncation, before output storage.
--  Packed input URAM and packed weight BRAM both have three-cycle reads.
--  The two results directly drive the two downstream TDP write ports.
----------------------------------------------------------------------------------
*/

module conv3_relu_unrolled_ping_pong #(
        parameter width = 16,
        parameter FRAC_BITS = 10,
        parameter IC_DIM = 32,
        parameter OC_DIM = 64,
        parameter OUT_UNROLL = 2,
        parameter IN_H = 16,
        parameter IN_W = 16,
        parameter K = 3,
        parameter A_depth_bits = 8,
        parameter B_depth_bits = 9,
        parameter C_depth_bits = 5,
        parameter RES_depth_bits = 14
    ) (
        input clk,
        input rst,
        input consumer_ready,
        input producer_ready,

        // Shared read control and output status
        output reg consumer_frame_done,
        output reg producer_frame_done,
        output reg A_read_en,
        output reg [A_depth_bits-1:0] A_read_address,
        output reg B_read_en,
        output reg [B_depth_bits-1:0] B_read_address,
        output reg C_read_en,
        output reg [C_depth_bits-1:0] C_read_address,

        // Input feature map (32 channels)
        input [width-1:0] A_read_data_out_0,
        input [width-1:0] A_read_data_out_1,
        input [width-1:0] A_read_data_out_2,
        input [width-1:0] A_read_data_out_3,
        input [width-1:0] A_read_data_out_4,
        input [width-1:0] A_read_data_out_5,
        input [width-1:0] A_read_data_out_6,
        input [width-1:0] A_read_data_out_7,
        input [width-1:0] A_read_data_out_8,
        input [width-1:0] A_read_data_out_9,
        input [width-1:0] A_read_data_out_10,
        input [width-1:0] A_read_data_out_11,
        input [width-1:0] A_read_data_out_12,
        input [width-1:0] A_read_data_out_13,
        input [width-1:0] A_read_data_out_14,
        input [width-1:0] A_read_data_out_15,
        input [width-1:0] A_read_data_out_16,
        input [width-1:0] A_read_data_out_17,
        input [width-1:0] A_read_data_out_18,
        input [width-1:0] A_read_data_out_19,
        input [width-1:0] A_read_data_out_20,
        input [width-1:0] A_read_data_out_21,
        input [width-1:0] A_read_data_out_22,
        input [width-1:0] A_read_data_out_23,
        input [width-1:0] A_read_data_out_24,
        input [width-1:0] A_read_data_out_25,
        input [width-1:0] A_read_data_out_26,
        input [width-1:0] A_read_data_out_27,
        input [width-1:0] A_read_data_out_28,
        input [width-1:0] A_read_data_out_29,
        input [width-1:0] A_read_data_out_30,
        input [width-1:0] A_read_data_out_31,

        // Output channel 0 weights (32 input channels)
        input [width-1:0] B_read_data_out_0_oc0,
        input [width-1:0] B_read_data_out_1_oc0,
        input [width-1:0] B_read_data_out_2_oc0,
        input [width-1:0] B_read_data_out_3_oc0,
        input [width-1:0] B_read_data_out_4_oc0,
        input [width-1:0] B_read_data_out_5_oc0,
        input [width-1:0] B_read_data_out_6_oc0,
        input [width-1:0] B_read_data_out_7_oc0,
        input [width-1:0] B_read_data_out_8_oc0,
        input [width-1:0] B_read_data_out_9_oc0,
        input [width-1:0] B_read_data_out_10_oc0,
        input [width-1:0] B_read_data_out_11_oc0,
        input [width-1:0] B_read_data_out_12_oc0,
        input [width-1:0] B_read_data_out_13_oc0,
        input [width-1:0] B_read_data_out_14_oc0,
        input [width-1:0] B_read_data_out_15_oc0,
        input [width-1:0] B_read_data_out_16_oc0,
        input [width-1:0] B_read_data_out_17_oc0,
        input [width-1:0] B_read_data_out_18_oc0,
        input [width-1:0] B_read_data_out_19_oc0,
        input [width-1:0] B_read_data_out_20_oc0,
        input [width-1:0] B_read_data_out_21_oc0,
        input [width-1:0] B_read_data_out_22_oc0,
        input [width-1:0] B_read_data_out_23_oc0,
        input [width-1:0] B_read_data_out_24_oc0,
        input [width-1:0] B_read_data_out_25_oc0,
        input [width-1:0] B_read_data_out_26_oc0,
        input [width-1:0] B_read_data_out_27_oc0,
        input [width-1:0] B_read_data_out_28_oc0,
        input [width-1:0] B_read_data_out_29_oc0,
        input [width-1:0] B_read_data_out_30_oc0,
        input [width-1:0] B_read_data_out_31_oc0,

        // Output channel 1 weights (32 input channels)
        input [width-1:0] B_read_data_out_0_oc1,
        input [width-1:0] B_read_data_out_1_oc1,
        input [width-1:0] B_read_data_out_2_oc1,
        input [width-1:0] B_read_data_out_3_oc1,
        input [width-1:0] B_read_data_out_4_oc1,
        input [width-1:0] B_read_data_out_5_oc1,
        input [width-1:0] B_read_data_out_6_oc1,
        input [width-1:0] B_read_data_out_7_oc1,
        input [width-1:0] B_read_data_out_8_oc1,
        input [width-1:0] B_read_data_out_9_oc1,
        input [width-1:0] B_read_data_out_10_oc1,
        input [width-1:0] B_read_data_out_11_oc1,
        input [width-1:0] B_read_data_out_12_oc1,
        input [width-1:0] B_read_data_out_13_oc1,
        input [width-1:0] B_read_data_out_14_oc1,
        input [width-1:0] B_read_data_out_15_oc1,
        input [width-1:0] B_read_data_out_16_oc1,
        input [width-1:0] B_read_data_out_17_oc1,
        input [width-1:0] B_read_data_out_18_oc1,
        input [width-1:0] B_read_data_out_19_oc1,
        input [width-1:0] B_read_data_out_20_oc1,
        input [width-1:0] B_read_data_out_21_oc1,
        input [width-1:0] B_read_data_out_22_oc1,
        input [width-1:0] B_read_data_out_23_oc1,
        input [width-1:0] B_read_data_out_24_oc1,
        input [width-1:0] B_read_data_out_25_oc1,
        input [width-1:0] B_read_data_out_26_oc1,
        input [width-1:0] B_read_data_out_27_oc1,
        input [width-1:0] B_read_data_out_28_oc1,
        input [width-1:0] B_read_data_out_29_oc1,
        input [width-1:0] B_read_data_out_30_oc1,
        input [width-1:0] B_read_data_out_31_oc1,

        // Bias inputs
        input [width-1:0] C_read_data_out_oc0,
        input [width-1:0] C_read_data_out_oc1,

        // Output RAM interface
        output RES_write_en_even,
        output [RES_depth_bits-1:0] RES_write_address_even,
        output [width-1:0] RES_write_data_in_even,
        output RES_write_en_odd,
        output [RES_depth_bits-1:0] RES_write_address_odd,
        output [width-1:0] RES_write_data_in_odd
    );


    localparam ACC_WIDTH = 40, NUMBER_OF_OUTPUT_WORDS = OC_DIM*IN_H*IN_W;
    localparam IDLE = 4'b1000, COMPUTE = 4'b0100, FLUSH = 4'b0010, FINISH = 4'b0001;
    reg [3:0] state = IDLE, n_state = IDLE;
    wire [$clog2(OC_DIM/OUT_UNROLL)-1:0] oc;
    wire [$clog2(IN_H)-1:0] row;
    wire [$clog2(IN_W)-1:0] col;
    wire [$clog2(K)-1:0] k_row, k_col;
    reg en_k_col, en_k_row, en_col, en_row, en_oc;
    wire signed [width-1:0] A_bus[0:IC_DIM-1];
    // All datapath arrays follow OUT_UNROLL so no inactive lanes are elaborated.
    wire signed [width-1:0] B_bus[0:OUT_UNROLL-1][0:IC_DIM-1];
    wire signed [width-1:0] C_bus[0:OUT_UNROLL-1];
    assign A_bus[0] = A_read_data_out_0;
    assign A_bus[1] = A_read_data_out_1;
    assign A_bus[2] = A_read_data_out_2;
    assign A_bus[3] = A_read_data_out_3;
    assign A_bus[4] = A_read_data_out_4;
    assign A_bus[5] = A_read_data_out_5;
    assign A_bus[6] = A_read_data_out_6;
    assign A_bus[7] = A_read_data_out_7;
    assign A_bus[8] = A_read_data_out_8;
    assign A_bus[9] = A_read_data_out_9;
    assign A_bus[10] = A_read_data_out_10;
    assign A_bus[11] = A_read_data_out_11;
    assign A_bus[12] = A_read_data_out_12;
    assign A_bus[13] = A_read_data_out_13;
    assign A_bus[14] = A_read_data_out_14;
    assign A_bus[15] = A_read_data_out_15;
    assign A_bus[16] = A_read_data_out_16;
    assign A_bus[17] = A_read_data_out_17;
    assign A_bus[18] = A_read_data_out_18;
    assign A_bus[19] = A_read_data_out_19;
    assign A_bus[20] = A_read_data_out_20;
    assign A_bus[21] = A_read_data_out_21;
    assign A_bus[22] = A_read_data_out_22;
    assign A_bus[23] = A_read_data_out_23;
    assign A_bus[24] = A_read_data_out_24;
    assign A_bus[25] = A_read_data_out_25;
    assign A_bus[26] = A_read_data_out_26;
    assign A_bus[27] = A_read_data_out_27;
    assign A_bus[28] = A_read_data_out_28;
    assign A_bus[29] = A_read_data_out_29;
    assign A_bus[30] = A_read_data_out_30;
    assign A_bus[31] = A_read_data_out_31;
    assign B_bus[0][0] = B_read_data_out_0_oc0;
    assign B_bus[0][1] = B_read_data_out_1_oc0;
    assign B_bus[0][2] = B_read_data_out_2_oc0;
    assign B_bus[0][3] = B_read_data_out_3_oc0;
    assign B_bus[0][4] = B_read_data_out_4_oc0;
    assign B_bus[0][5] = B_read_data_out_5_oc0;
    assign B_bus[0][6] = B_read_data_out_6_oc0;
    assign B_bus[0][7] = B_read_data_out_7_oc0;
    assign B_bus[0][8] = B_read_data_out_8_oc0;
    assign B_bus[0][9] = B_read_data_out_9_oc0;
    assign B_bus[0][10] = B_read_data_out_10_oc0;
    assign B_bus[0][11] = B_read_data_out_11_oc0;
    assign B_bus[0][12] = B_read_data_out_12_oc0;
    assign B_bus[0][13] = B_read_data_out_13_oc0;
    assign B_bus[0][14] = B_read_data_out_14_oc0;
    assign B_bus[0][15] = B_read_data_out_15_oc0;
    assign B_bus[0][16] = B_read_data_out_16_oc0;
    assign B_bus[0][17] = B_read_data_out_17_oc0;
    assign B_bus[0][18] = B_read_data_out_18_oc0;
    assign B_bus[0][19] = B_read_data_out_19_oc0;
    assign B_bus[0][20] = B_read_data_out_20_oc0;
    assign B_bus[0][21] = B_read_data_out_21_oc0;
    assign B_bus[0][22] = B_read_data_out_22_oc0;
    assign B_bus[0][23] = B_read_data_out_23_oc0;
    assign B_bus[0][24] = B_read_data_out_24_oc0;
    assign B_bus[0][25] = B_read_data_out_25_oc0;
    assign B_bus[0][26] = B_read_data_out_26_oc0;
    assign B_bus[0][27] = B_read_data_out_27_oc0;
    assign B_bus[0][28] = B_read_data_out_28_oc0;
    assign B_bus[0][29] = B_read_data_out_29_oc0;
    assign B_bus[0][30] = B_read_data_out_30_oc0;
    assign B_bus[0][31] = B_read_data_out_31_oc0;
    assign B_bus[1][0] = B_read_data_out_0_oc1;
    assign B_bus[1][1] = B_read_data_out_1_oc1;
    assign B_bus[1][2] = B_read_data_out_2_oc1;
    assign B_bus[1][3] = B_read_data_out_3_oc1;
    assign B_bus[1][4] = B_read_data_out_4_oc1;
    assign B_bus[1][5] = B_read_data_out_5_oc1;
    assign B_bus[1][6] = B_read_data_out_6_oc1;
    assign B_bus[1][7] = B_read_data_out_7_oc1;
    assign B_bus[1][8] = B_read_data_out_8_oc1;
    assign B_bus[1][9] = B_read_data_out_9_oc1;
    assign B_bus[1][10] = B_read_data_out_10_oc1;
    assign B_bus[1][11] = B_read_data_out_11_oc1;
    assign B_bus[1][12] = B_read_data_out_12_oc1;
    assign B_bus[1][13] = B_read_data_out_13_oc1;
    assign B_bus[1][14] = B_read_data_out_14_oc1;
    assign B_bus[1][15] = B_read_data_out_15_oc1;
    assign B_bus[1][16] = B_read_data_out_16_oc1;
    assign B_bus[1][17] = B_read_data_out_17_oc1;
    assign B_bus[1][18] = B_read_data_out_18_oc1;
    assign B_bus[1][19] = B_read_data_out_19_oc1;
    assign B_bus[1][20] = B_read_data_out_20_oc1;
    assign B_bus[1][21] = B_read_data_out_21_oc1;
    assign B_bus[1][22] = B_read_data_out_22_oc1;
    assign B_bus[1][23] = B_read_data_out_23_oc1;
    assign B_bus[1][24] = B_read_data_out_24_oc1;
    assign B_bus[1][25] = B_read_data_out_25_oc1;
    assign B_bus[1][26] = B_read_data_out_26_oc1;
    assign B_bus[1][27] = B_read_data_out_27_oc1;
    assign B_bus[1][28] = B_read_data_out_28_oc1;
    assign B_bus[1][29] = B_read_data_out_29_oc1;
    assign B_bus[1][30] = B_read_data_out_30_oc1;
    assign B_bus[1][31] = B_read_data_out_31_oc1;
    assign C_bus[0] = C_read_data_out_oc0;
    assign C_bus[1] = C_read_data_out_oc1;
    // -------------------------------------------------------------------------
    // Datapath Pipeline
    // -------------------------------------------------------------------------

    // Stage 1: Read1 - combinational AGU outputs and control
    wire [A_depth_bits-1:0] A_read_address_Read1;
    wire [B_depth_bits-1:0] B_read_address_Read1;
    wire [C_depth_bits-1:0] C_read_address_Read1;
    wire [RES_depth_bits-1:0] calc_addr_Read1[0:OUT_UNROLL-1];
    wire out_of_bounds_Read1;
    reg A_read_en_Read1, B_read_en_Read1, C_read_en_Read1, RES_write_en_Read1, counter_clear_Read1, reset_acc_Read1;
    // Stage 1.1: Read2 registers
    reg RES_write_en_Read2, reset_acc_Read2, out_of_bounds_Read2;
    reg [RES_depth_bits-1:0] calc_addr_Read2[0:OUT_UNROLL-1];
    // Stage 1.2: capture bias LUTRAM data while both synchronous memories read.
    reg RES_write_en_Read3, reset_acc_Read3, out_of_bounds_Read3;
    reg signed [width-1:0] C_Read3[0:OUT_UNROLL-1];
    reg [RES_depth_bits-1:0] addr_Read3[0:OUT_UNROLL-1];
    // Read4/Read5 keep bias, padding, and controls aligned with URAM/BRAM data.
    reg RES_write_en_Read4, reset_acc_Read4, out_of_bounds_Read4;
    reg signed [width-1:0] C_Read4[0:OUT_UNROLL-1];
    reg [RES_depth_bits-1:0] addr_Read4[0:OUT_UNROLL-1];
    reg RES_write_en_Read5, reset_acc_Read5, out_of_bounds_Read5;
    reg signed [width-1:0] C_Read5[0:OUT_UNROLL-1];
    reg [RES_depth_bits-1:0] addr_Read5[0:OUT_UNROLL-1];
    // Stage 2: Multiply registers and combinational products
    reg RES_write_en_Mul, reset_acc_Mul;
    reg signed [width-1:0] A_Mul[0:IC_DIM-1];
    reg signed [width-1:0] B_Mul[0:OUT_UNROLL-1][0:IC_DIM-1];
    reg signed [width-1:0] C_Mul[0:OUT_UNROLL-1];
    reg [RES_depth_bits-1:0] addr_Mul[0:OUT_UNROLL-1];
    wire signed [2*width-1:0] product_Mul[0:OUT_UNROLL-1][0:IC_DIM-1];
    // Stage 3: Acc1 registers all products
    reg signed [2*width-1:0] product_Acc1[0:OUT_UNROLL-1][0:IC_DIM-1];
    reg [width-1:0] bias_Acc1[0:OUT_UNROLL-1];
    reg [RES_depth_bits-1:0] addr_Acc1[0:OUT_UNROLL-1];
    reg RES_write_en_Acc1, reset_acc_Acc1;
    // Stage Acc2: 16 registered adder-tree terms
    wire signed [ACC_WIDTH-1:0] sum_Acc2_comb[0:OUT_UNROLL-1][0:15];
    reg signed [ACC_WIDTH-1:0] sum_Acc2[0:OUT_UNROLL-1][0:15];
    reg [width-1:0] bias_Acc2[0:OUT_UNROLL-1];
    reg [RES_depth_bits-1:0] addr_Acc2[0:OUT_UNROLL-1];
    reg RES_write_en_Acc2, reset_acc_Acc2;
    // Stage Acc3: 8 registered adder-tree terms
    wire signed [ACC_WIDTH-1:0] sum_Acc3_comb[0:OUT_UNROLL-1][0:7];
    reg signed [ACC_WIDTH-1:0] sum_Acc3[0:OUT_UNROLL-1][0:7];
    reg [width-1:0] bias_Acc3[0:OUT_UNROLL-1];
    reg [RES_depth_bits-1:0] addr_Acc3[0:OUT_UNROLL-1];
    reg RES_write_en_Acc3, reset_acc_Acc3;
    // Stage Acc4: 4 registered adder-tree terms
    wire signed [ACC_WIDTH-1:0] sum_Acc4_comb[0:OUT_UNROLL-1][0:3];
    reg signed [ACC_WIDTH-1:0] sum_Acc4[0:OUT_UNROLL-1][0:3];
    reg [width-1:0] bias_Acc4[0:OUT_UNROLL-1];
    reg [RES_depth_bits-1:0] addr_Acc4[0:OUT_UNROLL-1];
    reg RES_write_en_Acc4, reset_acc_Acc4;
    // Stage Acc5: 2 registered adder-tree terms
    wire signed [ACC_WIDTH-1:0] sum_Acc5_comb[0:OUT_UNROLL-1][0:1];
    reg signed [ACC_WIDTH-1:0] sum_Acc5[0:OUT_UNROLL-1][0:1];
    reg [width-1:0] bias_Acc5[0:OUT_UNROLL-1];
    reg [RES_depth_bits-1:0] addr_Acc5[0:OUT_UNROLL-1];
    reg RES_write_en_Acc5, reset_acc_Acc5;
    // Stage Acc6: 1 registered adder-tree terms
    wire signed [ACC_WIDTH-1:0] sum_Acc6_comb[0:OUT_UNROLL-1][0:0];
    reg signed [ACC_WIDTH-1:0] sum_Acc6[0:OUT_UNROLL-1][0:0];
    reg [width-1:0] bias_Acc6[0:OUT_UNROLL-1];
    reg [RES_depth_bits-1:0] addr_Acc6[0:OUT_UNROLL-1];
    reg RES_write_en_Acc6, reset_acc_Acc6;
    // Temporal kernel accumulation and bias/ReLU stages
    reg signed [ACC_WIDTH-1:0] accumulator_KAcc[0:OUT_UNROLL-1];
    wire signed [ACC_WIDTH-1:0] accumulator_next[0:OUT_UNROLL-1];
    reg [width-1:0] bias_KAcc[0:OUT_UNROLL-1];
    reg [RES_depth_bits-1:0] addr_KAcc[0:OUT_UNROLL-1];
    reg RES_write_en_KAcc;
    wire signed [ACC_WIDTH-1:0] aligned_bias[0:OUT_UNROLL-1], biased_value[0:OUT_UNROLL-1];
    wire [width-1:0] truncated_value[0:OUT_UNROLL-1];
    reg [width-1:0] data_Relu[0:OUT_UNROLL-1];
    reg [RES_depth_bits-1:0] addr_Relu[0:OUT_UNROLL-1];
    reg RES_write_en_Relu;
    wire [width-1:0] relu_value[0:OUT_UNROLL-1];
    // One registered output stage directly supplies both TDP write ports.
    reg RES_write_en_Out1;
    reg [width-1:0] data_Out1[0:OUT_UNROLL-1];
    reg [RES_depth_bits-1:0] addr_Out1[0:OUT_UNROLL-1];
    wire [RES_depth_bits-1:0] final_output_address = addr_Out1[OUT_UNROLL-1];
    integer lane, channel, term;
    always @(posedge clk) state <= n_state;
    // Next-State and Counter-Enable Combinational Logic
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
                RES_write_en_Read1 = (k_row == K - 1) && (k_col == K - 1);
                en_k_col = 1;
                en_k_row = en_k_col && (k_col == K - 1);
                en_col = en_k_row && (k_row == K - 1);
                en_row = en_col && (col == IN_W - 1);
                en_oc = en_row && (row == IN_H - 1);

                if (en_oc && (oc == (OC_DIM / OUT_UNROLL) - 1))
                    n_state = FLUSH;
            end

            FLUSH:
                if (RES_write_en_Out1 && (final_output_address == NUMBER_OF_OUTPUT_WORDS - 1))
                    n_state = FINISH;

            FINISH: begin
                consumer_frame_done = 1;
                producer_frame_done = 1;
                n_state = IDLE;
            end
        endcase
    end
    // Pipeline Register: Read1/Read2
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            A_read_en <= 0;
            B_read_en <= 0;
            C_read_en <= 0;
            A_read_address <= 0;
            B_read_address <= 0;
            C_read_address <= 0;
            RES_write_en_Read2 <= 0;
            reset_acc_Read2 <= 0;
            out_of_bounds_Read2 <= 0;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1)
                calc_addr_Read2[lane] <= 0;
        end
        else begin
            A_read_en <= A_read_en_Read1;
            B_read_en <= B_read_en_Read1;
            C_read_en <= C_read_en_Read1;
            A_read_address <= A_read_address_Read1;
            B_read_address <= B_read_address_Read1;
            C_read_address <= C_read_address_Read1;
            RES_write_en_Read2 <= RES_write_en_Read1;
            reset_acc_Read2 <= reset_acc_Read1;
            out_of_bounds_Read2 <= out_of_bounds_Read1;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1)
                calc_addr_Read2[lane] <= calc_addr_Read1[lane];
        end
    end
    // Pipeline Register: Read2/Read3
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Read3 <= 0;
            reset_acc_Read3 <= 0;
            out_of_bounds_Read3 <= 0;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                C_Read3[lane] <= 0;
                addr_Read3[lane] <= 0;
            end
        end
        else begin
            RES_write_en_Read3 <= RES_write_en_Read2;
            reset_acc_Read3 <= reset_acc_Read2;
            out_of_bounds_Read3 <= out_of_bounds_Read2;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                C_Read3[lane] <= C_bus[lane];
                addr_Read3[lane] <= calc_addr_Read2[lane];
            end
        end
    end

    // Pipeline Register: Read3/Read4
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Read4 <= 0;
            reset_acc_Read4 <= 0;
            out_of_bounds_Read4 <= 0;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                C_Read4[lane] <= 0;
                addr_Read4[lane] <= 0;
            end
        end
        else begin
            RES_write_en_Read4 <= RES_write_en_Read3;
            reset_acc_Read4 <= reset_acc_Read3;
            out_of_bounds_Read4 <= out_of_bounds_Read3;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                C_Read4[lane] <= C_Read3[lane];
                addr_Read4[lane] <= addr_Read3[lane];
            end
        end
    end

    // Pipeline Register: Read4/Read5
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Read5 <= 0;
            reset_acc_Read5 <= 0;
            out_of_bounds_Read5 <= 0;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                C_Read5[lane] <= 0;
                addr_Read5[lane] <= 0;
            end
        end
        else begin
            RES_write_en_Read5 <= RES_write_en_Read4;
            reset_acc_Read5 <= reset_acc_Read4;
            out_of_bounds_Read5 <= out_of_bounds_Read4;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                C_Read5[lane] <= C_Read4[lane];
                addr_Read5[lane] <= addr_Read4[lane];
            end
        end
    end

    // Pipeline Register: Read5/Multiply. Packed input URAM and weight BRAM
    // outputs now correspond to the same delayed address/control tuple.
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Mul <= 0;
            reset_acc_Mul <= 0;
            for (channel = 0; channel < IC_DIM; channel = channel + 1)
                A_Mul[channel] <= 0;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                C_Mul[lane] <= 0;
                addr_Mul[lane] <= 0;
                for (channel = 0; channel < IC_DIM; channel = channel + 1)
                    B_Mul[lane][channel] <= 0;
            end
        end
        else begin
            RES_write_en_Mul <= RES_write_en_Read5;
            reset_acc_Mul <= reset_acc_Read5;
            for (channel = 0; channel < IC_DIM; channel = channel + 1)
                A_Mul[channel] <= out_of_bounds_Read5 ? 0 : A_bus[channel];
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                C_Mul[lane] <= C_Read5[lane];
                addr_Mul[lane] <= addr_Read5[lane];
                for (channel = 0; channel < IC_DIM; channel = channel + 1)
                    B_Mul[lane][channel] <= B_bus[lane][channel];
            end
        end
    end
    // Multiply combinational logic
    genvar gm, gn;
generate for (gm = 0; gm < OUT_UNROLL; gm = gm + 1) begin:MUL_LANES for (gn = 0; gn < IC_DIM; gn = gn + 1)begin:MUL_INPUTS
                assign product_Mul[gm][gn] = $signed(A_Mul[gn])*$signed(B_Mul[gm][gn]);
            end
        end
    endgenerate
    // Pipeline Register: Multiply/Acc1
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Acc1 <= 0;
            reset_acc_Acc1 <= 0;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                bias_Acc1[lane] <= 0;
                addr_Acc1[lane] <= 0;
                for (channel = 0; channel < IC_DIM; channel = channel + 1)
                    product_Acc1[lane][channel] <= 0;
            end
        end
        else begin
            RES_write_en_Acc1 <= RES_write_en_Mul;
            reset_acc_Acc1 <= reset_acc_Mul;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                bias_Acc1[lane] <= C_Mul[lane];
                addr_Acc1[lane] <= addr_Mul[lane];
                for (channel = 0; channel < IC_DIM; channel = channel + 1)
                    product_Acc1[lane][channel] <= product_Mul[lane][channel];
            end
        end
    end
    // Stage Acc2: combinational pairwise additions
    assign sum_Acc2_comb[0][0] = $signed(product_Acc1[0][0]) + $signed(product_Acc1[0][1]);
    assign sum_Acc2_comb[0][1] = $signed(product_Acc1[0][2]) + $signed(product_Acc1[0][3]);
    assign sum_Acc2_comb[0][2] = $signed(product_Acc1[0][4]) + $signed(product_Acc1[0][5]);
    assign sum_Acc2_comb[0][3] = $signed(product_Acc1[0][6]) + $signed(product_Acc1[0][7]);
    assign sum_Acc2_comb[0][4] = $signed(product_Acc1[0][8]) + $signed(product_Acc1[0][9]);
    assign sum_Acc2_comb[0][5] = $signed(product_Acc1[0][10]) + $signed(product_Acc1[0][11]);
    assign sum_Acc2_comb[0][6] = $signed(product_Acc1[0][12]) + $signed(product_Acc1[0][13]);
    assign sum_Acc2_comb[0][7] = $signed(product_Acc1[0][14]) + $signed(product_Acc1[0][15]);
    assign sum_Acc2_comb[0][8] = $signed(product_Acc1[0][16]) + $signed(product_Acc1[0][17]);
    assign sum_Acc2_comb[0][9] = $signed(product_Acc1[0][18]) + $signed(product_Acc1[0][19]);
    assign sum_Acc2_comb[0][10] = $signed(product_Acc1[0][20]) + $signed(product_Acc1[0][21]);
    assign sum_Acc2_comb[0][11] = $signed(product_Acc1[0][22]) + $signed(product_Acc1[0][23]);
    assign sum_Acc2_comb[0][12] = $signed(product_Acc1[0][24]) + $signed(product_Acc1[0][25]);
    assign sum_Acc2_comb[0][13] = $signed(product_Acc1[0][26]) + $signed(product_Acc1[0][27]);
    assign sum_Acc2_comb[0][14] = $signed(product_Acc1[0][28]) + $signed(product_Acc1[0][29]);
    assign sum_Acc2_comb[0][15] = $signed(product_Acc1[0][30]) + $signed(product_Acc1[0][31]);
    assign sum_Acc2_comb[1][0] = $signed(product_Acc1[1][0]) + $signed(product_Acc1[1][1]);
    assign sum_Acc2_comb[1][1] = $signed(product_Acc1[1][2]) + $signed(product_Acc1[1][3]);
    assign sum_Acc2_comb[1][2] = $signed(product_Acc1[1][4]) + $signed(product_Acc1[1][5]);
    assign sum_Acc2_comb[1][3] = $signed(product_Acc1[1][6]) + $signed(product_Acc1[1][7]);
    assign sum_Acc2_comb[1][4] = $signed(product_Acc1[1][8]) + $signed(product_Acc1[1][9]);
    assign sum_Acc2_comb[1][5] = $signed(product_Acc1[1][10]) + $signed(product_Acc1[1][11]);
    assign sum_Acc2_comb[1][6] = $signed(product_Acc1[1][12]) + $signed(product_Acc1[1][13]);
    assign sum_Acc2_comb[1][7] = $signed(product_Acc1[1][14]) + $signed(product_Acc1[1][15]);
    assign sum_Acc2_comb[1][8] = $signed(product_Acc1[1][16]) + $signed(product_Acc1[1][17]);
    assign sum_Acc2_comb[1][9] = $signed(product_Acc1[1][18]) + $signed(product_Acc1[1][19]);
    assign sum_Acc2_comb[1][10] = $signed(product_Acc1[1][20]) + $signed(product_Acc1[1][21]);
    assign sum_Acc2_comb[1][11] = $signed(product_Acc1[1][22]) + $signed(product_Acc1[1][23]);
    assign sum_Acc2_comb[1][12] = $signed(product_Acc1[1][24]) + $signed(product_Acc1[1][25]);
    assign sum_Acc2_comb[1][13] = $signed(product_Acc1[1][26]) + $signed(product_Acc1[1][27]);
    assign sum_Acc2_comb[1][14] = $signed(product_Acc1[1][28]) + $signed(product_Acc1[1][29]);
    assign sum_Acc2_comb[1][15] = $signed(product_Acc1[1][30]) + $signed(product_Acc1[1][31]);
    // Pipeline Register: product_Acc1/Acc2
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Acc2 <= 0;
            reset_acc_Acc2 <= 0;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                bias_Acc2[lane] <= 0;
                addr_Acc2[lane] <= 0;
                for (term = 0; term < 16; term = term + 1)
                    sum_Acc2[lane][term] <= 0;
            end
        end
        else begin
            RES_write_en_Acc2 <= RES_write_en_Acc1;
            reset_acc_Acc2 <= reset_acc_Acc1;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                bias_Acc2[lane] <= bias_Acc1[lane];
                addr_Acc2[lane] <= addr_Acc1[lane];
                for (term = 0; term < 16; term = term + 1)
                    sum_Acc2[lane][term] <= sum_Acc2_comb[lane][term];
            end
        end
    end
    // Stage Acc3: combinational pairwise additions
    assign sum_Acc3_comb[0][0] = $signed(sum_Acc2[0][0]) + $signed(sum_Acc2[0][1]);
    assign sum_Acc3_comb[0][1] = $signed(sum_Acc2[0][2]) + $signed(sum_Acc2[0][3]);
    assign sum_Acc3_comb[0][2] = $signed(sum_Acc2[0][4]) + $signed(sum_Acc2[0][5]);
    assign sum_Acc3_comb[0][3] = $signed(sum_Acc2[0][6]) + $signed(sum_Acc2[0][7]);
    assign sum_Acc3_comb[0][4] = $signed(sum_Acc2[0][8]) + $signed(sum_Acc2[0][9]);
    assign sum_Acc3_comb[0][5] = $signed(sum_Acc2[0][10]) + $signed(sum_Acc2[0][11]);
    assign sum_Acc3_comb[0][6] = $signed(sum_Acc2[0][12]) + $signed(sum_Acc2[0][13]);
    assign sum_Acc3_comb[0][7] = $signed(sum_Acc2[0][14]) + $signed(sum_Acc2[0][15]);
    assign sum_Acc3_comb[1][0] = $signed(sum_Acc2[1][0]) + $signed(sum_Acc2[1][1]);
    assign sum_Acc3_comb[1][1] = $signed(sum_Acc2[1][2]) + $signed(sum_Acc2[1][3]);
    assign sum_Acc3_comb[1][2] = $signed(sum_Acc2[1][4]) + $signed(sum_Acc2[1][5]);
    assign sum_Acc3_comb[1][3] = $signed(sum_Acc2[1][6]) + $signed(sum_Acc2[1][7]);
    assign sum_Acc3_comb[1][4] = $signed(sum_Acc2[1][8]) + $signed(sum_Acc2[1][9]);
    assign sum_Acc3_comb[1][5] = $signed(sum_Acc2[1][10]) + $signed(sum_Acc2[1][11]);
    assign sum_Acc3_comb[1][6] = $signed(sum_Acc2[1][12]) + $signed(sum_Acc2[1][13]);
    assign sum_Acc3_comb[1][7] = $signed(sum_Acc2[1][14]) + $signed(sum_Acc2[1][15]);
    // Pipeline Register: sum_Acc2/Acc3
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Acc3 <= 0;
            reset_acc_Acc3 <= 0;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                bias_Acc3[lane] <= 0;
                addr_Acc3[lane] <= 0;
                for (term = 0; term < 8; term = term + 1)
                    sum_Acc3[lane][term] <= 0;
            end
        end
        else begin
            RES_write_en_Acc3 <= RES_write_en_Acc2;
            reset_acc_Acc3 <= reset_acc_Acc2;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                bias_Acc3[lane] <= bias_Acc2[lane];
                addr_Acc3[lane] <= addr_Acc2[lane];
                for (term = 0; term < 8; term = term + 1)
                    sum_Acc3[lane][term] <= sum_Acc3_comb[lane][term];
            end
        end
    end
    // Stage Acc4: combinational pairwise additions
    assign sum_Acc4_comb[0][0] = $signed(sum_Acc3[0][0]) + $signed(sum_Acc3[0][1]);
    assign sum_Acc4_comb[0][1] = $signed(sum_Acc3[0][2]) + $signed(sum_Acc3[0][3]);
    assign sum_Acc4_comb[0][2] = $signed(sum_Acc3[0][4]) + $signed(sum_Acc3[0][5]);
    assign sum_Acc4_comb[0][3] = $signed(sum_Acc3[0][6]) + $signed(sum_Acc3[0][7]);
    assign sum_Acc4_comb[1][0] = $signed(sum_Acc3[1][0]) + $signed(sum_Acc3[1][1]);
    assign sum_Acc4_comb[1][1] = $signed(sum_Acc3[1][2]) + $signed(sum_Acc3[1][3]);
    assign sum_Acc4_comb[1][2] = $signed(sum_Acc3[1][4]) + $signed(sum_Acc3[1][5]);
    assign sum_Acc4_comb[1][3] = $signed(sum_Acc3[1][6]) + $signed(sum_Acc3[1][7]);
    // Pipeline Register: sum_Acc3/Acc4
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Acc4 <= 0;
            reset_acc_Acc4 <= 0;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                bias_Acc4[lane] <= 0;
                addr_Acc4[lane] <= 0;
                for (term = 0; term < 4; term = term + 1)
                    sum_Acc4[lane][term] <= 0;
            end
        end
        else begin
            RES_write_en_Acc4 <= RES_write_en_Acc3;
            reset_acc_Acc4 <= reset_acc_Acc3;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                bias_Acc4[lane] <= bias_Acc3[lane];
                addr_Acc4[lane] <= addr_Acc3[lane];
                for (term = 0; term < 4; term = term + 1)
                    sum_Acc4[lane][term] <= sum_Acc4_comb[lane][term];
            end
        end
    end
    // Stage Acc5: combinational pairwise additions
    assign sum_Acc5_comb[0][0] = $signed(sum_Acc4[0][0]) + $signed(sum_Acc4[0][1]);
    assign sum_Acc5_comb[0][1] = $signed(sum_Acc4[0][2]) + $signed(sum_Acc4[0][3]);
    assign sum_Acc5_comb[1][0] = $signed(sum_Acc4[1][0]) + $signed(sum_Acc4[1][1]);
    assign sum_Acc5_comb[1][1] = $signed(sum_Acc4[1][2]) + $signed(sum_Acc4[1][3]);
    // Pipeline Register: sum_Acc4/Acc5
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Acc5 <= 0;
            reset_acc_Acc5 <= 0;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                bias_Acc5[lane] <= 0;
                addr_Acc5[lane] <= 0;
                for (term = 0; term < 2; term = term + 1)
                    sum_Acc5[lane][term] <= 0;
            end
        end
        else begin
            RES_write_en_Acc5 <= RES_write_en_Acc4;
            reset_acc_Acc5 <= reset_acc_Acc4;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                bias_Acc5[lane] <= bias_Acc4[lane];
                addr_Acc5[lane] <= addr_Acc4[lane];
                for (term = 0; term < 2; term = term + 1)
                    sum_Acc5[lane][term] <= sum_Acc5_comb[lane][term];
            end
        end
    end
    // Stage Acc6: combinational pairwise additions
    assign sum_Acc6_comb[0][0] = $signed(sum_Acc5[0][0]) + $signed(sum_Acc5[0][1]);
    assign sum_Acc6_comb[1][0] = $signed(sum_Acc5[1][0]) + $signed(sum_Acc5[1][1]);
    // Pipeline Register: sum_Acc5/Acc6
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Acc6 <= 0;
            reset_acc_Acc6 <= 0;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                bias_Acc6[lane] <= 0;
                addr_Acc6[lane] <= 0;
                for (term = 0; term < 1; term = term + 1)
                    sum_Acc6[lane][term] <= 0;
            end
        end
        else begin
            RES_write_en_Acc6 <= RES_write_en_Acc5;
            reset_acc_Acc6 <= reset_acc_Acc5;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                bias_Acc6[lane] <= bias_Acc5[lane];
                addr_Acc6[lane] <= addr_Acc5[lane];
                for (term = 0; term < 1; term = term + 1)
                    sum_Acc6[lane][term] <= sum_Acc6_comb[lane][term];
            end
        end
    end
    // Temporal accumulation combinational logic
    assign accumulator_next[0] = reset_acc_Acc6 ? sum_Acc6[0][0]:$signed(accumulator_KAcc[0]) + $signed(sum_Acc6[0][0]);
    assign accumulator_next[1] = reset_acc_Acc6 ? sum_Acc6[1][0]:$signed(accumulator_KAcc[1]) + $signed(sum_Acc6[1][0]);
    // Pipeline Register: Acc6/KAcc
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_KAcc <= 0;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                accumulator_KAcc[lane] <= 0;
                bias_KAcc[lane] <= 0;
                addr_KAcc[lane] <= 0;
            end
        end
        else begin
            RES_write_en_KAcc <= RES_write_en_Acc6;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                accumulator_KAcc[lane] <= accumulator_next[lane];
                bias_KAcc[lane] <= bias_Acc6[lane];
                addr_KAcc[lane] <= addr_Acc6[lane];
            end
        end
    end
    // Bias/truncate combinational logic
    assign aligned_bias[0] = $signed(bias_KAcc[0]) <<< FRAC_BITS;
    assign biased_value[0] = $signed(accumulator_KAcc[0]) + aligned_bias[0];
    assign truncated_value[0] = biased_value[0][width+FRAC_BITS-1:FRAC_BITS];
    assign aligned_bias[1] = $signed(bias_KAcc[1]) <<< FRAC_BITS;
    assign biased_value[1] = $signed(accumulator_KAcc[1]) + aligned_bias[1];
    assign truncated_value[1] = biased_value[1][width+FRAC_BITS-1:FRAC_BITS];
    // Pipeline Register: Bias/ReLU input
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Relu <= 0;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                data_Relu[lane] <= 0;
                addr_Relu[lane] <= 0;
            end
        end
        else begin
            RES_write_en_Relu <= RES_write_en_KAcc;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                data_Relu[lane] <= truncated_value[lane];
                addr_Relu[lane] <= addr_KAcc[lane];
            end
        end
    end
    // ReLU combinational logic
    assign relu_value[0] = data_Relu[0][width-1] ? 0:data_Relu[0];
    assign relu_value[1] = data_Relu[1][width-1] ? 0:data_Relu[1];
    // Pipeline Register: ReLU/Output1
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Out1 <= 0;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                data_Out1[lane] <= 0;
                addr_Out1[lane] <= 0;
            end
        end
        else begin
            RES_write_en_Out1 <= RES_write_en_Relu;
            for (lane = 0; lane < OUT_UNROLL; lane = lane + 1) begin
                data_Out1[lane] <= relu_value[lane];
                addr_Out1[lane] <= addr_Relu[lane];
            end
        end
    end
    // Both OU2 results are written in the same cycle to distinct channel RAMs.
    assign RES_write_en_even = RES_write_en_Out1;
    assign RES_write_en_odd = RES_write_en_Out1;
    assign RES_write_address_even = addr_Out1[0];
    assign RES_write_address_odd = addr_Out1[1];
    assign RES_write_data_in_even = data_Out1[0];
    assign RES_write_data_in_odd = data_Out1[1];

    // -------------------------------------------------------------------------
    // Counter Initialization
    // -------------------------------------------------------------------------

    // Kernel Column Counter (Innermost)
    Counter #(
                .ADDR_WIDTH ($clog2(K)),
                .CNT_INIT (0),
                .MAX_VAL (K - 1)
            ) k_col_counter (
                .CLK (clk),
                .RESET (!rst || counter_clear_Read1),
                .ENABLE (en_k_col),
                .CNT (k_col)
            );

    // Kernel Row Counter
    Counter #(
                .ADDR_WIDTH ($clog2(K)),
                .CNT_INIT (0),
                .MAX_VAL (K - 1)
            ) k_row_counter (
                .CLK (clk),
                .RESET (!rst || counter_clear_Read1),
                .ENABLE (en_k_row),
                .CNT (k_row)
            );

    // Image Column Counter
    Counter #(
                .ADDR_WIDTH ($clog2(IN_W)),
                .CNT_INIT (0),
                .MAX_VAL (IN_W - 1)
            ) col_counter (
                .CLK (clk),
                .RESET (!rst || counter_clear_Read1),
                .ENABLE (en_col),
                .CNT (col)
            );

    // Image Row Counter
    Counter #(
                .ADDR_WIDTH ($clog2(IN_H)),
                .CNT_INIT (0),
                .MAX_VAL (IN_H - 1)
            ) row_counter (
                .CLK (clk),
                .RESET (!rst || counter_clear_Read1),
                .ENABLE (en_row),
                .CNT (row)
            );

    // Output Channel Counter (Outermost)
    Counter #(
                .ADDR_WIDTH ($clog2(OC_DIM / OUT_UNROLL)),
                .CNT_INIT (0),
                .MAX_VAL ((OC_DIM / OUT_UNROLL) - 1)
            ) oc_counter (
                .CLK (clk),
                .RESET (!rst || counter_clear_Read1),
                .ENABLE (en_oc),
                .CNT (oc)
            );

    // -------------------------------------------------------------------------
    // Address Generation Units
    // -------------------------------------------------------------------------

    // Input memory address generator
    conv3_input_agu #(
                        .ADDR_WIDTH (A_depth_bits),
                        .IN_H (IN_H),
                        .IN_W (IN_W),
                        .K (K),
                        .PAD (1)
                    ) image_address_gen (
                        .row (row),
                        .col (col),
                        .k_row (k_row),
                        .k_col (k_col),
                        .A_read_address (A_read_address_Read1),
                        .out_of_bounds (out_of_bounds_Read1)
                    );

    // Weights memory address generator
    conv3_weight_agu #(
                         .ADDR_WIDTH (B_depth_bits),
                         .OC_DIM (OC_DIM),
                         .K (K),
                         .OUTPUT_UNROLL (OUT_UNROLL)
                     ) weight_address_gen (
                         .oc (oc),
                         .k_row (k_row),
                         .k_col (k_col),
                         .B_read_address(B_read_address_Read1)
                     );

    // Output RAM address generator
    conv3_res_agu #(
                      .ADDR_WIDTH (RES_depth_bits),
                      .OC_DIM (OC_DIM),
                      .IN_H (IN_H),
                      .IN_W (IN_W),
                      .OUTPUT_UNROLL (OUT_UNROLL)
                  ) res_address_gen (
                      .oc (oc),
                      .row (row),
                      .col (col),
                      .RES_write_address_0 (calc_addr_Read1[0]),
                      .RES_write_address_1 (calc_addr_Read1[1])
                  );

    assign C_read_address_Read1 = oc;
endmodule

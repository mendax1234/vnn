`timescale 1ns / 1ps

/* ----------------------------------------------------------------------------------
--  RTL Implementation of Fused Convolution Layer 6 + ReLU
--  Input-channel unroll factor: 64 (two groups cover all 128 input channels)
--  Output-channel unroll factor: 2 (two output channels processed in parallel)
--  The 3x3 kernel positions and input-channel groups are processed temporally.
--  Input-channel products are reduced through a pipelined adder tree.
--  ReLU is applied after bias addition and truncation, before output storage.
--  Two results are written together through the two TDP output ports.
----------------------------------------------------------------------------------
*/

module conv6_relu_unrolled_ping_pong #(
        parameter width = 16,
        parameter FRAC_BITS = 10,
        parameter IC_DIM = 128,
        parameter IC_UNROLL = 64,
        parameter OC_DIM = 128,
        parameter OUT_UNROLL = 2,
        parameter IN_H = 8,
        parameter IN_W = 8,
        parameter K = 3,
        parameter A_depth_bits = 7,
        parameter B_depth_bits = 11,
        parameter C_depth_bits = 6,
        parameter RES_depth_bits = 13
    ) (
        // Clock, reset and handshake signals
        input clk,
        input rst,
        input consumer_ready,
        input producer_ready,
        output reg consumer_frame_done,
        output reg producer_frame_done,
        output reg A_read_en,
        output reg [A_depth_bits-1:0] A_read_address,
        output reg B_read_en,
        output reg [B_depth_bits-1:0] B_read_address,
        output reg C_read_en,
        output reg [C_depth_bits-1:0] C_read_address,

        // Input feature map (64 channels)
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
        input [width-1:0] A_read_data_out_32,
        input [width-1:0] A_read_data_out_33,
        input [width-1:0] A_read_data_out_34,
        input [width-1:0] A_read_data_out_35,
        input [width-1:0] A_read_data_out_36,
        input [width-1:0] A_read_data_out_37,
        input [width-1:0] A_read_data_out_38,
        input [width-1:0] A_read_data_out_39,
        input [width-1:0] A_read_data_out_40,
        input [width-1:0] A_read_data_out_41,
        input [width-1:0] A_read_data_out_42,
        input [width-1:0] A_read_data_out_43,
        input [width-1:0] A_read_data_out_44,
        input [width-1:0] A_read_data_out_45,
        input [width-1:0] A_read_data_out_46,
        input [width-1:0] A_read_data_out_47,
        input [width-1:0] A_read_data_out_48,
        input [width-1:0] A_read_data_out_49,
        input [width-1:0] A_read_data_out_50,
        input [width-1:0] A_read_data_out_51,
        input [width-1:0] A_read_data_out_52,
        input [width-1:0] A_read_data_out_53,
        input [width-1:0] A_read_data_out_54,
        input [width-1:0] A_read_data_out_55,
        input [width-1:0] A_read_data_out_56,
        input [width-1:0] A_read_data_out_57,
        input [width-1:0] A_read_data_out_58,
        input [width-1:0] A_read_data_out_59,
        input [width-1:0] A_read_data_out_60,
        input [width-1:0] A_read_data_out_61,
        input [width-1:0] A_read_data_out_62,
        input [width-1:0] A_read_data_out_63,

        // Weight values for output channel 0 (64 input channels)
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
        input [width-1:0] B_read_data_out_32_oc0,
        input [width-1:0] B_read_data_out_33_oc0,
        input [width-1:0] B_read_data_out_34_oc0,
        input [width-1:0] B_read_data_out_35_oc0,
        input [width-1:0] B_read_data_out_36_oc0,
        input [width-1:0] B_read_data_out_37_oc0,
        input [width-1:0] B_read_data_out_38_oc0,
        input [width-1:0] B_read_data_out_39_oc0,
        input [width-1:0] B_read_data_out_40_oc0,
        input [width-1:0] B_read_data_out_41_oc0,
        input [width-1:0] B_read_data_out_42_oc0,
        input [width-1:0] B_read_data_out_43_oc0,
        input [width-1:0] B_read_data_out_44_oc0,
        input [width-1:0] B_read_data_out_45_oc0,
        input [width-1:0] B_read_data_out_46_oc0,
        input [width-1:0] B_read_data_out_47_oc0,
        input [width-1:0] B_read_data_out_48_oc0,
        input [width-1:0] B_read_data_out_49_oc0,
        input [width-1:0] B_read_data_out_50_oc0,
        input [width-1:0] B_read_data_out_51_oc0,
        input [width-1:0] B_read_data_out_52_oc0,
        input [width-1:0] B_read_data_out_53_oc0,
        input [width-1:0] B_read_data_out_54_oc0,
        input [width-1:0] B_read_data_out_55_oc0,
        input [width-1:0] B_read_data_out_56_oc0,
        input [width-1:0] B_read_data_out_57_oc0,
        input [width-1:0] B_read_data_out_58_oc0,
        input [width-1:0] B_read_data_out_59_oc0,
        input [width-1:0] B_read_data_out_60_oc0,
        input [width-1:0] B_read_data_out_61_oc0,
        input [width-1:0] B_read_data_out_62_oc0,
        input [width-1:0] B_read_data_out_63_oc0,

        // Weight values for output channel 1 (64 input channels)
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
        input [width-1:0] B_read_data_out_32_oc1,
        input [width-1:0] B_read_data_out_33_oc1,
        input [width-1:0] B_read_data_out_34_oc1,
        input [width-1:0] B_read_data_out_35_oc1,
        input [width-1:0] B_read_data_out_36_oc1,
        input [width-1:0] B_read_data_out_37_oc1,
        input [width-1:0] B_read_data_out_38_oc1,
        input [width-1:0] B_read_data_out_39_oc1,
        input [width-1:0] B_read_data_out_40_oc1,
        input [width-1:0] B_read_data_out_41_oc1,
        input [width-1:0] B_read_data_out_42_oc1,
        input [width-1:0] B_read_data_out_43_oc1,
        input [width-1:0] B_read_data_out_44_oc1,
        input [width-1:0] B_read_data_out_45_oc1,
        input [width-1:0] B_read_data_out_46_oc1,
        input [width-1:0] B_read_data_out_47_oc1,
        input [width-1:0] B_read_data_out_48_oc1,
        input [width-1:0] B_read_data_out_49_oc1,
        input [width-1:0] B_read_data_out_50_oc1,
        input [width-1:0] B_read_data_out_51_oc1,
        input [width-1:0] B_read_data_out_52_oc1,
        input [width-1:0] B_read_data_out_53_oc1,
        input [width-1:0] B_read_data_out_54_oc1,
        input [width-1:0] B_read_data_out_55_oc1,
        input [width-1:0] B_read_data_out_56_oc1,
        input [width-1:0] B_read_data_out_57_oc1,
        input [width-1:0] B_read_data_out_58_oc1,
        input [width-1:0] B_read_data_out_59_oc1,
        input [width-1:0] B_read_data_out_60_oc1,
        input [width-1:0] B_read_data_out_61_oc1,
        input [width-1:0] B_read_data_out_62_oc1,
        input [width-1:0] B_read_data_out_63_oc1,



        // Bias data (two output channels)
        input [width-1:0] C_read_data_out_oc0,
        input [width-1:0] C_read_data_out_oc1,

        // Output RAM (EVEN and ODD via TDP RAM)
        output RES_write_en_even,
        output [RES_depth_bits-1:0] RES_write_address_even,
        output [width-1:0] RES_write_data_even,
        output RES_write_en_odd,
        output [RES_depth_bits-1:0] RES_write_address_odd,
        output [width-1:0] RES_write_data_odd
    );


    // -------------------------------------------------------------------------
    // System Parameters and FSM States
    // -------------------------------------------------------------------------
    localparam IC_GROUPS = IC_DIM / IC_UNROLL;
    localparam ACC_WIDTH = 2 * width + 8;
    localparam NUMBER_OF_OUTPUT_WORDS = OC_DIM * IN_H * IN_W;
    localparam IDLE = 4'b1000, COMPUTE = 4'b0100, FLUSH = 4'b0010, FINISH = 4'b0001;
    reg[3:0] state = IDLE, n_state = IDLE;
    wire[$clog2(OC_DIM/OUT_UNROLL)-1:0] oc;
    wire[$clog2(IC_GROUPS)-1:0] ic_group;
    reg en_ic_group;
    wire[$clog2(IN_H)-1:0] row;
    wire[$clog2(IN_W)-1:0] col;
    wire[$clog2(K)-1:0] k_row, k_col;
    reg en_k_col, en_k_row, en_col, en_row, en_oc;
    wire signed[width-1:0] A_bus[0:IC_UNROLL-1];
    wire signed[width-1:0] B_bus[0:OUT_UNROLL-1][0:IC_UNROLL-1];
    wire signed[width-1:0] C_bus[0:OUT_UNROLL-1];
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
    assign A_bus[32] = A_read_data_out_32;
    assign A_bus[33] = A_read_data_out_33;
    assign A_bus[34] = A_read_data_out_34;
    assign A_bus[35] = A_read_data_out_35;
    assign A_bus[36] = A_read_data_out_36;
    assign A_bus[37] = A_read_data_out_37;
    assign A_bus[38] = A_read_data_out_38;
    assign A_bus[39] = A_read_data_out_39;
    assign A_bus[40] = A_read_data_out_40;
    assign A_bus[41] = A_read_data_out_41;
    assign A_bus[42] = A_read_data_out_42;
    assign A_bus[43] = A_read_data_out_43;
    assign A_bus[44] = A_read_data_out_44;
    assign A_bus[45] = A_read_data_out_45;
    assign A_bus[46] = A_read_data_out_46;
    assign A_bus[47] = A_read_data_out_47;
    assign A_bus[48] = A_read_data_out_48;
    assign A_bus[49] = A_read_data_out_49;
    assign A_bus[50] = A_read_data_out_50;
    assign A_bus[51] = A_read_data_out_51;
    assign A_bus[52] = A_read_data_out_52;
    assign A_bus[53] = A_read_data_out_53;
    assign A_bus[54] = A_read_data_out_54;
    assign A_bus[55] = A_read_data_out_55;
    assign A_bus[56] = A_read_data_out_56;
    assign A_bus[57] = A_read_data_out_57;
    assign A_bus[58] = A_read_data_out_58;
    assign A_bus[59] = A_read_data_out_59;
    assign A_bus[60] = A_read_data_out_60;
    assign A_bus[61] = A_read_data_out_61;
    assign A_bus[62] = A_read_data_out_62;
    assign A_bus[63] = A_read_data_out_63;
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
    assign B_bus[0][32] = B_read_data_out_32_oc0;
    assign B_bus[0][33] = B_read_data_out_33_oc0;
    assign B_bus[0][34] = B_read_data_out_34_oc0;
    assign B_bus[0][35] = B_read_data_out_35_oc0;
    assign B_bus[0][36] = B_read_data_out_36_oc0;
    assign B_bus[0][37] = B_read_data_out_37_oc0;
    assign B_bus[0][38] = B_read_data_out_38_oc0;
    assign B_bus[0][39] = B_read_data_out_39_oc0;
    assign B_bus[0][40] = B_read_data_out_40_oc0;
    assign B_bus[0][41] = B_read_data_out_41_oc0;
    assign B_bus[0][42] = B_read_data_out_42_oc0;
    assign B_bus[0][43] = B_read_data_out_43_oc0;
    assign B_bus[0][44] = B_read_data_out_44_oc0;
    assign B_bus[0][45] = B_read_data_out_45_oc0;
    assign B_bus[0][46] = B_read_data_out_46_oc0;
    assign B_bus[0][47] = B_read_data_out_47_oc0;
    assign B_bus[0][48] = B_read_data_out_48_oc0;
    assign B_bus[0][49] = B_read_data_out_49_oc0;
    assign B_bus[0][50] = B_read_data_out_50_oc0;
    assign B_bus[0][51] = B_read_data_out_51_oc0;
    assign B_bus[0][52] = B_read_data_out_52_oc0;
    assign B_bus[0][53] = B_read_data_out_53_oc0;
    assign B_bus[0][54] = B_read_data_out_54_oc0;
    assign B_bus[0][55] = B_read_data_out_55_oc0;
    assign B_bus[0][56] = B_read_data_out_56_oc0;
    assign B_bus[0][57] = B_read_data_out_57_oc0;
    assign B_bus[0][58] = B_read_data_out_58_oc0;
    assign B_bus[0][59] = B_read_data_out_59_oc0;
    assign B_bus[0][60] = B_read_data_out_60_oc0;
    assign B_bus[0][61] = B_read_data_out_61_oc0;
    assign B_bus[0][62] = B_read_data_out_62_oc0;
    assign B_bus[0][63] = B_read_data_out_63_oc0;
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
    assign B_bus[1][32] = B_read_data_out_32_oc1;
    assign B_bus[1][33] = B_read_data_out_33_oc1;
    assign B_bus[1][34] = B_read_data_out_34_oc1;
    assign B_bus[1][35] = B_read_data_out_35_oc1;
    assign B_bus[1][36] = B_read_data_out_36_oc1;
    assign B_bus[1][37] = B_read_data_out_37_oc1;
    assign B_bus[1][38] = B_read_data_out_38_oc1;
    assign B_bus[1][39] = B_read_data_out_39_oc1;
    assign B_bus[1][40] = B_read_data_out_40_oc1;
    assign B_bus[1][41] = B_read_data_out_41_oc1;
    assign B_bus[1][42] = B_read_data_out_42_oc1;
    assign B_bus[1][43] = B_read_data_out_43_oc1;
    assign B_bus[1][44] = B_read_data_out_44_oc1;
    assign B_bus[1][45] = B_read_data_out_45_oc1;
    assign B_bus[1][46] = B_read_data_out_46_oc1;
    assign B_bus[1][47] = B_read_data_out_47_oc1;
    assign B_bus[1][48] = B_read_data_out_48_oc1;
    assign B_bus[1][49] = B_read_data_out_49_oc1;
    assign B_bus[1][50] = B_read_data_out_50_oc1;
    assign B_bus[1][51] = B_read_data_out_51_oc1;
    assign B_bus[1][52] = B_read_data_out_52_oc1;
    assign B_bus[1][53] = B_read_data_out_53_oc1;
    assign B_bus[1][54] = B_read_data_out_54_oc1;
    assign B_bus[1][55] = B_read_data_out_55_oc1;
    assign B_bus[1][56] = B_read_data_out_56_oc1;
    assign B_bus[1][57] = B_read_data_out_57_oc1;
    assign B_bus[1][58] = B_read_data_out_58_oc1;
    assign B_bus[1][59] = B_read_data_out_59_oc1;
    assign B_bus[1][60] = B_read_data_out_60_oc1;
    assign B_bus[1][61] = B_read_data_out_61_oc1;
    assign B_bus[1][62] = B_read_data_out_62_oc1;
    assign B_bus[1][63] = B_read_data_out_63_oc1;
    assign C_bus[0] = C_read_data_out_oc0;
    assign C_bus[1] = C_read_data_out_oc1;
    wire[A_depth_bits-1:0] A_read_address_Read1;
    wire[B_depth_bits-1:0] B_read_address_Read1;
    wire[C_depth_bits-1:0] C_read_address_Read1;
    wire[RES_depth_bits-1:0] calc_addr_Read1[0:OUT_UNROLL-1];
    wire out_of_bounds_Read1;
    reg A_read_en_Read1, B_read_en_Read1, C_read_en_Read1;
    reg RES_write_en_Read1, counter_clear_Read1, reset_acc_Read1;
    reg RES_write_en_Read2, reset_acc_Read2, out_of_bounds_Read2;
    reg[RES_depth_bits-1:0] calc_addr_Read2[0:OUT_UNROLL-1];
    wire signed[width-1:0] padded_A_Read2[0:IC_UNROLL-1];
    reg RES_write_en_Read3, reset_acc_Read3;
    reg signed[width-1:0] A_Read3[0:IC_UNROLL-1];
    reg signed[width-1:0] C_Read3[0:OUT_UNROLL-1];
    reg[RES_depth_bits-1:0] addr_Read3[0:OUT_UNROLL-1];
    // Delay asynchronous LUTRAM input/bias data and control to align with the
    // three-cycle packed UltraRAM weight output.
    reg RES_write_en_Read4, reset_acc_Read4;
    reg signed[width-1:0] A_Read4[0:IC_UNROLL-1];
    reg signed[width-1:0] C_Read4[0:OUT_UNROLL-1];
    reg[RES_depth_bits-1:0] addr_Read4[0:OUT_UNROLL-1];
    reg RES_write_en_Read5, reset_acc_Read5;
    reg signed[width-1:0] A_Read5[0:IC_UNROLL-1];
    reg signed[width-1:0] C_Read5[0:OUT_UNROLL-1];
    reg[RES_depth_bits-1:0] addr_Read5[0:OUT_UNROLL-1];
    reg RES_write_en_Mul, reset_acc_Mul;
    reg signed[width-1:0] A_Mul[0:IC_UNROLL-1];
    reg signed[width-1:0] B_Mul[0:OUT_UNROLL-1][0:IC_UNROLL-1];
    reg signed[width-1:0] C_Mul[0:OUT_UNROLL-1];
    reg[RES_depth_bits-1:0] addr_Mul[0:OUT_UNROLL-1];
    wire signed[2*width-1:0] product_Mul[0:OUT_UNROLL-1][0:IC_UNROLL-1];
    reg signed[2*width-1:0] product_Acc1[0:OUT_UNROLL-1][0:IC_UNROLL-1];
    reg[width-1:0] bias_Acc1[0:OUT_UNROLL-1];
    reg[RES_depth_bits-1:0] addr_Acc1[0:OUT_UNROLL-1];
    reg RES_write_en_Acc1, reset_acc_Acc1;
    wire signed[ACC_WIDTH-1:0] sum_Acc2_comb[0:OUT_UNROLL-1][0:31];
    reg signed[ACC_WIDTH-1:0] sum_Acc2[0:OUT_UNROLL-1][0:31];
    reg[width-1:0] bias_Acc2[0:OUT_UNROLL-1];
    reg[RES_depth_bits-1:0] addr_Acc2[0:OUT_UNROLL-1];
    reg RES_write_en_Acc2, reset_acc_Acc2;
    wire signed[ACC_WIDTH-1:0] sum_Acc3_comb[0:OUT_UNROLL-1][0:15];
    reg signed[ACC_WIDTH-1:0] sum_Acc3[0:OUT_UNROLL-1][0:15];
    reg[width-1:0] bias_Acc3[0:OUT_UNROLL-1];
    reg[RES_depth_bits-1:0] addr_Acc3[0:OUT_UNROLL-1];
    reg RES_write_en_Acc3, reset_acc_Acc3;
    wire signed[ACC_WIDTH-1:0] sum_Acc4_comb[0:OUT_UNROLL-1][0:7];
    reg signed[ACC_WIDTH-1:0] sum_Acc4[0:OUT_UNROLL-1][0:7];
    reg[width-1:0] bias_Acc4[0:OUT_UNROLL-1];
    reg[RES_depth_bits-1:0] addr_Acc4[0:OUT_UNROLL-1];
    reg RES_write_en_Acc4, reset_acc_Acc4;
    wire signed[ACC_WIDTH-1:0] sum_Acc5_comb[0:OUT_UNROLL-1][0:3];
    reg signed[ACC_WIDTH-1:0] sum_Acc5[0:OUT_UNROLL-1][0:3];
    reg[width-1:0] bias_Acc5[0:OUT_UNROLL-1];
    reg[RES_depth_bits-1:0] addr_Acc5[0:OUT_UNROLL-1];
    reg RES_write_en_Acc5, reset_acc_Acc5;
    wire signed[ACC_WIDTH-1:0] sum_Acc6_comb[0:OUT_UNROLL-1][0:1];
    reg signed[ACC_WIDTH-1:0] sum_Acc6[0:OUT_UNROLL-1][0:1];
    reg[width-1:0] bias_Acc6[0:OUT_UNROLL-1];
    reg[RES_depth_bits-1:0] addr_Acc6[0:OUT_UNROLL-1];
    reg RES_write_en_Acc6, reset_acc_Acc6;
    // -------------------------------------------------------------------------
    // Stage 4: Kernel Accumulation
    // -------------------------------------------------------------------------
    wire signed[ACC_WIDTH-1:0] sum_Acc7_comb[0:OUT_UNROLL-1][0:0];
    reg signed[ACC_WIDTH-1:0] sum_Acc7[0:OUT_UNROLL-1][0:0];
    reg[width-1:0] bias_Acc7[0:OUT_UNROLL-1];
    reg[RES_depth_bits-1:0] addr_Acc7[0:OUT_UNROLL-1];
    reg RES_write_en_Acc7, reset_acc_Acc7;
    reg signed[ACC_WIDTH-1:0] accumulator_KAcc[0:OUT_UNROLL-1];
    wire signed[ACC_WIDTH-1:0] accumulator_next[0:OUT_UNROLL-1];
    reg[width-1:0] bias_KAcc[0:OUT_UNROLL-1];
    reg[RES_depth_bits-1:0] addr_KAcc[0:OUT_UNROLL-1];
    reg RES_write_en_KAcc;
    wire signed[ACC_WIDTH-1:0] aligned_bias[0:OUT_UNROLL-1], biased_value[0:OUT_UNROLL-1];
    wire[width-1:0] truncated_value[0:OUT_UNROLL-1], relu_value[0:OUT_UNROLL-1];
    reg[width-1:0] data_Relu[0:OUT_UNROLL-1];
    reg[RES_depth_bits-1:0] addr_Relu[0:OUT_UNROLL-1];
    reg RES_write_en_Relu;
    reg RES_write_en_Out1;
    reg[width-1:0] data_Out1[0:OUT_UNROLL-1];
    reg[RES_depth_bits-1:0] addr_Out1[0:OUT_UNROLL-1];
    // -------------------------------------------------------------------------
    // Control Logic
    // -------------------------------------------------------------------------
    integer lane, channel, term;
    always @(posedge clk) state <= n_state;
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
        en_ic_group = 0;
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
                reset_acc_Read1 = (ic_group == 0) && (k_row == 0) && (k_col == 0);
                RES_write_en_Read1 = (ic_group == IC_GROUPS - 1) && (k_row == K - 1) && (k_col == K - 1);
                en_ic_group = 1;
                en_k_col = en_ic_group && (ic_group == IC_GROUPS - 1);
                en_k_row = en_k_col && (k_col == K - 1);
                en_col = en_k_row && (k_row == K - 1);
                en_row = en_col && (col == IN_W - 1);
                en_oc = en_row && (row == IN_H - 1);
                if (en_oc && (oc == (OC_DIM / OUT_UNROLL) - 1))
                    n_state = FLUSH;
            end
            FLUSH:
                if (RES_write_en_Out1 && (addr_Out1[1] == NUMBER_OF_OUTPUT_WORDS - 1))
                    n_state = FINISH;
            FINISH: begin
                consumer_frame_done = 1;
                producer_frame_done = 1;
                n_state = IDLE;
            end
        endcase
    end
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
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1)
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
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1)
                calc_addr_Read2[lane] <= calc_addr_Read1[lane];
        end
    end
    // -------------------------------------------------------------------------
    // Datapath
    // Stage 1: Read / Padding
    // -------------------------------------------------------------------------
    genvar ga;
    generate
        for (ga = 0; ga < IC_UNROLL; ga = ga + 1) begin : PAD_INPUT
            assign padded_A_Read2[ga] = out_of_bounds_Read2 ? 0 : A_bus[ga];
        end
    endgenerate
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Read3 <= 0;
            reset_acc_Read3 <= 0;
            for (channel = 0;channel < IC_UNROLL;channel = channel + 1)
                A_Read3[channel] <= 0;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                C_Read3[lane] <= 0;
                addr_Read3[lane] <= 0;
            end
        end
        else begin
            RES_write_en_Read3 <= RES_write_en_Read2;
            reset_acc_Read3 <= reset_acc_Read2;
            for (channel = 0;channel < IC_UNROLL;channel = channel + 1)
                A_Read3[channel] <= padded_A_Read2[channel];
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                C_Read3[lane] <= C_bus[lane];
                addr_Read3[lane] <= calc_addr_Read2[lane];
            end
        end
    end

    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Read4 <= 0;
            reset_acc_Read4 <= 0;
            for (channel = 0;channel < IC_UNROLL;channel = channel + 1)
                A_Read4[channel] <= 0;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                C_Read4[lane] <= 0;
                addr_Read4[lane] <= 0;
            end
        end
        else begin
            RES_write_en_Read4 <= RES_write_en_Read3;
            reset_acc_Read4 <= reset_acc_Read3;
            for (channel = 0;channel < IC_UNROLL;channel = channel + 1)
                A_Read4[channel] <= A_Read3[channel];
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                C_Read4[lane] <= C_Read3[lane];
                addr_Read4[lane] <= addr_Read3[lane];
            end
        end
    end

    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Read5 <= 0;
            reset_acc_Read5 <= 0;
            for (channel = 0;channel < IC_UNROLL;channel = channel + 1)
                A_Read5[channel] <= 0;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                C_Read5[lane] <= 0;
                addr_Read5[lane] <= 0;
            end
        end
        else begin
            RES_write_en_Read5 <= RES_write_en_Read4;
            reset_acc_Read5 <= reset_acc_Read4;
            for (channel = 0;channel < IC_UNROLL;channel = channel + 1)
                A_Read5[channel] <= A_Read4[channel];
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                C_Read5[lane] <= C_Read4[lane];
                addr_Read5[lane] <= addr_Read4[lane];
            end
        end
    end

    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Mul <= 0;
            reset_acc_Mul <= 0;
            for (channel = 0;channel < IC_UNROLL;channel = channel + 1)
                A_Mul[channel] <= 0;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                C_Mul[lane] <= 0;
                addr_Mul[lane] <= 0;
                for (channel = 0;channel < IC_UNROLL;channel = channel + 1)
                    B_Mul[lane][channel] <= 0;
            end
        end
        else begin
            RES_write_en_Mul <= RES_write_en_Read5;
            reset_acc_Mul <= reset_acc_Read5;
            for (channel = 0;channel < IC_UNROLL;channel = channel + 1)
                A_Mul[channel] <= A_Read5[channel];
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                C_Mul[lane] <= C_Read5[lane];
                addr_Mul[lane] <= addr_Read5[lane];
                for (channel = 0;channel < IC_UNROLL;channel = channel + 1)
                    B_Mul[lane][channel] <= B_bus[lane][channel];
            end
        end
    end
    genvar gm, gn;
    generate
        for (gm = 0; gm < OUT_UNROLL; gm = gm + 1) begin : MUL_LANES
            for (gn = 0; gn < IC_UNROLL; gn = gn + 1) begin : MUL_INPUTS
                assign product_Mul[gm][gn] = $signed(A_Mul[gn]) * $signed(B_Mul[gm][gn]);
            end
        end
    endgenerate
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Acc1 <= 0;
            reset_acc_Acc1 <= 0;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                bias_Acc1[lane] <= 0;
                addr_Acc1[lane] <= 0;
                for (channel = 0;channel < IC_UNROLL;channel = channel + 1)
                    product_Acc1[lane][channel] <= 0;
            end
        end
        else begin
            RES_write_en_Acc1 <= RES_write_en_Mul;
            reset_acc_Acc1 <= reset_acc_Mul;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                bias_Acc1[lane] <= C_Mul[lane];
                addr_Acc1[lane] <= addr_Mul[lane];
                for (channel = 0;channel < IC_UNROLL;channel = channel + 1)
                    product_Acc1[lane][channel] <= product_Mul[lane][channel];
            end
        end
    end
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
    assign sum_Acc2_comb[0][16] = $signed(product_Acc1[0][32]) + $signed(product_Acc1[0][33]);
    assign sum_Acc2_comb[0][17] = $signed(product_Acc1[0][34]) + $signed(product_Acc1[0][35]);
    assign sum_Acc2_comb[0][18] = $signed(product_Acc1[0][36]) + $signed(product_Acc1[0][37]);
    assign sum_Acc2_comb[0][19] = $signed(product_Acc1[0][38]) + $signed(product_Acc1[0][39]);
    assign sum_Acc2_comb[0][20] = $signed(product_Acc1[0][40]) + $signed(product_Acc1[0][41]);
    assign sum_Acc2_comb[0][21] = $signed(product_Acc1[0][42]) + $signed(product_Acc1[0][43]);
    assign sum_Acc2_comb[0][22] = $signed(product_Acc1[0][44]) + $signed(product_Acc1[0][45]);
    assign sum_Acc2_comb[0][23] = $signed(product_Acc1[0][46]) + $signed(product_Acc1[0][47]);
    assign sum_Acc2_comb[0][24] = $signed(product_Acc1[0][48]) + $signed(product_Acc1[0][49]);
    assign sum_Acc2_comb[0][25] = $signed(product_Acc1[0][50]) + $signed(product_Acc1[0][51]);
    assign sum_Acc2_comb[0][26] = $signed(product_Acc1[0][52]) + $signed(product_Acc1[0][53]);
    assign sum_Acc2_comb[0][27] = $signed(product_Acc1[0][54]) + $signed(product_Acc1[0][55]);
    assign sum_Acc2_comb[0][28] = $signed(product_Acc1[0][56]) + $signed(product_Acc1[0][57]);
    assign sum_Acc2_comb[0][29] = $signed(product_Acc1[0][58]) + $signed(product_Acc1[0][59]);
    assign sum_Acc2_comb[0][30] = $signed(product_Acc1[0][60]) + $signed(product_Acc1[0][61]);
    assign sum_Acc2_comb[0][31] = $signed(product_Acc1[0][62]) + $signed(product_Acc1[0][63]);
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
    assign sum_Acc2_comb[1][16] = $signed(product_Acc1[1][32]) + $signed(product_Acc1[1][33]);
    assign sum_Acc2_comb[1][17] = $signed(product_Acc1[1][34]) + $signed(product_Acc1[1][35]);
    assign sum_Acc2_comb[1][18] = $signed(product_Acc1[1][36]) + $signed(product_Acc1[1][37]);
    assign sum_Acc2_comb[1][19] = $signed(product_Acc1[1][38]) + $signed(product_Acc1[1][39]);
    assign sum_Acc2_comb[1][20] = $signed(product_Acc1[1][40]) + $signed(product_Acc1[1][41]);
    assign sum_Acc2_comb[1][21] = $signed(product_Acc1[1][42]) + $signed(product_Acc1[1][43]);
    assign sum_Acc2_comb[1][22] = $signed(product_Acc1[1][44]) + $signed(product_Acc1[1][45]);
    assign sum_Acc2_comb[1][23] = $signed(product_Acc1[1][46]) + $signed(product_Acc1[1][47]);
    assign sum_Acc2_comb[1][24] = $signed(product_Acc1[1][48]) + $signed(product_Acc1[1][49]);
    assign sum_Acc2_comb[1][25] = $signed(product_Acc1[1][50]) + $signed(product_Acc1[1][51]);
    assign sum_Acc2_comb[1][26] = $signed(product_Acc1[1][52]) + $signed(product_Acc1[1][53]);
    assign sum_Acc2_comb[1][27] = $signed(product_Acc1[1][54]) + $signed(product_Acc1[1][55]);
    assign sum_Acc2_comb[1][28] = $signed(product_Acc1[1][56]) + $signed(product_Acc1[1][57]);
    assign sum_Acc2_comb[1][29] = $signed(product_Acc1[1][58]) + $signed(product_Acc1[1][59]);
    assign sum_Acc2_comb[1][30] = $signed(product_Acc1[1][60]) + $signed(product_Acc1[1][61]);
    assign sum_Acc2_comb[1][31] = $signed(product_Acc1[1][62]) + $signed(product_Acc1[1][63]);
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Acc2 <= 0;
            reset_acc_Acc2 <= 0;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                bias_Acc2[lane] <= 0;
                addr_Acc2[lane] <= 0;
                for (term = 0;term < 32;term = term + 1)
                    sum_Acc2[lane][term] <= 0;
            end
        end
        else begin
            RES_write_en_Acc2 <= RES_write_en_Acc1;
            reset_acc_Acc2 <= reset_acc_Acc1;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                bias_Acc2[lane] <= bias_Acc1[lane];
                addr_Acc2[lane] <= addr_Acc1[lane];
                for (term = 0;term < 32;term = term + 1)
                    sum_Acc2[lane][term] <= sum_Acc2_comb[lane][term];
            end
        end
    end
    assign sum_Acc3_comb[0][0] = $signed(sum_Acc2[0][0]) + $signed(sum_Acc2[0][1]);
    assign sum_Acc3_comb[0][1] = $signed(sum_Acc2[0][2]) + $signed(sum_Acc2[0][3]);
    assign sum_Acc3_comb[0][2] = $signed(sum_Acc2[0][4]) + $signed(sum_Acc2[0][5]);
    assign sum_Acc3_comb[0][3] = $signed(sum_Acc2[0][6]) + $signed(sum_Acc2[0][7]);
    assign sum_Acc3_comb[0][4] = $signed(sum_Acc2[0][8]) + $signed(sum_Acc2[0][9]);
    assign sum_Acc3_comb[0][5] = $signed(sum_Acc2[0][10]) + $signed(sum_Acc2[0][11]);
    assign sum_Acc3_comb[0][6] = $signed(sum_Acc2[0][12]) + $signed(sum_Acc2[0][13]);
    assign sum_Acc3_comb[0][7] = $signed(sum_Acc2[0][14]) + $signed(sum_Acc2[0][15]);
    assign sum_Acc3_comb[0][8] = $signed(sum_Acc2[0][16]) + $signed(sum_Acc2[0][17]);
    assign sum_Acc3_comb[0][9] = $signed(sum_Acc2[0][18]) + $signed(sum_Acc2[0][19]);
    assign sum_Acc3_comb[0][10] = $signed(sum_Acc2[0][20]) + $signed(sum_Acc2[0][21]);
    assign sum_Acc3_comb[0][11] = $signed(sum_Acc2[0][22]) + $signed(sum_Acc2[0][23]);
    assign sum_Acc3_comb[0][12] = $signed(sum_Acc2[0][24]) + $signed(sum_Acc2[0][25]);
    assign sum_Acc3_comb[0][13] = $signed(sum_Acc2[0][26]) + $signed(sum_Acc2[0][27]);
    assign sum_Acc3_comb[0][14] = $signed(sum_Acc2[0][28]) + $signed(sum_Acc2[0][29]);
    assign sum_Acc3_comb[0][15] = $signed(sum_Acc2[0][30]) + $signed(sum_Acc2[0][31]);
    assign sum_Acc3_comb[1][0] = $signed(sum_Acc2[1][0]) + $signed(sum_Acc2[1][1]);
    assign sum_Acc3_comb[1][1] = $signed(sum_Acc2[1][2]) + $signed(sum_Acc2[1][3]);
    assign sum_Acc3_comb[1][2] = $signed(sum_Acc2[1][4]) + $signed(sum_Acc2[1][5]);
    assign sum_Acc3_comb[1][3] = $signed(sum_Acc2[1][6]) + $signed(sum_Acc2[1][7]);
    assign sum_Acc3_comb[1][4] = $signed(sum_Acc2[1][8]) + $signed(sum_Acc2[1][9]);
    assign sum_Acc3_comb[1][5] = $signed(sum_Acc2[1][10]) + $signed(sum_Acc2[1][11]);
    assign sum_Acc3_comb[1][6] = $signed(sum_Acc2[1][12]) + $signed(sum_Acc2[1][13]);
    assign sum_Acc3_comb[1][7] = $signed(sum_Acc2[1][14]) + $signed(sum_Acc2[1][15]);
    assign sum_Acc3_comb[1][8] = $signed(sum_Acc2[1][16]) + $signed(sum_Acc2[1][17]);
    assign sum_Acc3_comb[1][9] = $signed(sum_Acc2[1][18]) + $signed(sum_Acc2[1][19]);
    assign sum_Acc3_comb[1][10] = $signed(sum_Acc2[1][20]) + $signed(sum_Acc2[1][21]);
    assign sum_Acc3_comb[1][11] = $signed(sum_Acc2[1][22]) + $signed(sum_Acc2[1][23]);
    assign sum_Acc3_comb[1][12] = $signed(sum_Acc2[1][24]) + $signed(sum_Acc2[1][25]);
    assign sum_Acc3_comb[1][13] = $signed(sum_Acc2[1][26]) + $signed(sum_Acc2[1][27]);
    assign sum_Acc3_comb[1][14] = $signed(sum_Acc2[1][28]) + $signed(sum_Acc2[1][29]);
    assign sum_Acc3_comb[1][15] = $signed(sum_Acc2[1][30]) + $signed(sum_Acc2[1][31]);
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Acc3 <= 0;
            reset_acc_Acc3 <= 0;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                bias_Acc3[lane] <= 0;
                addr_Acc3[lane] <= 0;
                for (term = 0;term < 16;term = term + 1)
                    sum_Acc3[lane][term] <= 0;
            end
        end
        else begin
            RES_write_en_Acc3 <= RES_write_en_Acc2;
            reset_acc_Acc3 <= reset_acc_Acc2;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                bias_Acc3[lane] <= bias_Acc2[lane];
                addr_Acc3[lane] <= addr_Acc2[lane];
                for (term = 0;term < 16;term = term + 1)
                    sum_Acc3[lane][term] <= sum_Acc3_comb[lane][term];
            end
        end
    end
    assign sum_Acc4_comb[0][0] = $signed(sum_Acc3[0][0]) + $signed(sum_Acc3[0][1]);
    assign sum_Acc4_comb[0][1] = $signed(sum_Acc3[0][2]) + $signed(sum_Acc3[0][3]);
    assign sum_Acc4_comb[0][2] = $signed(sum_Acc3[0][4]) + $signed(sum_Acc3[0][5]);
    assign sum_Acc4_comb[0][3] = $signed(sum_Acc3[0][6]) + $signed(sum_Acc3[0][7]);
    assign sum_Acc4_comb[0][4] = $signed(sum_Acc3[0][8]) + $signed(sum_Acc3[0][9]);
    assign sum_Acc4_comb[0][5] = $signed(sum_Acc3[0][10]) + $signed(sum_Acc3[0][11]);
    assign sum_Acc4_comb[0][6] = $signed(sum_Acc3[0][12]) + $signed(sum_Acc3[0][13]);
    assign sum_Acc4_comb[0][7] = $signed(sum_Acc3[0][14]) + $signed(sum_Acc3[0][15]);
    assign sum_Acc4_comb[1][0] = $signed(sum_Acc3[1][0]) + $signed(sum_Acc3[1][1]);
    assign sum_Acc4_comb[1][1] = $signed(sum_Acc3[1][2]) + $signed(sum_Acc3[1][3]);
    assign sum_Acc4_comb[1][2] = $signed(sum_Acc3[1][4]) + $signed(sum_Acc3[1][5]);
    assign sum_Acc4_comb[1][3] = $signed(sum_Acc3[1][6]) + $signed(sum_Acc3[1][7]);
    assign sum_Acc4_comb[1][4] = $signed(sum_Acc3[1][8]) + $signed(sum_Acc3[1][9]);
    assign sum_Acc4_comb[1][5] = $signed(sum_Acc3[1][10]) + $signed(sum_Acc3[1][11]);
    assign sum_Acc4_comb[1][6] = $signed(sum_Acc3[1][12]) + $signed(sum_Acc3[1][13]);
    assign sum_Acc4_comb[1][7] = $signed(sum_Acc3[1][14]) + $signed(sum_Acc3[1][15]);
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Acc4 <= 0;
            reset_acc_Acc4 <= 0;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                bias_Acc4[lane] <= 0;
                addr_Acc4[lane] <= 0;
                for (term = 0;term < 8;term = term + 1)
                    sum_Acc4[lane][term] <= 0;
            end
        end
        else begin
            RES_write_en_Acc4 <= RES_write_en_Acc3;
            reset_acc_Acc4 <= reset_acc_Acc3;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                bias_Acc4[lane] <= bias_Acc3[lane];
                addr_Acc4[lane] <= addr_Acc3[lane];
                for (term = 0;term < 8;term = term + 1)
                    sum_Acc4[lane][term] <= sum_Acc4_comb[lane][term];
            end
        end
    end
    assign sum_Acc5_comb[0][0] = $signed(sum_Acc4[0][0]) + $signed(sum_Acc4[0][1]);
    assign sum_Acc5_comb[0][1] = $signed(sum_Acc4[0][2]) + $signed(sum_Acc4[0][3]);
    assign sum_Acc5_comb[0][2] = $signed(sum_Acc4[0][4]) + $signed(sum_Acc4[0][5]);
    assign sum_Acc5_comb[0][3] = $signed(sum_Acc4[0][6]) + $signed(sum_Acc4[0][7]);
    assign sum_Acc5_comb[1][0] = $signed(sum_Acc4[1][0]) + $signed(sum_Acc4[1][1]);
    assign sum_Acc5_comb[1][1] = $signed(sum_Acc4[1][2]) + $signed(sum_Acc4[1][3]);
    assign sum_Acc5_comb[1][2] = $signed(sum_Acc4[1][4]) + $signed(sum_Acc4[1][5]);
    assign sum_Acc5_comb[1][3] = $signed(sum_Acc4[1][6]) + $signed(sum_Acc4[1][7]);
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Acc5 <= 0;
            reset_acc_Acc5 <= 0;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                bias_Acc5[lane] <= 0;
                addr_Acc5[lane] <= 0;
                for (term = 0;term < 4;term = term + 1)
                    sum_Acc5[lane][term] <= 0;
            end
        end
        else begin
            RES_write_en_Acc5 <= RES_write_en_Acc4;
            reset_acc_Acc5 <= reset_acc_Acc4;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                bias_Acc5[lane] <= bias_Acc4[lane];
                addr_Acc5[lane] <= addr_Acc4[lane];
                for (term = 0;term < 4;term = term + 1)
                    sum_Acc5[lane][term] <= sum_Acc5_comb[lane][term];
            end
        end
    end
    assign sum_Acc6_comb[0][0] = $signed(sum_Acc5[0][0]) + $signed(sum_Acc5[0][1]);
    assign sum_Acc6_comb[0][1] = $signed(sum_Acc5[0][2]) + $signed(sum_Acc5[0][3]);
    assign sum_Acc6_comb[1][0] = $signed(sum_Acc5[1][0]) + $signed(sum_Acc5[1][1]);
    assign sum_Acc6_comb[1][1] = $signed(sum_Acc5[1][2]) + $signed(sum_Acc5[1][3]);
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Acc6 <= 0;
            reset_acc_Acc6 <= 0;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                bias_Acc6[lane] <= 0;
                addr_Acc6[lane] <= 0;
                for (term = 0;term < 2;term = term + 1)
                    sum_Acc6[lane][term] <= 0;
            end
        end
        else begin
            RES_write_en_Acc6 <= RES_write_en_Acc5;
            reset_acc_Acc6 <= reset_acc_Acc5;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                bias_Acc6[lane] <= bias_Acc5[lane];
                addr_Acc6[lane] <= addr_Acc5[lane];
                for (term = 0;term < 2;term = term + 1)
                    sum_Acc6[lane][term] <= sum_Acc6_comb[lane][term];
            end
        end
    end
    assign sum_Acc7_comb[0][0] = $signed(sum_Acc6[0][0]) + $signed(sum_Acc6[0][1]);
    assign sum_Acc7_comb[1][0] = $signed(sum_Acc6[1][0]) + $signed(sum_Acc6[1][1]);
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Acc7 <= 0;
            reset_acc_Acc7 <= 0;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                bias_Acc7[lane] <= 0;
                addr_Acc7[lane] <= 0;
                for (term = 0;term < 1;term = term + 1)
                    sum_Acc7[lane][term] <= 0;
            end
        end
        else begin
            RES_write_en_Acc7 <= RES_write_en_Acc6;
            reset_acc_Acc7 <= reset_acc_Acc6;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                bias_Acc7[lane] <= bias_Acc6[lane];
                addr_Acc7[lane] <= addr_Acc6[lane];
                for (term = 0;term < 1;term = term + 1)
                    sum_Acc7[lane][term] <= sum_Acc7_comb[lane][term];
            end
        end
    end
    assign accumulator_next[0] = reset_acc_Acc7 ? sum_Acc7[0][0]:$signed(accumulator_KAcc[0]) + $signed(sum_Acc7[0][0]);
    assign accumulator_next[1] = reset_acc_Acc7 ? sum_Acc7[1][0]:$signed(accumulator_KAcc[1]) + $signed(sum_Acc7[1][0]);
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_KAcc <= 0;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                accumulator_KAcc[lane] <= 0;
                bias_KAcc[lane] <= 0;
                addr_KAcc[lane] <= 0;
            end
        end
        else begin
            RES_write_en_KAcc <= RES_write_en_Acc7;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                accumulator_KAcc[lane] <= accumulator_next[lane];
                bias_KAcc[lane] <= bias_Acc7[lane];
                addr_KAcc[lane] <= addr_Acc7[lane];
            end
        end
    end
    // -------------------------------------------------------------------------
    // Stage 5: Bias Addition and Truncation
    // -------------------------------------------------------------------------
    assign aligned_bias[0] = $signed(bias_KAcc[0]) <<< FRAC_BITS;
    assign biased_value[0] = $signed(accumulator_KAcc[0]) + aligned_bias[0];
    assign truncated_value[0] = biased_value[0][width+FRAC_BITS-1:FRAC_BITS];
    // -------------------------------------------------------------------------
    // Stage 6: ReLU
    // -------------------------------------------------------------------------
    assign relu_value[0] = data_Relu[0][width-1] ? 0:data_Relu[0];
    assign aligned_bias[1] = $signed(bias_KAcc[1]) <<< FRAC_BITS;
    assign biased_value[1] = $signed(accumulator_KAcc[1]) + aligned_bias[1];
    assign truncated_value[1] = biased_value[1][width+FRAC_BITS-1:FRAC_BITS];
    assign relu_value[1] = data_Relu[1][width-1] ? 0:data_Relu[1];
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Relu <= 0;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                data_Relu[lane] <= 0;
                addr_Relu[lane] <= 0;
            end
        end
        else begin
            RES_write_en_Relu <= RES_write_en_KAcc;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                data_Relu[lane] <= truncated_value[lane];
                addr_Relu[lane] <= addr_KAcc[lane];
            end
        end
    end
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            RES_write_en_Out1 <= 0;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                data_Out1[lane] <= 0;
                addr_Out1[lane] <= 0;
            end
        end
        else begin
            RES_write_en_Out1 <= RES_write_en_Relu;
            for (lane = 0;lane < OUT_UNROLL;lane = lane + 1) begin
                data_Out1[lane] <= relu_value[lane];
                addr_Out1[lane] <= addr_Relu[lane];
            end
        end
    end
    // -------------------------------------------------------------------------
    // Stage 7: Output
    // -------------------------------------------------------------------------
    // OU2 writes both output channels in the same cycle through the TDP ports.
    assign RES_write_en_even = RES_write_en_Out1;
    assign RES_write_en_odd = RES_write_en_Out1;
    assign RES_write_address_even = addr_Out1[0];
    assign RES_write_address_odd = addr_Out1[1];
    assign RES_write_data_even = data_Out1[0];
    assign RES_write_data_odd = data_Out1[1];

    // -------------------------------------------------------------------------
    // Counter and Address Generation
    // -------------------------------------------------------------------------
    Counter #(
                .ADDR_WIDTH($clog2(IC_GROUPS)),
                .CNT_INIT(0),
                .MAX_VAL(IC_GROUPS - 1)
            ) ic_group_counter (
                .CLK(clk),
                .RESET(!rst || counter_clear_Read1),
                .ENABLE(en_ic_group),
                .CNT(ic_group)
            );
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
    Counter #(
                .ADDR_WIDTH($clog2(OC_DIM / OUT_UNROLL)),
                .CNT_INIT(0),
                .MAX_VAL(OC_DIM / OUT_UNROLL - 1)
            ) oc_counter (
                .CLK(clk),
                .RESET(!rst || counter_clear_Read1),
                .ENABLE(en_oc),
                .CNT(oc)
            );
    conv6_input_agu #(
                        .ADDR_WIDTH(A_depth_bits),
                        .IN_H(IN_H),
                        .IN_W(IN_W),
                        .K(K),
                        .PAD(1),
                        .IC_GROUPS(IC_GROUPS)
                    ) input_address_gen (
                        .ic_group(ic_group),
                        .row(row),
                        .col(col),
                        .k_row(k_row),
                        .k_col(k_col),
                        .A_read_address(A_read_address_Read1),
                        .out_of_bounds(out_of_bounds_Read1)
                    );
    conv6_weight_agu #(
                         .ADDR_WIDTH(B_depth_bits),
                         .OC_DIM(OC_DIM),
                         .OUTPUT_UNROLL(OUT_UNROLL),
                         .K(K),
                         .IC_GROUPS(IC_GROUPS)
                     ) weight_address_gen (
                         .oc(oc),
                         .ic_group(ic_group),
                         .k_row(k_row),
                         .k_col(k_col),
                         .B_read_address(B_read_address_Read1)
                     );
    conv6_res_agu #(
                      .ADDR_WIDTH(RES_depth_bits),
                      .OC_DIM(OC_DIM),
                      .OUTPUT_UNROLL(OUT_UNROLL),
                      .IN_H(IN_H),
                      .IN_W(IN_W)
                  ) result_address_gen (
                      .oc(oc),
                      .row(row),
                      .col(col),
                      .RES_write_address_0(calc_addr_Read1[0]),
                      .RES_write_address_1(calc_addr_Read1[1])
                  );
    assign C_read_address_Read1 = oc;
endmodule

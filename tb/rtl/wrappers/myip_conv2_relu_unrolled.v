`timescale 1ns / 1ps

/* Standalone AXI-stream wrapper for updated fused Conv2 + ReLU.
 * Input: activation[32][32][32], weights[32][32][3][3], biases[32].
 * Output: activation[32][32][32], channel-major.
 */
module myip_conv2_relu_unrolled(
        input ACLK,input ARESETN,output reg S_AXIS_TREADY,input[31:0]S_AXIS_TDATA,
        input S_AXIS_TLAST,input S_AXIS_TVALID,output reg M_AXIS_TVALID,
        output reg[31:0]M_AXIS_TDATA,output reg M_AXIS_TLAST,input M_AXIS_TREADY);
    localparam width=16,NUM_A=32768,NUM_B=9216,NUM_C=32,NUM_RES=32768;
    localparam Idle=4'b1000,Read_Inputs=4'b0100,Compute=4'b0010,Write_Outputs=4'b0001;
    reg[3:0]state,n_state;
    reg[15:0]read_counter;
    reg[14:0]write_counter;
    reg[31:0]A_we;
    reg[127:0]B_we;
    reg[3:0]C_we;
    reg[9:0]A_waddr;
    reg[6:0]B_waddr;
    reg[2:0]C_waddr;
    reg[3:0]weight_chunk_counter;
    reg[6:0]weight_ram_select;
    reg[6:0]weight_base_address;
    wire A_read_en,B_read_en,C_read_en;
    wire[9:0]A_read_address;
    wire[6:0]B_read_address;
    wire[2:0]C_read_address;
    wire[15:0]A_read_data[0:31];
    wire[15:0]B_read_data[0:127];
    wire[15:0]C_read_data[0:3];
    wire[127:0]A_read_data_packed[0:3];
    wire[63:0]B_read_data_packed[0:31];
    wire RES_write_en_even,RES_write_en_odd;
    wire[14:0]RES_write_address_even,RES_write_address_odd;
    wire[15:0]RES_write_data_even,RES_write_data_odd;
    wire consumer_frame_done,producer_frame_done;
    wire core_ready=(state==Compute);
    wire[15:0]output_read_data;
    reg[1:0]output_read_latency;
    reg output_word_valid;
    wire output_read_en=(state==Write_Outputs)&&!output_word_valid&&(output_read_latency==0);
    always@(posedge ACLK)if(!ARESETN)
            state<=Idle;
        else
            state<=n_state;
    always@(*) begin
        n_state=state;
        S_AXIS_TREADY=0;
        M_AXIS_TVALID=0;
        M_AXIS_TLAST=0;
        M_AXIS_TDATA={16'd0,output_read_data};
        A_we=0;
        B_we=0;
        C_we=0;
        A_waddr=0;
        B_waddr=0;
        C_waddr=0;
        case(state)
            Idle:
                if(S_AXIS_TVALID)
                    n_state=Read_Inputs;
            Read_Inputs: begin
                S_AXIS_TREADY=1;
                if(S_AXIS_TVALID) begin
                    if(read_counter<NUM_A) begin
                        A_we[read_counter[14:10]]=1;
                        A_waddr=read_counter[9:0];
                    end
                    else if(read_counter<NUM_A+NUM_B) begin
                        B_we[weight_ram_select]=1;
                        B_waddr=weight_base_address+weight_chunk_counter;
                    end
                    else begin
                        C_we[(read_counter-NUM_A-NUM_B)&3]=1;
                        C_waddr=(read_counter-NUM_A-NUM_B)>>2;
                    end
                    if(read_counter==NUM_A+NUM_B+NUM_C-1)
                        n_state=Compute;
                end
            end
            Compute:
                if(producer_frame_done)
                    n_state=Write_Outputs;
            Write_Outputs: begin
                M_AXIS_TVALID=output_word_valid;
                M_AXIS_TLAST=(write_counter==NUM_RES-1);
                if(output_word_valid&&M_AXIS_TREADY&&write_counter==NUM_RES-1)
                    n_state=Idle;
            end
        endcase
    end
    always@(posedge ACLK) begin
        if(!ARESETN) begin
            read_counter<=0;
            write_counter<=0;
            weight_chunk_counter<=0;
            weight_ram_select<=0;
            weight_base_address<=0;
            output_read_latency<=0;
            output_word_valid<=0;
        end
        else begin
            if(state==Read_Inputs&&S_AXIS_TVALID) begin
                if(read_counter>=NUM_A&&read_counter<NUM_A+NUM_B) begin
                    if(weight_chunk_counter==8) begin
                        weight_chunk_counter<=0;
                        if(weight_ram_select==127) begin
                            weight_ram_select<=0;
                            weight_base_address<=weight_base_address+9;
                        end
                        else
                            weight_ram_select<=weight_ram_select+1;
                    end
                    else
                        weight_chunk_counter<=weight_chunk_counter+1;
                end
                if(read_counter==NUM_A+NUM_B+NUM_C-1) begin
                    read_counter<=0;
                    weight_chunk_counter<=0;
                    weight_ram_select<=0;
                    weight_base_address<=0;
                end
                else
                    read_counter<=read_counter+1;
            end
            if(state==Write_Outputs&&M_AXIS_TVALID&&M_AXIS_TREADY)
                write_counter<=(write_counter==NUM_RES-1)?0:write_counter+1;
            if(state!=Write_Outputs) begin
                output_read_latency<=0;
                output_word_valid<=0;
            end
            else if(output_word_valid) begin
                if(M_AXIS_TREADY)
                    output_word_valid<=0;
            end
            else if(output_read_latency==2) begin
                output_read_latency<=0;
                output_word_valid<=1;
            end
            else
                output_read_latency<=output_read_latency+1;
        end
    end
    genvar g;
    generate
        // Match the Conv1-to-Conv2 boundary: eight activation channels share
        // each UltraRAM and return together after a three-cycle read.
        for(g=0;g<4;g=g+1) begin:INPUT_MEMORIES
            memory_uram_packed8_frame_2w#(.width(width),.depth_bits(10))input_RAM(
                                            .clk(ACLK),.write_en_0(|A_we[g*8 +: 8]),.write_channel_0(read_counter[12:10]),
                                            .write_address_0(A_waddr),.write_data_0(S_AXIS_TDATA[15:0]),
                                            .write_en_1(1'b0),.write_channel_1(3'b0),.write_address_1(10'b0),.write_data_1(16'b0),
                                            .read_en(A_read_en),.read_address(A_read_address),.read_data_out(A_read_data_packed[g]));
            assign A_read_data[g*8]   = A_read_data_packed[g][15:0];
            assign A_read_data[g*8+1] = A_read_data_packed[g][31:16];
            assign A_read_data[g*8+2] = A_read_data_packed[g][47:32];
            assign A_read_data[g*8+3] = A_read_data_packed[g][63:48];
            assign A_read_data[g*8+4] = A_read_data_packed[g][79:64];
            assign A_read_data[g*8+5] = A_read_data_packed[g][95:80];
            assign A_read_data[g*8+6] = A_read_data_packed[g][111:96];
            assign A_read_data[g*8+7] = A_read_data_packed[g][127:112];
        end
        // Four logical 128x16 weight banks share every TDP BRAM36.
        for(g=0;g<32;g=g+1) begin:WEIGHT_MEMORIES
            wire[3:0]packed_write_en={B_we[96+g],B_we[64+g],B_we[32+g],B_we[g]};
            memory_bram_tdp_packed4#(.width(width),.depth_bits(7))weight_RAM(
                                       .clk(ACLK),.write_en(packed_write_en),.write_address(B_waddr),.write_data_in(S_AXIS_TDATA[15:0]),
                                       .read_en(B_read_en),.read_address(B_read_address),.read_data_out(B_read_data_packed[g]));
            assign B_read_data[g]    = B_read_data_packed[g][15:0];
            assign B_read_data[32+g] = B_read_data_packed[g][31:16];
            assign B_read_data[64+g] = B_read_data_packed[g][47:32];
            assign B_read_data[96+g] = B_read_data_packed[g][63:48];
        end
        for(g=0;g<4;g=g+1) begin:BIAS_MEMORIES memory_lutram#(.width(width),.depth_bits(3))bias_RAM(
                .clk(ACLK),.write_en(C_we[g]),.write_address(C_waddr),.write_data_in(S_AXIS_TDATA[15:0]),.read_en(C_read_en),.read_address(C_read_address),.read_data_out(C_read_data[g]));
        end
    endgenerate
    memory_bram_tdp#(.width(width),.depth_bits(15))output_RAM(
                       .clk(ACLK),.port_a_enable((state==Compute)?RES_write_en_even:output_read_en),
                       .port_a_write_en((state==Compute)?RES_write_en_even:1'b0),.port_a_address((state==Compute)?RES_write_address_even:write_counter),
                       .port_a_data_in(RES_write_data_even),.port_a_data_out(output_read_data),
                       .port_b_enable((state==Compute)?RES_write_en_odd:1'b0),.port_b_write_en((state==Compute)?RES_write_en_odd:1'b0),
                       .port_b_address(RES_write_address_odd),.port_b_data_in(RES_write_data_odd),.port_b_data_out());
    conv2_relu_unrolled_ping_pong#(.A_depth_bits(10),.B_depth_bits(7),.C_depth_bits(3),.RES_depth_bits(15),
                                   .IC_DIM(32),.OC_DIM(32),.OUT_UNROLL(4),.IN_H(32),.IN_W(32))core(
                                     .clk(ACLK),.rst(ARESETN),.consumer_ready(core_ready),.producer_ready(core_ready),
                                     .consumer_frame_done(consumer_frame_done),.producer_frame_done(producer_frame_done),
                                     .A_read_en(A_read_en),.A_read_address(A_read_address),
                                     .A_read_data_out_0(A_read_data[0]),
                                     .A_read_data_out_1(A_read_data[1]),
                                     .A_read_data_out_2(A_read_data[2]),
                                     .A_read_data_out_3(A_read_data[3]),
                                     .A_read_data_out_4(A_read_data[4]),
                                     .A_read_data_out_5(A_read_data[5]),
                                     .A_read_data_out_6(A_read_data[6]),
                                     .A_read_data_out_7(A_read_data[7]),
                                     .A_read_data_out_8(A_read_data[8]),
                                     .A_read_data_out_9(A_read_data[9]),
                                     .A_read_data_out_10(A_read_data[10]),
                                     .A_read_data_out_11(A_read_data[11]),
                                     .A_read_data_out_12(A_read_data[12]),
                                     .A_read_data_out_13(A_read_data[13]),
                                     .A_read_data_out_14(A_read_data[14]),
                                     .A_read_data_out_15(A_read_data[15]),
                                     .A_read_data_out_16(A_read_data[16]),
                                     .A_read_data_out_17(A_read_data[17]),
                                     .A_read_data_out_18(A_read_data[18]),
                                     .A_read_data_out_19(A_read_data[19]),
                                     .A_read_data_out_20(A_read_data[20]),
                                     .A_read_data_out_21(A_read_data[21]),
                                     .A_read_data_out_22(A_read_data[22]),
                                     .A_read_data_out_23(A_read_data[23]),
                                     .A_read_data_out_24(A_read_data[24]),
                                     .A_read_data_out_25(A_read_data[25]),
                                     .A_read_data_out_26(A_read_data[26]),
                                     .A_read_data_out_27(A_read_data[27]),
                                     .A_read_data_out_28(A_read_data[28]),
                                     .A_read_data_out_29(A_read_data[29]),
                                     .A_read_data_out_30(A_read_data[30]),
                                     .A_read_data_out_31(A_read_data[31]),
                                     .B_read_en(B_read_en),.B_read_address(B_read_address),
                                     .B_read_data_out_0_oc0(B_read_data[0]),
                                     .B_read_data_out_1_oc0(B_read_data[1]),
                                     .B_read_data_out_2_oc0(B_read_data[2]),
                                     .B_read_data_out_3_oc0(B_read_data[3]),
                                     .B_read_data_out_4_oc0(B_read_data[4]),
                                     .B_read_data_out_5_oc0(B_read_data[5]),
                                     .B_read_data_out_6_oc0(B_read_data[6]),
                                     .B_read_data_out_7_oc0(B_read_data[7]),
                                     .B_read_data_out_8_oc0(B_read_data[8]),
                                     .B_read_data_out_9_oc0(B_read_data[9]),
                                     .B_read_data_out_10_oc0(B_read_data[10]),
                                     .B_read_data_out_11_oc0(B_read_data[11]),
                                     .B_read_data_out_12_oc0(B_read_data[12]),
                                     .B_read_data_out_13_oc0(B_read_data[13]),
                                     .B_read_data_out_14_oc0(B_read_data[14]),
                                     .B_read_data_out_15_oc0(B_read_data[15]),
                                     .B_read_data_out_16_oc0(B_read_data[16]),
                                     .B_read_data_out_17_oc0(B_read_data[17]),
                                     .B_read_data_out_18_oc0(B_read_data[18]),
                                     .B_read_data_out_19_oc0(B_read_data[19]),
                                     .B_read_data_out_20_oc0(B_read_data[20]),
                                     .B_read_data_out_21_oc0(B_read_data[21]),
                                     .B_read_data_out_22_oc0(B_read_data[22]),
                                     .B_read_data_out_23_oc0(B_read_data[23]),
                                     .B_read_data_out_24_oc0(B_read_data[24]),
                                     .B_read_data_out_25_oc0(B_read_data[25]),
                                     .B_read_data_out_26_oc0(B_read_data[26]),
                                     .B_read_data_out_27_oc0(B_read_data[27]),
                                     .B_read_data_out_28_oc0(B_read_data[28]),
                                     .B_read_data_out_29_oc0(B_read_data[29]),
                                     .B_read_data_out_30_oc0(B_read_data[30]),
                                     .B_read_data_out_31_oc0(B_read_data[31]),
                                     .B_read_data_out_0_oc1(B_read_data[32]),
                                     .B_read_data_out_1_oc1(B_read_data[33]),
                                     .B_read_data_out_2_oc1(B_read_data[34]),
                                     .B_read_data_out_3_oc1(B_read_data[35]),
                                     .B_read_data_out_4_oc1(B_read_data[36]),
                                     .B_read_data_out_5_oc1(B_read_data[37]),
                                     .B_read_data_out_6_oc1(B_read_data[38]),
                                     .B_read_data_out_7_oc1(B_read_data[39]),
                                     .B_read_data_out_8_oc1(B_read_data[40]),
                                     .B_read_data_out_9_oc1(B_read_data[41]),
                                     .B_read_data_out_10_oc1(B_read_data[42]),
                                     .B_read_data_out_11_oc1(B_read_data[43]),
                                     .B_read_data_out_12_oc1(B_read_data[44]),
                                     .B_read_data_out_13_oc1(B_read_data[45]),
                                     .B_read_data_out_14_oc1(B_read_data[46]),
                                     .B_read_data_out_15_oc1(B_read_data[47]),
                                     .B_read_data_out_16_oc1(B_read_data[48]),
                                     .B_read_data_out_17_oc1(B_read_data[49]),
                                     .B_read_data_out_18_oc1(B_read_data[50]),
                                     .B_read_data_out_19_oc1(B_read_data[51]),
                                     .B_read_data_out_20_oc1(B_read_data[52]),
                                     .B_read_data_out_21_oc1(B_read_data[53]),
                                     .B_read_data_out_22_oc1(B_read_data[54]),
                                     .B_read_data_out_23_oc1(B_read_data[55]),
                                     .B_read_data_out_24_oc1(B_read_data[56]),
                                     .B_read_data_out_25_oc1(B_read_data[57]),
                                     .B_read_data_out_26_oc1(B_read_data[58]),
                                     .B_read_data_out_27_oc1(B_read_data[59]),
                                     .B_read_data_out_28_oc1(B_read_data[60]),
                                     .B_read_data_out_29_oc1(B_read_data[61]),
                                     .B_read_data_out_30_oc1(B_read_data[62]),
                                     .B_read_data_out_31_oc1(B_read_data[63]),
                                     .B_read_data_out_0_oc2(B_read_data[64]),
                                     .B_read_data_out_1_oc2(B_read_data[65]),
                                     .B_read_data_out_2_oc2(B_read_data[66]),
                                     .B_read_data_out_3_oc2(B_read_data[67]),
                                     .B_read_data_out_4_oc2(B_read_data[68]),
                                     .B_read_data_out_5_oc2(B_read_data[69]),
                                     .B_read_data_out_6_oc2(B_read_data[70]),
                                     .B_read_data_out_7_oc2(B_read_data[71]),
                                     .B_read_data_out_8_oc2(B_read_data[72]),
                                     .B_read_data_out_9_oc2(B_read_data[73]),
                                     .B_read_data_out_10_oc2(B_read_data[74]),
                                     .B_read_data_out_11_oc2(B_read_data[75]),
                                     .B_read_data_out_12_oc2(B_read_data[76]),
                                     .B_read_data_out_13_oc2(B_read_data[77]),
                                     .B_read_data_out_14_oc2(B_read_data[78]),
                                     .B_read_data_out_15_oc2(B_read_data[79]),
                                     .B_read_data_out_16_oc2(B_read_data[80]),
                                     .B_read_data_out_17_oc2(B_read_data[81]),
                                     .B_read_data_out_18_oc2(B_read_data[82]),
                                     .B_read_data_out_19_oc2(B_read_data[83]),
                                     .B_read_data_out_20_oc2(B_read_data[84]),
                                     .B_read_data_out_21_oc2(B_read_data[85]),
                                     .B_read_data_out_22_oc2(B_read_data[86]),
                                     .B_read_data_out_23_oc2(B_read_data[87]),
                                     .B_read_data_out_24_oc2(B_read_data[88]),
                                     .B_read_data_out_25_oc2(B_read_data[89]),
                                     .B_read_data_out_26_oc2(B_read_data[90]),
                                     .B_read_data_out_27_oc2(B_read_data[91]),
                                     .B_read_data_out_28_oc2(B_read_data[92]),
                                     .B_read_data_out_29_oc2(B_read_data[93]),
                                     .B_read_data_out_30_oc2(B_read_data[94]),
                                     .B_read_data_out_31_oc2(B_read_data[95]),
                                     .B_read_data_out_0_oc3(B_read_data[96]),
                                     .B_read_data_out_1_oc3(B_read_data[97]),
                                     .B_read_data_out_2_oc3(B_read_data[98]),
                                     .B_read_data_out_3_oc3(B_read_data[99]),
                                     .B_read_data_out_4_oc3(B_read_data[100]),
                                     .B_read_data_out_5_oc3(B_read_data[101]),
                                     .B_read_data_out_6_oc3(B_read_data[102]),
                                     .B_read_data_out_7_oc3(B_read_data[103]),
                                     .B_read_data_out_8_oc3(B_read_data[104]),
                                     .B_read_data_out_9_oc3(B_read_data[105]),
                                     .B_read_data_out_10_oc3(B_read_data[106]),
                                     .B_read_data_out_11_oc3(B_read_data[107]),
                                     .B_read_data_out_12_oc3(B_read_data[108]),
                                     .B_read_data_out_13_oc3(B_read_data[109]),
                                     .B_read_data_out_14_oc3(B_read_data[110]),
                                     .B_read_data_out_15_oc3(B_read_data[111]),
                                     .B_read_data_out_16_oc3(B_read_data[112]),
                                     .B_read_data_out_17_oc3(B_read_data[113]),
                                     .B_read_data_out_18_oc3(B_read_data[114]),
                                     .B_read_data_out_19_oc3(B_read_data[115]),
                                     .B_read_data_out_20_oc3(B_read_data[116]),
                                     .B_read_data_out_21_oc3(B_read_data[117]),
                                     .B_read_data_out_22_oc3(B_read_data[118]),
                                     .B_read_data_out_23_oc3(B_read_data[119]),
                                     .B_read_data_out_24_oc3(B_read_data[120]),
                                     .B_read_data_out_25_oc3(B_read_data[121]),
                                     .B_read_data_out_26_oc3(B_read_data[122]),
                                     .B_read_data_out_27_oc3(B_read_data[123]),
                                     .B_read_data_out_28_oc3(B_read_data[124]),
                                     .B_read_data_out_29_oc3(B_read_data[125]),
                                     .B_read_data_out_30_oc3(B_read_data[126]),
                                     .B_read_data_out_31_oc3(B_read_data[127]),
                                     .C_read_en(C_read_en),.C_read_address(C_read_address),
                                     .C_read_data_out_oc0(C_read_data[0]),
                                     .C_read_data_out_oc1(C_read_data[1]),
                                     .C_read_data_out_oc2(C_read_data[2]),
                                     .C_read_data_out_oc3(C_read_data[3]),
                                     .RES_write_en_even(RES_write_en_even),.RES_write_address_even(RES_write_address_even),.RES_write_data_in_even(RES_write_data_even),
                                     .RES_write_en_odd(RES_write_en_odd),.RES_write_address_odd(RES_write_address_odd),.RES_write_data_in_odd(RES_write_data_odd));
endmodule

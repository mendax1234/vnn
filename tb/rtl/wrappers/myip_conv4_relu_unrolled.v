`timescale 1ns / 1ps

/* Standalone AXI-stream wrapper for updated fused Conv4 + ReLU.
 * Input: activation[64][16][16], weights[64][64][3][3], biases[64].
 * Output: activation[64][16][16], channel-major. */
module myip_conv4_relu_unrolled(input ACLK,input ARESETN,output reg S_AXIS_TREADY,input[31:0]S_AXIS_TDATA,
                                    input S_AXIS_TLAST,input S_AXIS_TVALID,output reg M_AXIS_TVALID,output reg[31:0]M_AXIS_TDATA,output reg M_AXIS_TLAST,input M_AXIS_TREADY);
    localparam width=16,NUM_A=16384,NUM_B=36864,NUM_C=64,NUM_RES=16384;
    localparam Idle=4'b1000,Read_Inputs=4'b0100,Compute=4'b0010,Write_Outputs=4'b0001;
    reg[3:0]state,n_state;
    reg[15:0]read_counter;
    reg[13:0]write_counter;
    reg[63:0]A_we;
    reg[127:0]B_we;
    reg[1:0]C_we;
    reg[7:0]A_waddr;
    reg[8:0]B_waddr,weight_base_address;
    reg[4:0]C_waddr;
    reg[3:0]weight_chunk_counter;
    reg[6:0]weight_ram_select;
    wire A_read_en,B_read_en,C_read_en;
    wire[7:0]A_read_address;
    wire[8:0]B_read_address;
    wire[4:0]C_read_address;
    wire[15:0]A_read_data[0:63];
    wire[15:0]B_read_data[0:127];
    wire[15:0]C_read_data[0:1];
    wire RES_write_en_even,RES_write_en_odd;
    wire[13:0]RES_write_address_even,RES_write_address_odd;
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
                        A_we[read_counter[13:8]]=1;
                        A_waddr=read_counter[7:0];
                    end
                    else if(read_counter<NUM_A+NUM_B) begin
                        B_we[weight_ram_select]=1;
                        B_waddr=weight_base_address+weight_chunk_counter;
                    end
                    else begin
                        C_we[(read_counter-NUM_A-NUM_B)&1]=1;
                        C_waddr=(read_counter-NUM_A-NUM_B)>>1;
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
    wire[63:0]B_packed_data[0:31];
    genvar g;
    generate
        for(g=0;g<64;g=g+1) begin:INPUT_MEMORIES
            memory_bram_tdp#(.width(width),.depth_bits(8))input_RAM(
                               .clk(ACLK),.port_a_enable(A_we[g]),.port_a_write_en(A_we[g]),
                               .port_a_address(A_waddr),.port_a_data_in(S_AXIS_TDATA[15:0]),.port_a_data_out(),
                               .port_b_enable(A_read_en),.port_b_write_en(1'b0),.port_b_address(A_read_address),
                               .port_b_data_in(16'd0),.port_b_data_out(A_read_data[g]));
        end
        for(g=0;g<32;g=g+1) begin:WEIGHT_MEMORIES
            wire[3:0]packed_write_en={B_we[96+g],B_we[64+g],B_we[32+g],B_we[g]};
            memory_bram_tdp_packed4#(.width(width),.depth_bits(9))weight_RAM(
                                       .clk(ACLK),.write_en(packed_write_en),.write_address(B_waddr),
                                       .write_data_in(S_AXIS_TDATA[15:0]),.read_en(B_read_en),
                                       .read_address(B_read_address),.read_data_out(B_packed_data[g]));
            assign B_read_data[g]=B_packed_data[g][15:0];
            assign B_read_data[32+g]=B_packed_data[g][31:16];
            assign B_read_data[64+g]=B_packed_data[g][47:32];
            assign B_read_data[96+g]=B_packed_data[g][63:48];
        end
        for(g=0;g<2;g=g+1) begin:BIAS_MEMORIES memory_lutram#(.width(width),.depth_bits(5))bias_RAM(.clk(ACLK),.write_en(C_we[g]),.write_address(C_waddr),.write_data_in(S_AXIS_TDATA[15:0]),.read_en(C_read_en),.read_address(C_read_address),.read_data_out(C_read_data[g]));end
    endgenerate
    memory_bram_tdp#(.width(width),.depth_bits(14))output_RAM(.clk(ACLK),
                   .port_a_enable((state==Compute)?RES_write_en_even:output_read_en),.port_a_write_en((state==Compute)?RES_write_en_even:1'b0),.port_a_address((state==Compute)?RES_write_address_even:write_counter),.port_a_data_in(RES_write_data_even),.port_a_data_out(output_read_data),
                   .port_b_enable((state==Compute)?RES_write_en_odd:1'b0),.port_b_write_en((state==Compute)?RES_write_en_odd:1'b0),.port_b_address(RES_write_address_odd),.port_b_data_in(RES_write_data_odd),.port_b_data_out());
    conv4_relu_unrolled_ping_pong#(.IC_DIM(64),.OC_DIM(64),.OUT_UNROLL(2),.IN_H(16),.IN_W(16),.A_depth_bits(8),.B_depth_bits(9),.C_depth_bits(5),.RES_depth_bits(14))core(
                                     .clk(ACLK),.rst(ARESETN),.consumer_ready(core_ready),.producer_ready(core_ready),.consumer_frame_done(consumer_frame_done),.producer_frame_done(producer_frame_done),.A_read_en(A_read_en),.A_read_address(A_read_address),
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
                                     .A_read_data_out_32(A_read_data[32]),
                                     .A_read_data_out_33(A_read_data[33]),
                                     .A_read_data_out_34(A_read_data[34]),
                                     .A_read_data_out_35(A_read_data[35]),
                                     .A_read_data_out_36(A_read_data[36]),
                                     .A_read_data_out_37(A_read_data[37]),
                                     .A_read_data_out_38(A_read_data[38]),
                                     .A_read_data_out_39(A_read_data[39]),
                                     .A_read_data_out_40(A_read_data[40]),
                                     .A_read_data_out_41(A_read_data[41]),
                                     .A_read_data_out_42(A_read_data[42]),
                                     .A_read_data_out_43(A_read_data[43]),
                                     .A_read_data_out_44(A_read_data[44]),
                                     .A_read_data_out_45(A_read_data[45]),
                                     .A_read_data_out_46(A_read_data[46]),
                                     .A_read_data_out_47(A_read_data[47]),
                                     .A_read_data_out_48(A_read_data[48]),
                                     .A_read_data_out_49(A_read_data[49]),
                                     .A_read_data_out_50(A_read_data[50]),
                                     .A_read_data_out_51(A_read_data[51]),
                                     .A_read_data_out_52(A_read_data[52]),
                                     .A_read_data_out_53(A_read_data[53]),
                                     .A_read_data_out_54(A_read_data[54]),
                                     .A_read_data_out_55(A_read_data[55]),
                                     .A_read_data_out_56(A_read_data[56]),
                                     .A_read_data_out_57(A_read_data[57]),
                                     .A_read_data_out_58(A_read_data[58]),
                                     .A_read_data_out_59(A_read_data[59]),
                                     .A_read_data_out_60(A_read_data[60]),
                                     .A_read_data_out_61(A_read_data[61]),
                                     .A_read_data_out_62(A_read_data[62]),
                                     .A_read_data_out_63(A_read_data[63]),
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
                                     .B_read_data_out_32_oc0(B_read_data[32]),
                                     .B_read_data_out_33_oc0(B_read_data[33]),
                                     .B_read_data_out_34_oc0(B_read_data[34]),
                                     .B_read_data_out_35_oc0(B_read_data[35]),
                                     .B_read_data_out_36_oc0(B_read_data[36]),
                                     .B_read_data_out_37_oc0(B_read_data[37]),
                                     .B_read_data_out_38_oc0(B_read_data[38]),
                                     .B_read_data_out_39_oc0(B_read_data[39]),
                                     .B_read_data_out_40_oc0(B_read_data[40]),
                                     .B_read_data_out_41_oc0(B_read_data[41]),
                                     .B_read_data_out_42_oc0(B_read_data[42]),
                                     .B_read_data_out_43_oc0(B_read_data[43]),
                                     .B_read_data_out_44_oc0(B_read_data[44]),
                                     .B_read_data_out_45_oc0(B_read_data[45]),
                                     .B_read_data_out_46_oc0(B_read_data[46]),
                                     .B_read_data_out_47_oc0(B_read_data[47]),
                                     .B_read_data_out_48_oc0(B_read_data[48]),
                                     .B_read_data_out_49_oc0(B_read_data[49]),
                                     .B_read_data_out_50_oc0(B_read_data[50]),
                                     .B_read_data_out_51_oc0(B_read_data[51]),
                                     .B_read_data_out_52_oc0(B_read_data[52]),
                                     .B_read_data_out_53_oc0(B_read_data[53]),
                                     .B_read_data_out_54_oc0(B_read_data[54]),
                                     .B_read_data_out_55_oc0(B_read_data[55]),
                                     .B_read_data_out_56_oc0(B_read_data[56]),
                                     .B_read_data_out_57_oc0(B_read_data[57]),
                                     .B_read_data_out_58_oc0(B_read_data[58]),
                                     .B_read_data_out_59_oc0(B_read_data[59]),
                                     .B_read_data_out_60_oc0(B_read_data[60]),
                                     .B_read_data_out_61_oc0(B_read_data[61]),
                                     .B_read_data_out_62_oc0(B_read_data[62]),
                                     .B_read_data_out_63_oc0(B_read_data[63]),
                                     .B_read_data_out_0_oc1(B_read_data[64]),
                                     .B_read_data_out_1_oc1(B_read_data[65]),
                                     .B_read_data_out_2_oc1(B_read_data[66]),
                                     .B_read_data_out_3_oc1(B_read_data[67]),
                                     .B_read_data_out_4_oc1(B_read_data[68]),
                                     .B_read_data_out_5_oc1(B_read_data[69]),
                                     .B_read_data_out_6_oc1(B_read_data[70]),
                                     .B_read_data_out_7_oc1(B_read_data[71]),
                                     .B_read_data_out_8_oc1(B_read_data[72]),
                                     .B_read_data_out_9_oc1(B_read_data[73]),
                                     .B_read_data_out_10_oc1(B_read_data[74]),
                                     .B_read_data_out_11_oc1(B_read_data[75]),
                                     .B_read_data_out_12_oc1(B_read_data[76]),
                                     .B_read_data_out_13_oc1(B_read_data[77]),
                                     .B_read_data_out_14_oc1(B_read_data[78]),
                                     .B_read_data_out_15_oc1(B_read_data[79]),
                                     .B_read_data_out_16_oc1(B_read_data[80]),
                                     .B_read_data_out_17_oc1(B_read_data[81]),
                                     .B_read_data_out_18_oc1(B_read_data[82]),
                                     .B_read_data_out_19_oc1(B_read_data[83]),
                                     .B_read_data_out_20_oc1(B_read_data[84]),
                                     .B_read_data_out_21_oc1(B_read_data[85]),
                                     .B_read_data_out_22_oc1(B_read_data[86]),
                                     .B_read_data_out_23_oc1(B_read_data[87]),
                                     .B_read_data_out_24_oc1(B_read_data[88]),
                                     .B_read_data_out_25_oc1(B_read_data[89]),
                                     .B_read_data_out_26_oc1(B_read_data[90]),
                                     .B_read_data_out_27_oc1(B_read_data[91]),
                                     .B_read_data_out_28_oc1(B_read_data[92]),
                                     .B_read_data_out_29_oc1(B_read_data[93]),
                                     .B_read_data_out_30_oc1(B_read_data[94]),
                                     .B_read_data_out_31_oc1(B_read_data[95]),
                                     .B_read_data_out_32_oc1(B_read_data[96]),
                                     .B_read_data_out_33_oc1(B_read_data[97]),
                                     .B_read_data_out_34_oc1(B_read_data[98]),
                                     .B_read_data_out_35_oc1(B_read_data[99]),
                                     .B_read_data_out_36_oc1(B_read_data[100]),
                                     .B_read_data_out_37_oc1(B_read_data[101]),
                                     .B_read_data_out_38_oc1(B_read_data[102]),
                                     .B_read_data_out_39_oc1(B_read_data[103]),
                                     .B_read_data_out_40_oc1(B_read_data[104]),
                                     .B_read_data_out_41_oc1(B_read_data[105]),
                                     .B_read_data_out_42_oc1(B_read_data[106]),
                                     .B_read_data_out_43_oc1(B_read_data[107]),
                                     .B_read_data_out_44_oc1(B_read_data[108]),
                                     .B_read_data_out_45_oc1(B_read_data[109]),
                                     .B_read_data_out_46_oc1(B_read_data[110]),
                                     .B_read_data_out_47_oc1(B_read_data[111]),
                                     .B_read_data_out_48_oc1(B_read_data[112]),
                                     .B_read_data_out_49_oc1(B_read_data[113]),
                                     .B_read_data_out_50_oc1(B_read_data[114]),
                                     .B_read_data_out_51_oc1(B_read_data[115]),
                                     .B_read_data_out_52_oc1(B_read_data[116]),
                                     .B_read_data_out_53_oc1(B_read_data[117]),
                                     .B_read_data_out_54_oc1(B_read_data[118]),
                                     .B_read_data_out_55_oc1(B_read_data[119]),
                                     .B_read_data_out_56_oc1(B_read_data[120]),
                                     .B_read_data_out_57_oc1(B_read_data[121]),
                                     .B_read_data_out_58_oc1(B_read_data[122]),
                                     .B_read_data_out_59_oc1(B_read_data[123]),
                                     .B_read_data_out_60_oc1(B_read_data[124]),
                                     .B_read_data_out_61_oc1(B_read_data[125]),
                                     .B_read_data_out_62_oc1(B_read_data[126]),
                                     .B_read_data_out_63_oc1(B_read_data[127]),
                                     .C_read_en(C_read_en),.C_read_address(C_read_address),
                                     .C_read_data_out_oc0(C_read_data[0]),
                                     .C_read_data_out_oc1(C_read_data[1]),
                                     .RES_write_en_even(RES_write_en_even),.RES_write_address_even(RES_write_address_even),.RES_write_data_in_even(RES_write_data_even),
                                     .RES_write_en_odd(RES_write_en_odd),.RES_write_address_odd(RES_write_address_odd),.RES_write_data_in_odd(RES_write_data_odd));
endmodule

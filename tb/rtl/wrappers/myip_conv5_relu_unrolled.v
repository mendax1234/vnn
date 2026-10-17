`timescale 1ns / 1ps
/* Standalone AXI-stream wrapper for fused Conv5 + ReLU, OC unroll 1. */
module myip_conv5_relu_unrolled(input ACLK,input ARESETN,output reg S_AXIS_TREADY,input[31:0]S_AXIS_TDATA,input S_AXIS_TLAST,input S_AXIS_TVALID,output reg M_AXIS_TVALID,output reg[31:0]M_AXIS_TDATA,output reg M_AXIS_TLAST,input M_AXIS_TREADY);
    localparam width=16,NUM_A=4096,NUM_B=73728,NUM_C=128,NUM_RES=8192;
    localparam IC_DIM_LOCAL=64;
    localparam Idle=4'b1000,Read_Inputs=4'b0100,Compute=4'b0010,Write_Outputs=4'b0001;
    reg[3:0]state,n_state;
    reg[16:0]read_counter;
    reg[12:0]write_counter;
    reg[63:0]A_we;
    reg[63:0]B_we;
    reg C_we;
    reg[5:0]A_waddr;
    reg[6:0]C_waddr;
    reg[10:0]B_waddr[0:63];
    reg[3:0]weight_kernel_counter;
    reg[5:0]weight_ic_counter;
    wire[5:0]weight_ram_select=weight_ic_counter[5:0];
    wire A_read_en,B_read_en,C_read_en;
    wire[5:0]A_read_address;
    wire[10:0]B_read_address;
    wire[6:0]C_read_address;
    wire[15:0]A_read_data[0:63];
    wire[15:0]B_read_data[0:63];
    wire[15:0]C_read_data;
    wire RES_write_en;
    wire[12:0]RES_write_address;
    wire[15:0]RES_write_data;
    wire consumer_frame_done,producer_frame_done;
    wire core_ready=(state==Compute);
    wire[15:0]output_read_data;
    reg[1:0]output_read_latency;
    reg output_word_valid;
    wire output_read_en=(state==Write_Outputs);
    integer i;
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
        C_waddr=0;
        case(state)
            Idle:
                if(S_AXIS_TVALID)
                    n_state=Read_Inputs;
            Read_Inputs: begin
                S_AXIS_TREADY=1;
                if(S_AXIS_TVALID) begin
                    if(read_counter<NUM_A) begin
                        A_we[read_counter[11:6]]=1;
                        A_waddr=read_counter[5:0];
                    end
                    else if(read_counter<NUM_A+NUM_B)
                        B_we[weight_ram_select]=1;
                    else begin
                        C_we=1;
                        C_waddr=read_counter-NUM_A-NUM_B;
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
            weight_kernel_counter<=0;
            weight_ic_counter<=0;
            output_read_latency<=0;
            output_word_valid<=0;
            for(i=0;i<64;i=i+1)
                B_waddr[i]<=0;
        end
        else begin
            if(state==Read_Inputs&&S_AXIS_TVALID) begin
                if(read_counter>=NUM_A&&read_counter<NUM_A+NUM_B) begin
                    B_waddr[weight_ram_select]<=B_waddr[weight_ram_select]+1'b1;
                    if(weight_kernel_counter==8) begin
                        weight_kernel_counter<=0;
                        if(weight_ic_counter==IC_DIM_LOCAL-1) begin
                            weight_ic_counter<=0;
                        end
                        else
                            weight_ic_counter<=weight_ic_counter+1'b1;
                    end
                    else
                        weight_kernel_counter<=weight_kernel_counter+1'b1;
                end
                if(read_counter==NUM_A+NUM_B+NUM_C-1) begin
                    read_counter<=0;
                    weight_kernel_counter<=0;
                    weight_ic_counter<=0;
                    for(i=0;i<64;i=i+1)
                        B_waddr[i]<=0;
                end
                else
                    read_counter<=read_counter+1'b1;
            end
            if(state==Write_Outputs&&M_AXIS_TVALID&&M_AXIS_TREADY)
                write_counter<=(write_counter==NUM_RES-1)?0:write_counter+1'b1;
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
                output_read_latency<=output_read_latency+1'b1;
        end
    end
    wire[127:0]B_uram_packed_data[0:7];
    genvar g,h,output_bank;
    generate
        for(g=0;g<64;g=g+1) begin:INPUT_MEMORIES memory_lutram#(.width(width),.depth_bits(6))input_RAM(.clk(ACLK),.write_en(A_we[g]),.write_address(A_waddr),.write_data_in(S_AXIS_TDATA[15:0]),.read_en(A_read_en),.read_address(A_read_address),.read_data_out(A_read_data[g]));end
        for(g=0;g<8;g=g+1) begin:WEIGHT_URAMS
            wire[7:0]packed_write_en=B_we[g*8+:8];
            wire[10:0]packed_write_address=
                B_we[g*8+0]?B_waddr[g*8+0]:B_we[g*8+1]?B_waddr[g*8+1]:
                B_we[g*8+2]?B_waddr[g*8+2]:B_we[g*8+3]?B_waddr[g*8+3]:
                B_we[g*8+4]?B_waddr[g*8+4]:B_we[g*8+5]?B_waddr[g*8+5]:
                B_we[g*8+6]?B_waddr[g*8+6]:B_waddr[g*8+7];
            memory_uram_tdp_packed8#(.width(width),.depth_bits(11))weight_RAM(
                                       .clk(ACLK),.write_en(packed_write_en),.write_address(packed_write_address),
                                       .write_data_in(S_AXIS_TDATA[15:0]),.read_en(B_read_en),
                                       .read_address(B_read_address),.read_data_out(B_uram_packed_data[g]));
            for(h=0;h<8;h=h+1) begin:UNPACK
                assign B_read_data[g*8+h]=B_uram_packed_data[g][h*16+:16];
            end
        end
    endgenerate
    memory_lutram#(.width(width),.depth_bits(7))bias_RAM(
                     .clk(ACLK),.write_en(C_we),.write_address(C_waddr),
                     .write_data_in(S_AXIS_TDATA[15:0]),.read_en(C_read_en),
                     .read_address(C_read_address),.read_data_out(C_read_data));

    // Fold 128 output channels into 64 LUTRAM banks. Address bit 6 selects
    // channels 0..63 or 64..127; the lower six bits select the spatial word.
    wire[63:0]output_write_en_bus=
        RES_write_en?(64'b1<<RES_write_address[11:6]):64'b0;
    wire[1023:0]output_read_data_bus;
    generate
        for(output_bank=0;output_bank<64;output_bank=output_bank+1) begin:OUTPUT_MEMORIES
            memory_lutram#(.width(width),.depth_bits(7))output_RAM(
                             .clk(ACLK),.write_en(output_write_en_bus[output_bank]),
                             .write_address({RES_write_address[12],RES_write_address[5:0]}),
                             .write_data_in(RES_write_data),.read_en(output_read_en),
                             .read_address({write_counter[12],write_counter[5:0]}),
                             .read_data_out(output_read_data_bus[output_bank*16+:16]));
        end
    endgenerate
    assign output_read_data=output_read_data_bus[write_counter[11:6]*16+:16];

    conv5_relu_unrolled_ping_pong#(.IC_DIM(64),.IC_UNROLL(64),.OC_DIM(128),.OUT_UNROLL(1),.IN_H(8),.IN_W(8),.A_depth_bits(6),.B_depth_bits(11),.C_depth_bits(7),.RES_depth_bits(13))core(
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
                                     .C_read_data_out_oc0(C_read_data),
                                     .B_read_en(B_read_en),.B_read_address(B_read_address),.C_read_en(C_read_en),.C_read_address(C_read_address),.RES_write_en(RES_write_en),.RES_write_address(RES_write_address),.RES_write_data(RES_write_data));
endmodule

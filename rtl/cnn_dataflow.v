`timescale 1ns / 1ps

module cnn_dataflow #(
        parameter NUM_IMAGES = 100
    )(
        input  ACLK,
        input  ARESETN,
        output wire S_AXIS_TREADY,
        input  [15:0] S_AXIS_TDATA,
        input  S_AXIS_TLAST,
        input  S_AXIS_TVALID,
        output wire M_AXIS_TVALID,
        output wire [15:0] M_AXIS_TDATA,
        output wire M_AXIS_TLAST,
        input  M_AXIS_TREADY
    );
    localparam width = 16;
    localparam T_W1 = 864;
    localparam T_W2 = T_W1 + 9216;
    localparam T_W3 = T_W2 + 18432;
    localparam T_W4 = T_W3 + 36864;
    localparam T_W5 = T_W4 + 73728;
    localparam T_W6 = T_W5 + 147456;
    localparam T_WF = T_W6 + 1280;
    localparam T_B1 = T_WF + 32;
    localparam T_B2 = T_B1 + 32;
    localparam T_B3 = T_B2 + 64;
    localparam T_B4 = T_B3 + 64;
    localparam T_B5 = T_B4 + 128;
    localparam T_B6 = T_B5 + 128;
    localparam T_BF = T_B6 + 10;
    localparam TOTAL_WEIGHTS = T_BF;
    localparam IDLE = 3'b100, LOAD_WEIGHTS = 3'b010, RUN_DATAFLOW = 3'b001;
    reg [2:0] state, n_state;
    reg weights_loaded;
    reg [18:0] read_counter;
    wire rx_tready, tx_batch_done;
    assign S_AXIS_TREADY = (state == LOAD_WEIGHTS) ? 1'b1 : (state == RUN_DATAFLOW) ? rx_tready : 1'b0;

    (* max_fanout = "32" *) reg [15:0] wd_conv1;
    reg [11:0] W1_write_en, W1_wen_d;
    reg [6:0] W1_write_addr, W1_waddr_d;
    reg [3:0] B1_write_en, B1_wen_d;
    reg [2:0] B1_write_addr, B1_waddr_d;
    reg [3:0] W1_chunk_cnt;
    reg [3:0] W1_ram_sel;
    reg [6:0] W1_base_addr;
    wire [6:0] shared_w1_addr = W1_base_addr + W1_chunk_cnt;
    (* max_fanout = "32" *) reg [15:0] wd_conv2;
    reg [127:0] W2_write_en, W2_wen_d;
    reg [6:0] W2_write_addr, W2_waddr_d;
    reg [3:0] B2_write_en, B2_wen_d;
    reg [2:0] B2_write_addr, B2_waddr_d;
    reg [3:0] W2_chunk_cnt;
    reg [6:0] W2_ram_sel;
    reg [6:0] W2_base_addr;
    wire [6:0] shared_w2_addr = W2_base_addr + W2_chunk_cnt;
    (* max_fanout = "32" *) reg [15:0] wd_conv3;
    reg [63:0] W3_write_en, W3_wen_d;
    reg [8:0] W3_write_addr, W3_waddr_d;
    reg [1:0] B3_write_en, B3_wen_d;
    reg [4:0] B3_write_addr, B3_waddr_d;
    reg [3:0] W3_chunk_cnt;
    reg [5:0] W3_ram_sel;
    reg [8:0] W3_base_addr;
    wire [8:0] shared_w3_addr = W3_base_addr + W3_chunk_cnt;
    (* max_fanout = "32" *) reg [15:0] wd_conv4;
    reg [127:0] W4_write_en, W4_wen_d;
    reg [8:0] W4_write_addr, W4_waddr_d;
    reg [1:0] B4_write_en, B4_wen_d;
    reg [4:0] B4_write_addr, B4_waddr_d;
    reg [3:0] W4_chunk_cnt;
    reg [6:0] W4_ram_sel;
    reg [8:0] W4_base_addr;
    wire [8:0] shared_w4_addr = W4_base_addr + W4_chunk_cnt;
    (* max_fanout = "32" *) reg [15:0] wd_conv5;
    reg [63:0] W5_write_en, W5_wen_d;
    reg [10:0] W5_write_addr, W5_waddr_d;
    reg B5_write_en, B5_wen_d;
    reg [6:0] B5_write_addr, B5_waddr_d;
    reg [3:0] W5_chunk_cnt;
    reg [5:0] W5_ram_sel;
    reg [10:0] W5_base_addr;
    wire [10:0] shared_w5_addr = W5_base_addr + W5_chunk_cnt;
    (* max_fanout = "32" *) reg [15:0] wd_conv6;
    reg [127:0] W6_write_en, W6_wen_d;
    reg [10:0] W6_write_addr, W6_waddr_d;
    reg [1:0] B6_write_en, B6_wen_d;
    reg [5:0] B6_write_addr, B6_waddr_d;
    reg [3:0] W6_chunk_cnt;
    reg [6:0] W6_ic;
    reg W6_lane;
    reg [10:0] W6_base_addr;
    wire [6:0] W6_ram_sel = {W6_lane, W6_ic[5:0]};
    wire [10:0] shared_w6_addr = W6_base_addr + (W6_ic[6] ? 11'd9 : 11'd0) + W6_chunk_cnt;
    (* max_fanout = "32" *) reg [15:0] wd_fc;
    reg WF_write_en, WF_wen_d, BF_write_en, BF_wen_d;
    reg [10:0] WF_write_addr, WF_waddr_d;
    reg [3:0] BF_write_addr, BF_waddr_d;

    always @(posedge ACLK) begin
        if (!ARESETN) begin
            state <= IDLE;
            weights_loaded <= 0;
        end
        else begin
            state <= n_state;
            if (state == LOAD_WEIGHTS && S_AXIS_TVALID && read_counter == TOTAL_WEIGHTS - 1)
                weights_loaded <= 1;
        end
    end
    always @(posedge ACLK) begin
        if (!ARESETN) begin
            read_counter <= 0;
            W1_chunk_cnt <= 0;
            W1_ram_sel <= 0;
            W1_base_addr <= 0;
            W2_chunk_cnt <= 0;
            W2_ram_sel <= 0;
            W2_base_addr <= 0;
            W3_chunk_cnt <= 0;
            W3_ram_sel <= 0;
            W3_base_addr <= 0;
            W4_chunk_cnt <= 0;
            W4_ram_sel <= 0;
            W4_base_addr <= 0;
            W5_chunk_cnt <= 0;
            W5_ram_sel <= 0;
            W5_base_addr <= 0;
            W6_chunk_cnt <= 0;
            W6_ic <= 0;
            W6_lane <= 0;
            W6_base_addr <= 0;
        end
        else if (state == LOAD_WEIGHTS && S_AXIS_TVALID) begin
            read_counter <= (read_counter == TOTAL_WEIGHTS - 1) ? 0 : read_counter + 1;
            if (read_counter >= 0 && read_counter<T_W1) begin
                if (W1_chunk_cnt == 8) begin
                    W1_chunk_cnt <= 0;
                    if (W1_ram_sel == 11) begin
                        W1_ram_sel <= 0;
                        W1_base_addr <= W1_base_addr + 9;
                    end
                    else
                        W1_ram_sel <= W1_ram_sel + 1;
                end
                else
                    W1_chunk_cnt <= W1_chunk_cnt + 1;
            end
            if (read_counter >= T_W1 && read_counter<T_W2) begin
                if (W2_chunk_cnt == 8) begin
                    W2_chunk_cnt <= 0;
                    if (W2_ram_sel == 127) begin
                        W2_ram_sel <= 0;
                        W2_base_addr <= W2_base_addr + 9;
                    end
                    else
                        W2_ram_sel <= W2_ram_sel + 1;
                end
                else
                    W2_chunk_cnt <= W2_chunk_cnt + 1;
            end
            if (read_counter >= T_W2 && read_counter<T_W3) begin
                if (W3_chunk_cnt == 8) begin
                    W3_chunk_cnt <= 0;
                    if (W3_ram_sel == 63) begin
                        W3_ram_sel <= 0;
                        W3_base_addr <= W3_base_addr + 9;
                    end
                    else
                        W3_ram_sel <= W3_ram_sel + 1;
                end
                else
                    W3_chunk_cnt <= W3_chunk_cnt + 1;
            end
            if (read_counter >= T_W3 && read_counter<T_W4) begin
                if (W4_chunk_cnt == 8) begin
                    W4_chunk_cnt <= 0;
                    if (W4_ram_sel == 127) begin
                        W4_ram_sel <= 0;
                        W4_base_addr <= W4_base_addr + 9;
                    end
                    else
                        W4_ram_sel <= W4_ram_sel + 1;
                end
                else
                    W4_chunk_cnt <= W4_chunk_cnt + 1;
            end
            if (read_counter >= T_W4 && read_counter<T_W5) begin
                if (W5_chunk_cnt == 8) begin
                    W5_chunk_cnt <= 0;
                    if (W5_ram_sel == 63) begin
                        W5_ram_sel <= 0;
                        W5_base_addr <= W5_base_addr + 9;
                    end
                    else
                        W5_ram_sel <= W5_ram_sel + 1;
                end
                else
                    W5_chunk_cnt <= W5_chunk_cnt + 1;
            end
            if (read_counter >= T_W5 && read_counter<T_W6) begin
                if (W6_chunk_cnt == 8) begin
                    W6_chunk_cnt <= 0;
                    if (W6_ic == 127) begin
                        W6_ic <= 0;
                        if (W6_lane == 1) begin
                            W6_lane <= 0;
                            W6_base_addr <= W6_base_addr + 18;
                        end
                        else
                            W6_lane <= W6_lane + 1;
                    end
                    else
                        W6_ic <= W6_ic + 1;
                end
                else
                    W6_chunk_cnt <= W6_chunk_cnt + 1;
            end
            if (read_counter == TOTAL_WEIGHTS - 1) begin
                W1_chunk_cnt <= 0;
                W1_ram_sel <= 0;
                W1_base_addr <= 0;
                W2_chunk_cnt <= 0;
                W2_ram_sel <= 0;
                W2_base_addr <= 0;
                W3_chunk_cnt <= 0;
                W3_ram_sel <= 0;
                W3_base_addr <= 0;
                W4_chunk_cnt <= 0;
                W4_ram_sel <= 0;
                W4_base_addr <= 0;
                W5_chunk_cnt <= 0;
                W5_ram_sel <= 0;
                W5_base_addr <= 0;
                W6_chunk_cnt <= 0;
                W6_ic <= 0;
                W6_lane <= 0;
                W6_base_addr <= 0;
            end
        end
    end

    always @(*) begin
        n_state = state;
        W1_write_en = 0;
        W1_write_addr = 0;
        B1_write_en = 0;
        B1_write_addr = 0;
        W2_write_en = 0;
        W2_write_addr = 0;
        B2_write_en = 0;
        B2_write_addr = 0;
        W3_write_en = 0;
        W3_write_addr = 0;
        B3_write_en = 0;
        B3_write_addr = 0;
        W4_write_en = 0;
        W4_write_addr = 0;
        B4_write_en = 0;
        B4_write_addr = 0;
        W5_write_en = 0;
        W5_write_addr = 0;
        B5_write_en = 0;
        B5_write_addr = 0;
        W6_write_en = 0;
        W6_write_addr = 0;
        B6_write_en = 0;
        B6_write_addr = 0;
        WF_write_en = 0;
        WF_write_addr = 0;
        BF_write_en = 0;
        BF_write_addr = 0;
        case (state)
            IDLE :
                if (S_AXIS_TVALID)
                    n_state = weights_loaded ? RUN_DATAFLOW : LOAD_WEIGHTS;
            LOAD_WEIGHTS :
                if (S_AXIS_TVALID) begin
                    if (read_counter == TOTAL_WEIGHTS - 1)
                        n_state = IDLE;
                    if (read_counter<T_W1) begin
                        W1_write_en[W1_ram_sel] = 1;
                        W1_write_addr = shared_w1_addr;
                    end
                    else if (read_counter<T_W2) begin
                        W2_write_en[W2_ram_sel] = 1;
                        W2_write_addr = shared_w2_addr;
                    end
                    else if (read_counter<T_W3) begin
                        W3_write_en[W3_ram_sel] = 1;
                        W3_write_addr = shared_w3_addr;
                    end
                    else if (read_counter<T_W4) begin
                        W4_write_en[W4_ram_sel] = 1;
                        W4_write_addr = shared_w4_addr;
                    end
                    else if (read_counter<T_W5) begin
                        W5_write_en[W5_ram_sel] = 1;
                        W5_write_addr = shared_w5_addr;
                    end
                    else if (read_counter<T_W6) begin
                        W6_write_en[W6_ram_sel] = 1;
                        W6_write_addr = shared_w6_addr;
                    end
                    else if (read_counter<T_WF) begin
                        WF_write_en = 1;
                        WF_write_addr = read_counter - T_W6;
                    end
                    else if (read_counter<T_B1) begin
                        B1_write_en[(read_counter - T_WF) % 4] = 1;
                        B1_write_addr = (read_counter - T_WF)/4;
                    end
                    else if (read_counter<T_B2) begin
                        B2_write_en[(read_counter - T_B1) % 4] = 1;
                        B2_write_addr = (read_counter - T_B1)/4;
                    end
                    else if (read_counter<T_B3) begin
                        B3_write_en[(read_counter - T_B2) % 2] = 1;
                        B3_write_addr = (read_counter - T_B2)/2;
                    end
                    else if (read_counter<T_B4) begin
                        B4_write_en[(read_counter - T_B3) % 2] = 1;
                        B4_write_addr = (read_counter - T_B3)/2;
                    end
                    else if (read_counter<T_B5) begin
                        B5_write_en = 1;
                        B5_write_addr = read_counter - T_B4;
                    end
                    else if (read_counter<T_B6) begin
                        B6_write_en[(read_counter - T_B5) % 2] = 1;
                        B6_write_addr = (read_counter - T_B5)/2;
                    end
                    else if (read_counter<T_BF) begin
                        BF_write_en = 1;
                        BF_write_addr = read_counter - T_B6;
                    end
                end
            RUN_DATAFLOW :
                if (tx_batch_done)
                    n_state = IDLE;
        endcase
    end

    always @(posedge ACLK) begin
        if (!ARESETN) begin
            wd_conv1 <= 0;
            W1_wen_d <= 0;
            W1_waddr_d <= 0;
            B1_wen_d <= 0;
            B1_waddr_d <= 0;
            wd_conv2 <= 0;
            W2_wen_d <= 0;
            W2_waddr_d <= 0;
            B2_wen_d <= 0;
            B2_waddr_d <= 0;
            wd_conv3 <= 0;
            W3_wen_d <= 0;
            W3_waddr_d <= 0;
            B3_wen_d <= 0;
            B3_waddr_d <= 0;
            wd_conv4 <= 0;
            W4_wen_d <= 0;
            W4_waddr_d <= 0;
            B4_wen_d <= 0;
            B4_waddr_d <= 0;
            wd_conv5 <= 0;
            W5_wen_d <= 0;
            W5_waddr_d <= 0;
            B5_wen_d <= 0;
            B5_waddr_d <= 0;
            wd_conv6 <= 0;
            W6_wen_d <= 0;
            W6_waddr_d <= 0;
            B6_wen_d <= 0;
            B6_waddr_d <= 0;
            wd_fc <= 0;
            WF_wen_d <= 0;
            WF_waddr_d <= 0;
            BF_wen_d <= 0;
            BF_waddr_d <= 0;
        end
        else begin
            wd_conv1 <= S_AXIS_TDATA;
            W1_wen_d <= W1_write_en;
            W1_waddr_d <= W1_write_addr;
            B1_wen_d <= B1_write_en;
            B1_waddr_d <= B1_write_addr;
            wd_conv2 <= S_AXIS_TDATA;
            W2_wen_d <= W2_write_en;
            W2_waddr_d <= W2_write_addr;
            B2_wen_d <= B2_write_en;
            B2_waddr_d <= B2_write_addr;
            wd_conv3 <= S_AXIS_TDATA;
            W3_wen_d <= W3_write_en;
            W3_waddr_d <= W3_write_addr;
            B3_wen_d <= B3_write_en;
            B3_waddr_d <= B3_write_addr;
            wd_conv4 <= S_AXIS_TDATA;
            W4_wen_d <= W4_write_en;
            W4_waddr_d <= W4_write_addr;
            B4_wen_d <= B4_write_en;
            B4_waddr_d <= B4_write_addr;
            wd_conv5 <= S_AXIS_TDATA;
            W5_wen_d <= W5_write_en;
            W5_waddr_d <= W5_write_addr;
            B5_wen_d <= B5_write_en;
            B5_waddr_d <= B5_write_addr;
            wd_conv6 <= S_AXIS_TDATA;
            W6_wen_d <= W6_write_en;
            W6_waddr_d <= W6_write_addr;
            B6_wen_d <= B6_write_en;
            B6_waddr_d <= B6_write_addr;
            wd_fc <= S_AXIS_TDATA;
            WF_wen_d <= WF_write_en;
            WF_waddr_d <= WF_write_addr;
            BF_wen_d <= BF_write_en;
            BF_waddr_d <= BF_write_addr;
        end
    end

    // Conv1: 12 shallow weight banks grouped four per asynchronous distributed-
    // RAM module, avoiding three under-filled RAMB36s.  The Conv1 core delays
    // the LUTRAM address to align these weights with its input BRAM pipeline.
    wire conv1_B_read_en, conv1_C_read_en;
    wire [6:0] conv1_B_addr;
    wire [2:0] conv1_C_addr;
    wire [15:0] conv1_B_data[0:11];
    wire [15:0] conv1_C_data[0:3];
    wire [63:0] conv1_B_packed_data[0:2];
    genvar gw1, gb1;
    generate
        for (gw1 = 0; gw1<3; gw1 = gw1 + 1) begin : CONV1_WEIGHT_MEMORIES
            wire [3:0] packed_write_en = W1_wen_d[gw1*4 +: 4];
            memory_lutram_packed4 #(.width (width), .depth_bits (7))weight_ram(
                                      .clk (ACLK),
                                      .write_en (packed_write_en),
                                      .write_address (W1_waddr_d),
                                      .write_data_in (wd_conv1),
                                      .read_en (conv1_B_read_en),
                                      .read_address (conv1_B_addr),
                                      .read_data_out (conv1_B_packed_data[gw1]));
            assign conv1_B_data[gw1*4]   = conv1_B_packed_data[gw1][15:0];
            assign conv1_B_data[gw1*4+1] = conv1_B_packed_data[gw1][31:16];
            assign conv1_B_data[gw1*4+2] = conv1_B_packed_data[gw1][47:32];
            assign conv1_B_data[gw1*4+3] = conv1_B_packed_data[gw1][63:48];
        end
        for (gb1 = 0; gb1<4; gb1 = gb1 + 1) begin : CONV1_BIAS_MEMORIES
            memory_lutram #(.width (width), .depth_bits (3))bias_ram(
                              .clk (ACLK), .write_en (B1_wen_d[gb1]), .write_address (B1_waddr_d),
                              .write_data_in (wd_conv1), .read_en (conv1_C_read_en),
                              .read_address (conv1_C_addr), .read_data_out (conv1_C_data[gb1]));
        end
    endgenerate

    // Conv2 : 128 logical weight banks packed four per TDP BRAM36,
    // plus four LUTRAM bias banks.
    wire conv2_B_read_en, conv2_C_read_en;
    wire [6:0] conv2_B_addr;
    wire [2:0] conv2_C_addr;
    wire [15:0] conv2_B_data[0:127];
    wire [15:0] conv2_C_data[0:3];
    wire [63:0] conv2_B_packed_data[0:31];
    genvar gw2, gb2;
    generate
        for (gw2 = 0; gw2<32; gw2 = gw2 + 1) begin : CONV2_WEIGHT_MEMORIES
            wire [3:0] packed_write_en = {
                     W2_wen_d[96+gw2], W2_wen_d[64+gw2],
                     W2_wen_d[32+gw2], W2_wen_d[gw2]
                 };
            memory_bram_tdp_packed4 #(.width (width), .depth_bits (7))weight_ram(
                                        .clk (ACLK), .write_en (packed_write_en),
                                        .write_address (W2_waddr_d), .write_data_in (wd_conv2),
                                        .read_en (conv2_B_read_en), .read_address (conv2_B_addr),
                                        .read_data_out (conv2_B_packed_data[gw2]));
            assign conv2_B_data[gw2]    = conv2_B_packed_data[gw2][15:0];
            assign conv2_B_data[32+gw2] = conv2_B_packed_data[gw2][31:16];
            assign conv2_B_data[64+gw2] = conv2_B_packed_data[gw2][47:32];
            assign conv2_B_data[96+gw2] = conv2_B_packed_data[gw2][63:48];
        end
        for (gb2 = 0; gb2<4; gb2 = gb2 + 1) begin : CONV2_BIAS_MEMORIES
            memory_lutram #(.width (width), .depth_bits (3))bias_ram(
                              .clk (ACLK), .write_en (B2_wen_d[gb2]), .write_address (B2_waddr_d),
                              .write_data_in (wd_conv2), .read_en (conv2_C_read_en),
                              .read_address (conv2_C_addr), .read_data_out (conv2_C_data[gb2]));
        end
    endgenerate

    // Conv3 OU2: 64 logical 512x16 weight banks packed four per TDP BRAM36,
    // plus two 32-deep LUTRAM bias banks.
    wire conv3_B_read_en, conv3_C_read_en;
    wire [8:0] conv3_B_addr;
    wire [4:0] conv3_C_addr;
    wire [15:0] conv3_B_data[0:63];
    wire [15:0] conv3_C_data[0:1];
    wire [63:0] conv3_B_packed_data[0:15];
    genvar gw3, gb3;
    generate
        for (gw3 = 0; gw3<16; gw3 = gw3 + 1) begin : CONV3_WEIGHT_MEMORIES
            wire [3:0] packed_write_en = {
                     W3_wen_d[48+gw3], W3_wen_d[32+gw3],
                     W3_wen_d[16+gw3], W3_wen_d[gw3]
                 };
            memory_bram_tdp_packed4 #(.width (width), .depth_bits (9))weight_ram(
                                        .clk (ACLK),
                                        .write_en (packed_write_en),
                                        .write_address (W3_waddr_d),
                                        .write_data_in (wd_conv3),
                                        .read_en (conv3_B_read_en),
                                        .read_address (conv3_B_addr),
                                        .read_data_out (conv3_B_packed_data[gw3]));
            assign conv3_B_data[gw3]    = conv3_B_packed_data[gw3][15:0];
            assign conv3_B_data[16+gw3] = conv3_B_packed_data[gw3][31:16];
            assign conv3_B_data[32+gw3] = conv3_B_packed_data[gw3][47:32];
            assign conv3_B_data[48+gw3] = conv3_B_packed_data[gw3][63:48];
        end
        for (gb3 = 0; gb3<2; gb3 = gb3 + 1) begin : CONV3_BIAS_MEMORIES
            memory_lutram #(.width (width), .depth_bits (5))bias_ram(
                              .clk (ACLK), .write_en (B3_wen_d[gb3]), .write_address (B3_waddr_d),
                              .write_data_in (wd_conv3), .read_en (conv3_C_read_en),
                              .read_address (conv3_C_addr), .read_data_out (conv3_C_data[gb3]));
        end
    endgenerate

    // Conv4 OU2: 128 logical 512x16 weight banks packed four per TDP BRAM36,
    // plus two 32-deep LUTRAM bias banks.
    wire conv4_B_read_en, conv4_C_read_en;
    wire [8:0] conv4_B_addr;
    wire [4:0] conv4_C_addr;
    wire [15:0] conv4_B_data[0:127];
    wire [15:0] conv4_C_data[0:1];
    wire [63:0] conv4_B_packed_data[0:31];
    genvar gw4, gb4;
    generate
        for (gw4 = 0; gw4<32; gw4 = gw4 + 1) begin : CONV4_WEIGHT_MEMORIES
            wire [3:0] packed_write_en = {
                     W4_wen_d[96+gw4], W4_wen_d[64+gw4],
                     W4_wen_d[32+gw4], W4_wen_d[gw4]
                 };
            memory_bram_tdp_packed4 #(.width (width), .depth_bits (9))weight_ram(
                                        .clk (ACLK),
                                        .write_en (packed_write_en),
                                        .write_address (W4_waddr_d),
                                        .write_data_in (wd_conv4),
                                        .read_en (conv4_B_read_en),
                                        .read_address (conv4_B_addr),
                                        .read_data_out (conv4_B_packed_data[gw4]));
            assign conv4_B_data[gw4]     = conv4_B_packed_data[gw4][15:0];
            assign conv4_B_data[32+gw4]  = conv4_B_packed_data[gw4][31:16];
            assign conv4_B_data[64+gw4]  = conv4_B_packed_data[gw4][47:32];
            assign conv4_B_data[96+gw4]  = conv4_B_packed_data[gw4][63:48];
        end
        for (gb4 = 0; gb4<2; gb4 = gb4 + 1) begin : CONV4_BIAS_MEMORIES
            memory_lutram #(.width (width), .depth_bits (5))bias_ram(
                              .clk (ACLK), .write_en (B4_wen_d[gb4]), .write_address (B4_waddr_d),
                              .write_data_in (wd_conv4), .read_en (conv4_C_read_en),
                              .read_address (conv4_C_addr), .read_data_out (conv4_C_data[gb4]));
        end
    endgenerate

    // Conv5 OU1: 64 logical 2048x16 weight banks packed eight per TDP
    // UltraRAM (eight URAM total), plus one 128x16 LUTRAM bias bank.
    wire conv5_B_read_en, conv5_C_read_en;
    wire [10:0] conv5_B_addr;
    wire [6:0] conv5_C_addr;
    wire [15:0] conv5_B_data[0:63];
    wire [15:0] conv5_C_data;
    wire [127:0] conv5_B_uram_packed_data[0:7];
    genvar guw5, glw5;
    generate
        for (guw5 = 0; guw5<8; guw5 = guw5 + 1) begin : CONV5_WEIGHT_URAMS
            wire [7:0] packed_write_en = W5_wen_d[guw5*8 +: 8];
            memory_uram_tdp_packed8 #(.width (width), .depth_bits (11))weight_ram(
                                        .clk (ACLK), .write_en (packed_write_en),
                                        .write_address (W5_waddr_d), .write_data_in (wd_conv5),
                                        .read_en (conv5_B_read_en), .read_address (conv5_B_addr),
                                        .read_data_out (conv5_B_uram_packed_data[guw5]));
            for (glw5 = 0; glw5<8; glw5 = glw5 + 1) begin : UNPACK
                assign conv5_B_data[guw5*8+glw5] =
                       conv5_B_uram_packed_data[guw5][glw5*16 +: 16];
            end
        end
    endgenerate
    memory_lutram #(.width (width), .depth_bits (7)) CONV5_BIAS_MEMORY(
                      .clk (ACLK), .write_en (B5_wen_d), .write_address (B5_waddr_d),
                      .write_data_in (wd_conv5), .read_en (conv5_C_read_en),
                      .read_address (conv5_C_addr), .read_data_out (conv5_C_data));

    // Conv6: 128 logical 2,048x16 weight banks packed eight per TDP
    // UltraRAM (16 URAM total), plus two 64x16 LUTRAM bias banks.
    wire conv6_B_read_en, conv6_C_read_en;
    wire [10:0] conv6_B_addr;
    wire [5:0] conv6_C_addr;
    wire [15:0] conv6_B_data[0:127];
    wire [15:0] conv6_C_data[0:1];
    wire [127:0] conv6_B_uram_packed_data[0:15];
    genvar guw6, glw6, gb6;
    generate
        for (guw6 = 0; guw6<16; guw6 = guw6 + 1) begin : CONV6_WEIGHT_URAMS
            wire [7:0] packed_write_en = W6_wen_d[guw6*8 +: 8];
            memory_uram_tdp_packed8 #(.width (width), .depth_bits (11))weight_ram(
                                        .clk (ACLK), .write_en (packed_write_en),
                                        .write_address (W6_waddr_d), .write_data_in (wd_conv6),
                                        .read_en (conv6_B_read_en), .read_address (conv6_B_addr),
                                        .read_data_out (conv6_B_uram_packed_data[guw6]));
            for (glw6 = 0; glw6<8; glw6 = glw6 + 1) begin : UNPACK
                assign conv6_B_data[guw6*8+glw6] =
                       conv6_B_uram_packed_data[guw6][glw6*16 +: 16];
            end
        end
        for (gb6 = 0; gb6<2; gb6 = gb6 + 1) begin : CONV6_BIAS_MEMORIES
            memory_lutram #(.width (width), .depth_bits (6))bias_ram(
                              .clk (ACLK), .write_en (B6_wen_d[gb6]), .write_address (B6_waddr_d),
                              .write_data_in (wd_conv6), .read_en (conv6_C_read_en),
                              .read_address (conv6_C_addr), .read_data_out (conv6_C_data[gb6]));
        end
    endgenerate

    // FC weights use 1,280 of the 2,048 entries in one 16-bit BRAM36.
    // The ten biases remain in a small asynchronous LUTRAM.
    wire fc_B_read_en, fc_C_read_en;
    wire [10:0] fc_B_addr;
    wire [3:0] fc_C_addr;
    wire [15:0] fc_B_data, fc_C_data;
    memory_RAM #(
                   .width      (16),
                   .depth_bits (11)
               ) FC_WEIGHT_RAM (
                   .clk           (ACLK),
                   .write_en      (WF_wen_d),
                   .write_address (WF_waddr_d),
                   .write_data_in (wd_fc),
                   .read_en       (fc_B_read_en),
                   .read_address  (fc_B_addr),
                   .read_data_out (fc_B_data)
               );
    // Ten FC biases are small enough for distributed RAM. The FC core delays
    // this asynchronous read to align with its three-cycle BRAM operands.
    memory_lutram #(
                      .width      (16),
                      .depth_bits (4)
                  ) FC_BIAS_RAM (
                      .clk           (ACLK),
                      .write_en      (BF_wen_d),
                      .write_address (BF_waddr_d),
                      .write_data_in (wd_fc),
                      .read_en       (fc_C_read_en),
                      .read_address  (fc_C_addr),
                      .read_data_out (fc_C_data)
                  );

    // Receiver - > three channel input ping - pong buffers.
    wire rx_we0, rx_we1, rx_we2;
    wire [9:0] rx_waddr;
    wire [15:0] rx_wdata;
    wire rx_frame_done, rx_buffer_ready;
    wire rx_tvalid_gated = S_AXIS_TVALID && (state == RUN_DATAFLOW);
    axi_image_receiver rx (
                           .clk           (ACLK),
                           .rst_n         (ARESETN),
                           .s_axis_tready (rx_tready),
                           .s_axis_tdata  (S_AXIS_TDATA),
                           .s_axis_tvalid (rx_tvalid_gated),
                           .buffer_ready  (rx_buffer_ready),
                           .write_en_a0   (rx_we0),
                           .write_en_a1   (rx_we1),
                           .write_en_a2   (rx_we2),
                           .write_addr    (rx_waddr),
                           .write_data    (rx_wdata),
                           .frame_done    (rx_frame_done)
                       );
    wire conv1_done_reading;
    wire conv1_A_read_en;
    wire [9:0] conv1_A_addr;
    wire conv1_can_read;
    wire [15:0] conv1_A_data[0:2];
    wire [47:0] conv1_A_data_bus;

    // Three physical 2Kx16 TDP BRAMs: one for each RGB channel.  Address bit
    // 10 selects the ping/pong frame while bits 9:0 select the image pixel.
    ping_pong_buffer_array_bram_tdp #(
                                        .DATA_WIDTH   (16),
                                        .DEPTH_BITS   (10),
                                        .NUM_CHANNELS (3)
                                    ) rx_to_conv1 (
                                        .clk                     (ACLK),
                                        .rst                     (~ARESETN),
                                        .producer_write_en_bus   ({rx_we2, rx_we1, rx_we0}),
                                        .producer_write_addr     (rx_waddr),
                                        .producer_write_data     (rx_wdata),
                                        .producer_frame_done     (rx_frame_done),
                                        .producer_ready          (rx_buffer_ready),
                                        .consumer_read_en        (conv1_A_read_en),
                                        .consumer_read_addr      (conv1_A_addr),
                                        .consumer_read_data_bus  (conv1_A_data_bus),
                                        .consumer_frame_done     (conv1_done_reading),
                                        .consumer_ready          (conv1_can_read)
                                    );

    assign conv1_A_data[0] = conv1_A_data_bus[15:0];
    assign conv1_A_data[1] = conv1_A_data_bus[31:16];
    assign conv1_A_data[2] = conv1_A_data_bus[47:32];

    wire conv1_can_write, conv1_done_writing, conv1_out_we0, conv1_out_we1;
    wire [14:0] conv1_out_addr0, conv1_out_addr1;
    wire [15:0] conv1_out_data0, conv1_out_data1;
    conv1_relu_unrolled_ping_pong #(.OC_DIM (32),
                                    .OUT_UNROLL (4), .IN_H (32), .IN_W (32),
                                    .A_depth_bits (10), .B_depth_bits (7),
                                    .C_depth_bits (3), .RES_depth_bits (15))conv1_inst(
                                      .clk (ACLK),
                                      .rst (ARESETN),
                                      .consumer_ready (conv1_can_read),
                                      .producer_ready (conv1_can_write),
                                      .consumer_frame_done (conv1_done_reading),
                                      .producer_frame_done (conv1_done_writing),
                                      .A_read_en (conv1_A_read_en),
                                      .A_read_address (conv1_A_addr),
                                      .A_read_data_out_0 (conv1_A_data[0]),
                                      .A_read_data_out_1 (conv1_A_data[1]),
                                      .A_read_data_out_2 (conv1_A_data[2]),
                                      .B_read_en (conv1_B_read_en),
                                      .B_read_address (conv1_B_addr),
                                      .B_read_data_out_0_oc0 (conv1_B_data[0]),
                                      .B_read_data_out_1_oc0 (conv1_B_data[1]),
                                      .B_read_data_out_2_oc0 (conv1_B_data[2]),
                                      .B_read_data_out_0_oc1 (conv1_B_data[3]),
                                      .B_read_data_out_1_oc1 (conv1_B_data[4]),
                                      .B_read_data_out_2_oc1 (conv1_B_data[5]),
                                      .B_read_data_out_0_oc2 (conv1_B_data[6]),
                                      .B_read_data_out_1_oc2 (conv1_B_data[7]),
                                      .B_read_data_out_2_oc2 (conv1_B_data[8]),
                                      .B_read_data_out_0_oc3 (conv1_B_data[9]),
                                      .B_read_data_out_1_oc3 (conv1_B_data[10]),
                                      .B_read_data_out_2_oc3 (conv1_B_data[11]),
                                      .C_read_en (conv1_C_read_en),
                                      .C_read_address (conv1_C_addr),
                                      .C_read_data_out_oc0 (conv1_C_data[0]),
                                      .C_read_data_out_oc1 (conv1_C_data[1]),
                                      .C_read_data_out_oc2 (conv1_C_data[2]),
                                      .C_read_data_out_oc3 (conv1_C_data[3]),
                                      .RES_write_en_even (conv1_out_we0),
                                      .RES_write_address_even (conv1_out_addr0),
                                      .RES_write_data_in_even (conv1_out_data0),
                                      .RES_write_en_odd (conv1_out_we1),
                                      .RES_write_address_odd (conv1_out_addr1),
                                      .RES_write_data_in_odd (conv1_out_data1));

    wire conv2_can_read, conv2_done_reading, conv2_A_read_en;
    wire [9:0] conv2_A_addr;
    wire [511:0] conv2_A_data_bus;
    // Four UltraRAMs per frame, with eight channels packed into each device.
    // Separate ping/pong devices preserve concurrent Conv1 writes and Conv2
    // reads even though a channel-parallel read consumes both URAM ports.
    ping_pong_buffer_array_uram_packed8_2w #(
                                               .DATA_WIDTH   (16),
                                               .DEPTH_BITS   (10),
                                               .NUM_CHANNELS (32)
                                           ) conv1_to_conv2 (
                                               .clk                      (ACLK),
                                               .rst                      (~ARESETN),
                                               .producer_write_en_0      (conv1_out_we0),
                                               .producer_write_channel_0 (conv1_out_addr0[14:10]),
                                               .producer_write_addr_0    (conv1_out_addr0[9:0]),
                                               .producer_write_data_0    (conv1_out_data0),
                                               .producer_write_en_1      (conv1_out_we1),
                                               .producer_write_channel_1 (conv1_out_addr1[14:10]),
                                               .producer_write_addr_1    (conv1_out_addr1[9:0]),
                                               .producer_write_data_1    (conv1_out_data1),
                                               .producer_frame_done      (conv1_done_writing),
                                               .producer_ready           (conv1_can_write),
                                               .consumer_read_en         (conv2_A_read_en),
                                               .consumer_read_addr       (conv2_A_addr),
                                               .consumer_read_data_bus   (conv2_A_data_bus),
                                               .consumer_frame_done      (conv2_done_reading),
                                               .consumer_ready           (conv2_can_read)
                                           );
    wire [15:0] conv2_A_data[0:31];
    genvar g_conv2_A_data;
generate for (g_conv2_A_data = 0; g_conv2_A_data<32; g_conv2_A_data = g_conv2_A_data + 1) begin : CONV2_A_DATA_UNPACK
            assign conv2_A_data[g_conv2_A_data] = conv2_A_data_bus[g_conv2_A_data*16 +:16];
        end
    endgenerate

    wire conv2_can_write, conv2_done_writing, conv2_out_we0, conv2_out_we1;
    wire [14:0] conv2_out_addr0, conv2_out_addr1;
    wire [15:0] conv2_out_data0, conv2_out_data1;
    conv2_relu_unrolled_ping_pong #(.IC_DIM (32), .OC_DIM (32),
                                    .OUT_UNROLL (4), .IN_H (32), .IN_W (32),
                                    .A_depth_bits (10), .B_depth_bits (7),
                                    .C_depth_bits (3), .RES_depth_bits (15))conv2_inst(
                                      .clk (ACLK),
                                      .rst (ARESETN),
                                      .consumer_ready (conv2_can_read),
                                      .producer_ready (conv2_can_write),
                                      .consumer_frame_done (conv2_done_reading),
                                      .producer_frame_done (conv2_done_writing),
                                      .A_read_en (conv2_A_read_en),
                                      .A_read_address (conv2_A_addr),
                                      .A_read_data_out_0 (conv2_A_data[0]),
                                      .A_read_data_out_1 (conv2_A_data[1]),
                                      .A_read_data_out_2 (conv2_A_data[2]),
                                      .A_read_data_out_3 (conv2_A_data[3]),
                                      .A_read_data_out_4 (conv2_A_data[4]),
                                      .A_read_data_out_5 (conv2_A_data[5]),
                                      .A_read_data_out_6 (conv2_A_data[6]),
                                      .A_read_data_out_7 (conv2_A_data[7]),
                                      .A_read_data_out_8 (conv2_A_data[8]),
                                      .A_read_data_out_9 (conv2_A_data[9]),
                                      .A_read_data_out_10 (conv2_A_data[10]),
                                      .A_read_data_out_11 (conv2_A_data[11]),
                                      .A_read_data_out_12 (conv2_A_data[12]),
                                      .A_read_data_out_13 (conv2_A_data[13]),
                                      .A_read_data_out_14 (conv2_A_data[14]),
                                      .A_read_data_out_15 (conv2_A_data[15]),
                                      .A_read_data_out_16 (conv2_A_data[16]),
                                      .A_read_data_out_17 (conv2_A_data[17]),
                                      .A_read_data_out_18 (conv2_A_data[18]),
                                      .A_read_data_out_19 (conv2_A_data[19]),
                                      .A_read_data_out_20 (conv2_A_data[20]),
                                      .A_read_data_out_21 (conv2_A_data[21]),
                                      .A_read_data_out_22 (conv2_A_data[22]),
                                      .A_read_data_out_23 (conv2_A_data[23]),
                                      .A_read_data_out_24 (conv2_A_data[24]),
                                      .A_read_data_out_25 (conv2_A_data[25]),
                                      .A_read_data_out_26 (conv2_A_data[26]),
                                      .A_read_data_out_27 (conv2_A_data[27]),
                                      .A_read_data_out_28 (conv2_A_data[28]),
                                      .A_read_data_out_29 (conv2_A_data[29]),
                                      .A_read_data_out_30 (conv2_A_data[30]),
                                      .A_read_data_out_31 (conv2_A_data[31]),
                                      .B_read_en (conv2_B_read_en),
                                      .B_read_address (conv2_B_addr),
                                      .B_read_data_out_0_oc0 (conv2_B_data[0]),
                                      .B_read_data_out_1_oc0 (conv2_B_data[1]),
                                      .B_read_data_out_2_oc0 (conv2_B_data[2]),
                                      .B_read_data_out_3_oc0 (conv2_B_data[3]),
                                      .B_read_data_out_4_oc0 (conv2_B_data[4]),
                                      .B_read_data_out_5_oc0 (conv2_B_data[5]),
                                      .B_read_data_out_6_oc0 (conv2_B_data[6]),
                                      .B_read_data_out_7_oc0 (conv2_B_data[7]),
                                      .B_read_data_out_8_oc0 (conv2_B_data[8]),
                                      .B_read_data_out_9_oc0 (conv2_B_data[9]),
                                      .B_read_data_out_10_oc0 (conv2_B_data[10]),
                                      .B_read_data_out_11_oc0 (conv2_B_data[11]),
                                      .B_read_data_out_12_oc0 (conv2_B_data[12]),
                                      .B_read_data_out_13_oc0 (conv2_B_data[13]),
                                      .B_read_data_out_14_oc0 (conv2_B_data[14]),
                                      .B_read_data_out_15_oc0 (conv2_B_data[15]),
                                      .B_read_data_out_16_oc0 (conv2_B_data[16]),
                                      .B_read_data_out_17_oc0 (conv2_B_data[17]),
                                      .B_read_data_out_18_oc0 (conv2_B_data[18]),
                                      .B_read_data_out_19_oc0 (conv2_B_data[19]),
                                      .B_read_data_out_20_oc0 (conv2_B_data[20]),
                                      .B_read_data_out_21_oc0 (conv2_B_data[21]),
                                      .B_read_data_out_22_oc0 (conv2_B_data[22]),
                                      .B_read_data_out_23_oc0 (conv2_B_data[23]),
                                      .B_read_data_out_24_oc0 (conv2_B_data[24]),
                                      .B_read_data_out_25_oc0 (conv2_B_data[25]),
                                      .B_read_data_out_26_oc0 (conv2_B_data[26]),
                                      .B_read_data_out_27_oc0 (conv2_B_data[27]),
                                      .B_read_data_out_28_oc0 (conv2_B_data[28]),
                                      .B_read_data_out_29_oc0 (conv2_B_data[29]),
                                      .B_read_data_out_30_oc0 (conv2_B_data[30]),
                                      .B_read_data_out_31_oc0 (conv2_B_data[31]),
                                      .B_read_data_out_0_oc1 (conv2_B_data[32]),
                                      .B_read_data_out_1_oc1 (conv2_B_data[33]),
                                      .B_read_data_out_2_oc1 (conv2_B_data[34]),
                                      .B_read_data_out_3_oc1 (conv2_B_data[35]),
                                      .B_read_data_out_4_oc1 (conv2_B_data[36]),
                                      .B_read_data_out_5_oc1 (conv2_B_data[37]),
                                      .B_read_data_out_6_oc1 (conv2_B_data[38]),
                                      .B_read_data_out_7_oc1 (conv2_B_data[39]),
                                      .B_read_data_out_8_oc1 (conv2_B_data[40]),
                                      .B_read_data_out_9_oc1 (conv2_B_data[41]),
                                      .B_read_data_out_10_oc1 (conv2_B_data[42]),
                                      .B_read_data_out_11_oc1 (conv2_B_data[43]),
                                      .B_read_data_out_12_oc1 (conv2_B_data[44]),
                                      .B_read_data_out_13_oc1 (conv2_B_data[45]),
                                      .B_read_data_out_14_oc1 (conv2_B_data[46]),
                                      .B_read_data_out_15_oc1 (conv2_B_data[47]),
                                      .B_read_data_out_16_oc1 (conv2_B_data[48]),
                                      .B_read_data_out_17_oc1 (conv2_B_data[49]),
                                      .B_read_data_out_18_oc1 (conv2_B_data[50]),
                                      .B_read_data_out_19_oc1 (conv2_B_data[51]),
                                      .B_read_data_out_20_oc1 (conv2_B_data[52]),
                                      .B_read_data_out_21_oc1 (conv2_B_data[53]),
                                      .B_read_data_out_22_oc1 (conv2_B_data[54]),
                                      .B_read_data_out_23_oc1 (conv2_B_data[55]),
                                      .B_read_data_out_24_oc1 (conv2_B_data[56]),
                                      .B_read_data_out_25_oc1 (conv2_B_data[57]),
                                      .B_read_data_out_26_oc1 (conv2_B_data[58]),
                                      .B_read_data_out_27_oc1 (conv2_B_data[59]),
                                      .B_read_data_out_28_oc1 (conv2_B_data[60]),
                                      .B_read_data_out_29_oc1 (conv2_B_data[61]),
                                      .B_read_data_out_30_oc1 (conv2_B_data[62]),
                                      .B_read_data_out_31_oc1 (conv2_B_data[63]),
                                      .B_read_data_out_0_oc2 (conv2_B_data[64]),
                                      .B_read_data_out_1_oc2 (conv2_B_data[65]),
                                      .B_read_data_out_2_oc2 (conv2_B_data[66]),
                                      .B_read_data_out_3_oc2 (conv2_B_data[67]),
                                      .B_read_data_out_4_oc2 (conv2_B_data[68]),
                                      .B_read_data_out_5_oc2 (conv2_B_data[69]),
                                      .B_read_data_out_6_oc2 (conv2_B_data[70]),
                                      .B_read_data_out_7_oc2 (conv2_B_data[71]),
                                      .B_read_data_out_8_oc2 (conv2_B_data[72]),
                                      .B_read_data_out_9_oc2 (conv2_B_data[73]),
                                      .B_read_data_out_10_oc2 (conv2_B_data[74]),
                                      .B_read_data_out_11_oc2 (conv2_B_data[75]),
                                      .B_read_data_out_12_oc2 (conv2_B_data[76]),
                                      .B_read_data_out_13_oc2 (conv2_B_data[77]),
                                      .B_read_data_out_14_oc2 (conv2_B_data[78]),
                                      .B_read_data_out_15_oc2 (conv2_B_data[79]),
                                      .B_read_data_out_16_oc2 (conv2_B_data[80]),
                                      .B_read_data_out_17_oc2 (conv2_B_data[81]),
                                      .B_read_data_out_18_oc2 (conv2_B_data[82]),
                                      .B_read_data_out_19_oc2 (conv2_B_data[83]),
                                      .B_read_data_out_20_oc2 (conv2_B_data[84]),
                                      .B_read_data_out_21_oc2 (conv2_B_data[85]),
                                      .B_read_data_out_22_oc2 (conv2_B_data[86]),
                                      .B_read_data_out_23_oc2 (conv2_B_data[87]),
                                      .B_read_data_out_24_oc2 (conv2_B_data[88]),
                                      .B_read_data_out_25_oc2 (conv2_B_data[89]),
                                      .B_read_data_out_26_oc2 (conv2_B_data[90]),
                                      .B_read_data_out_27_oc2 (conv2_B_data[91]),
                                      .B_read_data_out_28_oc2 (conv2_B_data[92]),
                                      .B_read_data_out_29_oc2 (conv2_B_data[93]),
                                      .B_read_data_out_30_oc2 (conv2_B_data[94]),
                                      .B_read_data_out_31_oc2 (conv2_B_data[95]),
                                      .B_read_data_out_0_oc3 (conv2_B_data[96]),
                                      .B_read_data_out_1_oc3 (conv2_B_data[97]),
                                      .B_read_data_out_2_oc3 (conv2_B_data[98]),
                                      .B_read_data_out_3_oc3 (conv2_B_data[99]),
                                      .B_read_data_out_4_oc3 (conv2_B_data[100]),
                                      .B_read_data_out_5_oc3 (conv2_B_data[101]),
                                      .B_read_data_out_6_oc3 (conv2_B_data[102]),
                                      .B_read_data_out_7_oc3 (conv2_B_data[103]),
                                      .B_read_data_out_8_oc3 (conv2_B_data[104]),
                                      .B_read_data_out_9_oc3 (conv2_B_data[105]),
                                      .B_read_data_out_10_oc3 (conv2_B_data[106]),
                                      .B_read_data_out_11_oc3 (conv2_B_data[107]),
                                      .B_read_data_out_12_oc3 (conv2_B_data[108]),
                                      .B_read_data_out_13_oc3 (conv2_B_data[109]),
                                      .B_read_data_out_14_oc3 (conv2_B_data[110]),
                                      .B_read_data_out_15_oc3 (conv2_B_data[111]),
                                      .B_read_data_out_16_oc3 (conv2_B_data[112]),
                                      .B_read_data_out_17_oc3 (conv2_B_data[113]),
                                      .B_read_data_out_18_oc3 (conv2_B_data[114]),
                                      .B_read_data_out_19_oc3 (conv2_B_data[115]),
                                      .B_read_data_out_20_oc3 (conv2_B_data[116]),
                                      .B_read_data_out_21_oc3 (conv2_B_data[117]),
                                      .B_read_data_out_22_oc3 (conv2_B_data[118]),
                                      .B_read_data_out_23_oc3 (conv2_B_data[119]),
                                      .B_read_data_out_24_oc3 (conv2_B_data[120]),
                                      .B_read_data_out_25_oc3 (conv2_B_data[121]),
                                      .B_read_data_out_26_oc3 (conv2_B_data[122]),
                                      .B_read_data_out_27_oc3 (conv2_B_data[123]),
                                      .B_read_data_out_28_oc3 (conv2_B_data[124]),
                                      .B_read_data_out_29_oc3 (conv2_B_data[125]),
                                      .B_read_data_out_30_oc3 (conv2_B_data[126]),
                                      .B_read_data_out_31_oc3 (conv2_B_data[127]),
                                      .C_read_en (conv2_C_read_en),
                                      .C_read_address (conv2_C_addr),
                                      .C_read_data_out_oc0 (conv2_C_data[0]),
                                      .C_read_data_out_oc1 (conv2_C_data[1]),
                                      .C_read_data_out_oc2 (conv2_C_data[2]),
                                      .C_read_data_out_oc3 (conv2_C_data[3]),
                                      .RES_write_en_even (conv2_out_we0),
                                      .RES_write_address_even (conv2_out_addr0),
                                      .RES_write_data_in_even (conv2_out_data0),
                                      .RES_write_en_odd (conv2_out_we1),
                                      .RES_write_address_odd (conv2_out_addr1),
                                      .RES_write_data_in_odd (conv2_out_data1));

    wire mp1_can_read, mp1_done_reading, mp1_A_read_en;
    wire [14:0] mp1_A_addr;
    wire [15:0] mp1_A_data;
    ping_pong_buffer_uram_tdp #(
                                  .DATA_WIDTH (16),
                                  .DEPTH_BITS (15)
                              ) conv2_to_mp1 (
                                  .clk                      (ACLK),
                                  .rst                      (~ARESETN),
                                  .producer_write_en_even   (conv2_out_we0),
                                  .producer_write_addr_even (conv2_out_addr0),
                                  .producer_write_data_even (conv2_out_data0),
                                  .producer_write_en_odd    (conv2_out_we1),
                                  .producer_write_addr_odd  (conv2_out_addr1),
                                  .producer_write_data_odd  (conv2_out_data1),
                                  .producer_frame_done      (conv2_done_writing),
                                  .producer_ready           (conv2_can_write),
                                  .consumer_read_en         (mp1_A_read_en),
                                  .consumer_read_addr       (mp1_A_addr),
                                  .consumer_read_data       (mp1_A_data),
                                  .consumer_frame_done      (mp1_done_reading),
                                  .consumer_ready           (mp1_can_read)
                              );
    wire mp1_can_write, mp1_done_writing;
    wire [31:0] mp1_we;
    wire [4:0] mp1_wchannel;
    wire [7:0] mp1_waddr;
    wire [15:0] mp1_wdata;
    max_pool_param_ping_pong #(
                                 .C_DIM          (32),
                                 .IN_H           (32),
                                 .IN_W           (32),
                                 .A_depth_bits   (15),
                                 .RES_depth_bits (8)
                             ) mp1 (
                                 .clk                 (ACLK),
                                 .rst                 (ARESETN),
                                 .consumer_ready      (mp1_can_read),
                                 .producer_ready      (mp1_can_write),
                                 .consumer_frame_done (mp1_done_reading),
                                 .producer_frame_done (mp1_done_writing),
                                 .A_read_en           (mp1_A_read_en),
                                 .A_read_address      (mp1_A_addr),
                                 .A_read_data_out     (mp1_A_data),
                                 .RES_write_en        (mp1_we),
                                 .RES_write_channel   (mp1_wchannel),
                                 .RES_write_address   (mp1_waddr),
                                 .RES_write_data_in   (mp1_wdata)
                             );
    wire conv3_can_read, conv3_done_reading, conv3_A_read_en;
    wire [7:0] conv3_A_addr;
    wire [511:0] conv3_A_data_bus;
    // Pool1 writes one channel per cycle. Eight channels share each UltraRAM,
    // while ping and pong frames use separate devices so Pool1 and Conv3 can
    // operate concurrently. The unused second write port is tied off.
    ping_pong_buffer_array_uram_packed8_2w #(
                                               .DATA_WIDTH   (16),
                                               .DEPTH_BITS   (8),
                                               .NUM_CHANNELS (32)
                                           ) mp1_to_conv3 (
                                               .clk                      (ACLK),
                                               .rst                      (~ARESETN),
                                               .producer_write_en_0      (|mp1_we),
                                               .producer_write_channel_0 (mp1_wchannel),
                                               .producer_write_addr_0    (mp1_waddr),
                                               .producer_write_data_0    (mp1_wdata),
                                               .producer_write_en_1      (1'b0),
                                               .producer_write_channel_1 (5'd0),
                                               .producer_write_addr_1    (8'd0),
                                               .producer_write_data_1    (16'd0),
                                               .producer_frame_done      (mp1_done_writing),
                                               .producer_ready           (mp1_can_write),
                                               .consumer_read_en         (conv3_A_read_en),
                                               .consumer_read_addr       (conv3_A_addr),
                                               .consumer_read_data_bus   (conv3_A_data_bus),
                                               .consumer_frame_done      (conv3_done_reading),
                                               .consumer_ready           (conv3_can_read)
                                           );
    wire [15:0] conv3_A_data[0:31];
    genvar g_conv3_A_data;
generate for (g_conv3_A_data = 0; g_conv3_A_data<32; g_conv3_A_data = g_conv3_A_data + 1) begin : CONV3_A_DATA_UNPACK
            assign conv3_A_data[g_conv3_A_data] = conv3_A_data_bus[g_conv3_A_data*16 +:16];
        end
    endgenerate

    wire conv3_can_write, conv3_done_writing, conv3_out_we0, conv3_out_we1;
    wire [13:0] conv3_out_addr0, conv3_out_addr1;
    wire [15:0] conv3_out_data0, conv3_out_data1;
    conv3_relu_unrolled_ping_pong #(.IC_DIM (32), .OC_DIM (64),
                                    .OUT_UNROLL (2), .IN_H (16), .IN_W (16),
                                    .A_depth_bits (8), .B_depth_bits (9),
                                    .C_depth_bits (5), .RES_depth_bits (14))conv3_inst(
                                      .clk (ACLK),
                                      .rst (ARESETN),
                                      .consumer_ready (conv3_can_read),
                                      .producer_ready (conv3_can_write),
                                      .consumer_frame_done (conv3_done_reading),
                                      .producer_frame_done (conv3_done_writing),
                                      .A_read_en (conv3_A_read_en),
                                      .A_read_address (conv3_A_addr),
                                      .A_read_data_out_0 (conv3_A_data[0]),
                                      .A_read_data_out_1 (conv3_A_data[1]),
                                      .A_read_data_out_2 (conv3_A_data[2]),
                                      .A_read_data_out_3 (conv3_A_data[3]),
                                      .A_read_data_out_4 (conv3_A_data[4]),
                                      .A_read_data_out_5 (conv3_A_data[5]),
                                      .A_read_data_out_6 (conv3_A_data[6]),
                                      .A_read_data_out_7 (conv3_A_data[7]),
                                      .A_read_data_out_8 (conv3_A_data[8]),
                                      .A_read_data_out_9 (conv3_A_data[9]),
                                      .A_read_data_out_10 (conv3_A_data[10]),
                                      .A_read_data_out_11 (conv3_A_data[11]),
                                      .A_read_data_out_12 (conv3_A_data[12]),
                                      .A_read_data_out_13 (conv3_A_data[13]),
                                      .A_read_data_out_14 (conv3_A_data[14]),
                                      .A_read_data_out_15 (conv3_A_data[15]),
                                      .A_read_data_out_16 (conv3_A_data[16]),
                                      .A_read_data_out_17 (conv3_A_data[17]),
                                      .A_read_data_out_18 (conv3_A_data[18]),
                                      .A_read_data_out_19 (conv3_A_data[19]),
                                      .A_read_data_out_20 (conv3_A_data[20]),
                                      .A_read_data_out_21 (conv3_A_data[21]),
                                      .A_read_data_out_22 (conv3_A_data[22]),
                                      .A_read_data_out_23 (conv3_A_data[23]),
                                      .A_read_data_out_24 (conv3_A_data[24]),
                                      .A_read_data_out_25 (conv3_A_data[25]),
                                      .A_read_data_out_26 (conv3_A_data[26]),
                                      .A_read_data_out_27 (conv3_A_data[27]),
                                      .A_read_data_out_28 (conv3_A_data[28]),
                                      .A_read_data_out_29 (conv3_A_data[29]),
                                      .A_read_data_out_30 (conv3_A_data[30]),
                                      .A_read_data_out_31 (conv3_A_data[31]),
                                      .B_read_en (conv3_B_read_en),
                                      .B_read_address (conv3_B_addr),
                                      .B_read_data_out_0_oc0 (conv3_B_data[0]),
                                      .B_read_data_out_1_oc0 (conv3_B_data[1]),
                                      .B_read_data_out_2_oc0 (conv3_B_data[2]),
                                      .B_read_data_out_3_oc0 (conv3_B_data[3]),
                                      .B_read_data_out_4_oc0 (conv3_B_data[4]),
                                      .B_read_data_out_5_oc0 (conv3_B_data[5]),
                                      .B_read_data_out_6_oc0 (conv3_B_data[6]),
                                      .B_read_data_out_7_oc0 (conv3_B_data[7]),
                                      .B_read_data_out_8_oc0 (conv3_B_data[8]),
                                      .B_read_data_out_9_oc0 (conv3_B_data[9]),
                                      .B_read_data_out_10_oc0 (conv3_B_data[10]),
                                      .B_read_data_out_11_oc0 (conv3_B_data[11]),
                                      .B_read_data_out_12_oc0 (conv3_B_data[12]),
                                      .B_read_data_out_13_oc0 (conv3_B_data[13]),
                                      .B_read_data_out_14_oc0 (conv3_B_data[14]),
                                      .B_read_data_out_15_oc0 (conv3_B_data[15]),
                                      .B_read_data_out_16_oc0 (conv3_B_data[16]),
                                      .B_read_data_out_17_oc0 (conv3_B_data[17]),
                                      .B_read_data_out_18_oc0 (conv3_B_data[18]),
                                      .B_read_data_out_19_oc0 (conv3_B_data[19]),
                                      .B_read_data_out_20_oc0 (conv3_B_data[20]),
                                      .B_read_data_out_21_oc0 (conv3_B_data[21]),
                                      .B_read_data_out_22_oc0 (conv3_B_data[22]),
                                      .B_read_data_out_23_oc0 (conv3_B_data[23]),
                                      .B_read_data_out_24_oc0 (conv3_B_data[24]),
                                      .B_read_data_out_25_oc0 (conv3_B_data[25]),
                                      .B_read_data_out_26_oc0 (conv3_B_data[26]),
                                      .B_read_data_out_27_oc0 (conv3_B_data[27]),
                                      .B_read_data_out_28_oc0 (conv3_B_data[28]),
                                      .B_read_data_out_29_oc0 (conv3_B_data[29]),
                                      .B_read_data_out_30_oc0 (conv3_B_data[30]),
                                      .B_read_data_out_31_oc0 (conv3_B_data[31]),
                                      .B_read_data_out_0_oc1 (conv3_B_data[32]),
                                      .B_read_data_out_1_oc1 (conv3_B_data[33]),
                                      .B_read_data_out_2_oc1 (conv3_B_data[34]),
                                      .B_read_data_out_3_oc1 (conv3_B_data[35]),
                                      .B_read_data_out_4_oc1 (conv3_B_data[36]),
                                      .B_read_data_out_5_oc1 (conv3_B_data[37]),
                                      .B_read_data_out_6_oc1 (conv3_B_data[38]),
                                      .B_read_data_out_7_oc1 (conv3_B_data[39]),
                                      .B_read_data_out_8_oc1 (conv3_B_data[40]),
                                      .B_read_data_out_9_oc1 (conv3_B_data[41]),
                                      .B_read_data_out_10_oc1 (conv3_B_data[42]),
                                      .B_read_data_out_11_oc1 (conv3_B_data[43]),
                                      .B_read_data_out_12_oc1 (conv3_B_data[44]),
                                      .B_read_data_out_13_oc1 (conv3_B_data[45]),
                                      .B_read_data_out_14_oc1 (conv3_B_data[46]),
                                      .B_read_data_out_15_oc1 (conv3_B_data[47]),
                                      .B_read_data_out_16_oc1 (conv3_B_data[48]),
                                      .B_read_data_out_17_oc1 (conv3_B_data[49]),
                                      .B_read_data_out_18_oc1 (conv3_B_data[50]),
                                      .B_read_data_out_19_oc1 (conv3_B_data[51]),
                                      .B_read_data_out_20_oc1 (conv3_B_data[52]),
                                      .B_read_data_out_21_oc1 (conv3_B_data[53]),
                                      .B_read_data_out_22_oc1 (conv3_B_data[54]),
                                      .B_read_data_out_23_oc1 (conv3_B_data[55]),
                                      .B_read_data_out_24_oc1 (conv3_B_data[56]),
                                      .B_read_data_out_25_oc1 (conv3_B_data[57]),
                                      .B_read_data_out_26_oc1 (conv3_B_data[58]),
                                      .B_read_data_out_27_oc1 (conv3_B_data[59]),
                                      .B_read_data_out_28_oc1 (conv3_B_data[60]),
                                      .B_read_data_out_29_oc1 (conv3_B_data[61]),
                                      .B_read_data_out_30_oc1 (conv3_B_data[62]),
                                      .B_read_data_out_31_oc1 (conv3_B_data[63]),
                                      .C_read_en (conv3_C_read_en),
                                      .C_read_address (conv3_C_addr),
                                      .C_read_data_out_oc0 (conv3_C_data[0]),
                                      .C_read_data_out_oc1 (conv3_C_data[1]),
                                      .RES_write_en_even (conv3_out_we0),
                                      .RES_write_address_even (conv3_out_addr0),
                                      .RES_write_data_in_even (conv3_out_data0),
                                      .RES_write_en_odd (conv3_out_we1),
                                      .RES_write_address_odd (conv3_out_addr1),
                                      .RES_write_data_in_odd (conv3_out_data1));

    wire conv4_can_read, conv4_done_reading, conv4_A_read_en;
    wire [7:0] conv4_A_addr;
    wire [1023:0] conv4_A_data_bus;
    // Each channel uses one combined-frame TDP RAMB18. Ping/pong select the
    // upper address bit; Conv3 writes port A while Conv4 reads port B.
    ping_pong_buffer_array_bram_tdp_2w #(
                                           .DATA_WIDTH   (16),
                                           .DEPTH_BITS   (8),
                                           .NUM_CHANNELS (64)
                                       ) conv3_to_conv4 (
                                           .clk                      (ACLK),
                                           .rst                      (~ARESETN),
                                           .producer_write_en_0      (conv3_out_we0),
                                           .producer_write_channel_0 (conv3_out_addr0[13:8]),
                                           .producer_write_addr_0    (conv3_out_addr0[7:0]),
                                           .producer_write_data_0    (conv3_out_data0),
                                           .producer_write_en_1      (conv3_out_we1),
                                           .producer_write_channel_1 (conv3_out_addr1[13:8]),
                                           .producer_write_addr_1    (conv3_out_addr1[7:0]),
                                           .producer_write_data_1    (conv3_out_data1),
                                           .producer_frame_done      (conv3_done_writing),
                                           .producer_ready           (conv3_can_write),
                                           .consumer_read_en         (conv4_A_read_en),
                                           .consumer_read_addr       (conv4_A_addr),
                                           .consumer_read_data_bus   (conv4_A_data_bus),
                                           .consumer_frame_done      (conv4_done_reading),
                                           .consumer_ready           (conv4_can_read)
                                       );
    wire [15:0] conv4_A_data[0:63];
    genvar g_conv4_A_data;
generate for (g_conv4_A_data = 0; g_conv4_A_data<64; g_conv4_A_data = g_conv4_A_data + 1) begin : CONV4_A_DATA_UNPACK
            assign conv4_A_data[g_conv4_A_data] = conv4_A_data_bus[g_conv4_A_data*16 +:16];
        end
    endgenerate

    wire conv4_can_write, conv4_done_writing, conv4_out_we0, conv4_out_we1;
    wire [13:0] conv4_out_addr0, conv4_out_addr1;
    wire [15:0] conv4_out_data0, conv4_out_data1;
    conv4_relu_unrolled_ping_pong #(.IC_DIM (64), .OC_DIM (64),
                                    .OUT_UNROLL (2), .IN_H (16), .IN_W (16),
                                    .A_depth_bits (8), .B_depth_bits (9),
                                    .C_depth_bits (5), .RES_depth_bits (14))conv4_inst(
                                      .clk (ACLK),
                                      .rst (ARESETN),
                                      .consumer_ready (conv4_can_read),
                                      .producer_ready (conv4_can_write),
                                      .consumer_frame_done (conv4_done_reading),
                                      .producer_frame_done (conv4_done_writing),
                                      .A_read_en (conv4_A_read_en),
                                      .A_read_address (conv4_A_addr),
                                      .A_read_data_out_0 (conv4_A_data[0]),
                                      .A_read_data_out_1 (conv4_A_data[1]),
                                      .A_read_data_out_2 (conv4_A_data[2]),
                                      .A_read_data_out_3 (conv4_A_data[3]),
                                      .A_read_data_out_4 (conv4_A_data[4]),
                                      .A_read_data_out_5 (conv4_A_data[5]),
                                      .A_read_data_out_6 (conv4_A_data[6]),
                                      .A_read_data_out_7 (conv4_A_data[7]),
                                      .A_read_data_out_8 (conv4_A_data[8]),
                                      .A_read_data_out_9 (conv4_A_data[9]),
                                      .A_read_data_out_10 (conv4_A_data[10]),
                                      .A_read_data_out_11 (conv4_A_data[11]),
                                      .A_read_data_out_12 (conv4_A_data[12]),
                                      .A_read_data_out_13 (conv4_A_data[13]),
                                      .A_read_data_out_14 (conv4_A_data[14]),
                                      .A_read_data_out_15 (conv4_A_data[15]),
                                      .A_read_data_out_16 (conv4_A_data[16]),
                                      .A_read_data_out_17 (conv4_A_data[17]),
                                      .A_read_data_out_18 (conv4_A_data[18]),
                                      .A_read_data_out_19 (conv4_A_data[19]),
                                      .A_read_data_out_20 (conv4_A_data[20]),
                                      .A_read_data_out_21 (conv4_A_data[21]),
                                      .A_read_data_out_22 (conv4_A_data[22]),
                                      .A_read_data_out_23 (conv4_A_data[23]),
                                      .A_read_data_out_24 (conv4_A_data[24]),
                                      .A_read_data_out_25 (conv4_A_data[25]),
                                      .A_read_data_out_26 (conv4_A_data[26]),
                                      .A_read_data_out_27 (conv4_A_data[27]),
                                      .A_read_data_out_28 (conv4_A_data[28]),
                                      .A_read_data_out_29 (conv4_A_data[29]),
                                      .A_read_data_out_30 (conv4_A_data[30]),
                                      .A_read_data_out_31 (conv4_A_data[31]),
                                      .A_read_data_out_32 (conv4_A_data[32]),
                                      .A_read_data_out_33 (conv4_A_data[33]),
                                      .A_read_data_out_34 (conv4_A_data[34]),
                                      .A_read_data_out_35 (conv4_A_data[35]),
                                      .A_read_data_out_36 (conv4_A_data[36]),
                                      .A_read_data_out_37 (conv4_A_data[37]),
                                      .A_read_data_out_38 (conv4_A_data[38]),
                                      .A_read_data_out_39 (conv4_A_data[39]),
                                      .A_read_data_out_40 (conv4_A_data[40]),
                                      .A_read_data_out_41 (conv4_A_data[41]),
                                      .A_read_data_out_42 (conv4_A_data[42]),
                                      .A_read_data_out_43 (conv4_A_data[43]),
                                      .A_read_data_out_44 (conv4_A_data[44]),
                                      .A_read_data_out_45 (conv4_A_data[45]),
                                      .A_read_data_out_46 (conv4_A_data[46]),
                                      .A_read_data_out_47 (conv4_A_data[47]),
                                      .A_read_data_out_48 (conv4_A_data[48]),
                                      .A_read_data_out_49 (conv4_A_data[49]),
                                      .A_read_data_out_50 (conv4_A_data[50]),
                                      .A_read_data_out_51 (conv4_A_data[51]),
                                      .A_read_data_out_52 (conv4_A_data[52]),
                                      .A_read_data_out_53 (conv4_A_data[53]),
                                      .A_read_data_out_54 (conv4_A_data[54]),
                                      .A_read_data_out_55 (conv4_A_data[55]),
                                      .A_read_data_out_56 (conv4_A_data[56]),
                                      .A_read_data_out_57 (conv4_A_data[57]),
                                      .A_read_data_out_58 (conv4_A_data[58]),
                                      .A_read_data_out_59 (conv4_A_data[59]),
                                      .A_read_data_out_60 (conv4_A_data[60]),
                                      .A_read_data_out_61 (conv4_A_data[61]),
                                      .A_read_data_out_62 (conv4_A_data[62]),
                                      .A_read_data_out_63 (conv4_A_data[63]),
                                      .B_read_en (conv4_B_read_en),
                                      .B_read_address (conv4_B_addr),
                                      .B_read_data_out_0_oc0 (conv4_B_data[0]),
                                      .B_read_data_out_1_oc0 (conv4_B_data[1]),
                                      .B_read_data_out_2_oc0 (conv4_B_data[2]),
                                      .B_read_data_out_3_oc0 (conv4_B_data[3]),
                                      .B_read_data_out_4_oc0 (conv4_B_data[4]),
                                      .B_read_data_out_5_oc0 (conv4_B_data[5]),
                                      .B_read_data_out_6_oc0 (conv4_B_data[6]),
                                      .B_read_data_out_7_oc0 (conv4_B_data[7]),
                                      .B_read_data_out_8_oc0 (conv4_B_data[8]),
                                      .B_read_data_out_9_oc0 (conv4_B_data[9]),
                                      .B_read_data_out_10_oc0 (conv4_B_data[10]),
                                      .B_read_data_out_11_oc0 (conv4_B_data[11]),
                                      .B_read_data_out_12_oc0 (conv4_B_data[12]),
                                      .B_read_data_out_13_oc0 (conv4_B_data[13]),
                                      .B_read_data_out_14_oc0 (conv4_B_data[14]),
                                      .B_read_data_out_15_oc0 (conv4_B_data[15]),
                                      .B_read_data_out_16_oc0 (conv4_B_data[16]),
                                      .B_read_data_out_17_oc0 (conv4_B_data[17]),
                                      .B_read_data_out_18_oc0 (conv4_B_data[18]),
                                      .B_read_data_out_19_oc0 (conv4_B_data[19]),
                                      .B_read_data_out_20_oc0 (conv4_B_data[20]),
                                      .B_read_data_out_21_oc0 (conv4_B_data[21]),
                                      .B_read_data_out_22_oc0 (conv4_B_data[22]),
                                      .B_read_data_out_23_oc0 (conv4_B_data[23]),
                                      .B_read_data_out_24_oc0 (conv4_B_data[24]),
                                      .B_read_data_out_25_oc0 (conv4_B_data[25]),
                                      .B_read_data_out_26_oc0 (conv4_B_data[26]),
                                      .B_read_data_out_27_oc0 (conv4_B_data[27]),
                                      .B_read_data_out_28_oc0 (conv4_B_data[28]),
                                      .B_read_data_out_29_oc0 (conv4_B_data[29]),
                                      .B_read_data_out_30_oc0 (conv4_B_data[30]),
                                      .B_read_data_out_31_oc0 (conv4_B_data[31]),
                                      .B_read_data_out_32_oc0 (conv4_B_data[32]),
                                      .B_read_data_out_33_oc0 (conv4_B_data[33]),
                                      .B_read_data_out_34_oc0 (conv4_B_data[34]),
                                      .B_read_data_out_35_oc0 (conv4_B_data[35]),
                                      .B_read_data_out_36_oc0 (conv4_B_data[36]),
                                      .B_read_data_out_37_oc0 (conv4_B_data[37]),
                                      .B_read_data_out_38_oc0 (conv4_B_data[38]),
                                      .B_read_data_out_39_oc0 (conv4_B_data[39]),
                                      .B_read_data_out_40_oc0 (conv4_B_data[40]),
                                      .B_read_data_out_41_oc0 (conv4_B_data[41]),
                                      .B_read_data_out_42_oc0 (conv4_B_data[42]),
                                      .B_read_data_out_43_oc0 (conv4_B_data[43]),
                                      .B_read_data_out_44_oc0 (conv4_B_data[44]),
                                      .B_read_data_out_45_oc0 (conv4_B_data[45]),
                                      .B_read_data_out_46_oc0 (conv4_B_data[46]),
                                      .B_read_data_out_47_oc0 (conv4_B_data[47]),
                                      .B_read_data_out_48_oc0 (conv4_B_data[48]),
                                      .B_read_data_out_49_oc0 (conv4_B_data[49]),
                                      .B_read_data_out_50_oc0 (conv4_B_data[50]),
                                      .B_read_data_out_51_oc0 (conv4_B_data[51]),
                                      .B_read_data_out_52_oc0 (conv4_B_data[52]),
                                      .B_read_data_out_53_oc0 (conv4_B_data[53]),
                                      .B_read_data_out_54_oc0 (conv4_B_data[54]),
                                      .B_read_data_out_55_oc0 (conv4_B_data[55]),
                                      .B_read_data_out_56_oc0 (conv4_B_data[56]),
                                      .B_read_data_out_57_oc0 (conv4_B_data[57]),
                                      .B_read_data_out_58_oc0 (conv4_B_data[58]),
                                      .B_read_data_out_59_oc0 (conv4_B_data[59]),
                                      .B_read_data_out_60_oc0 (conv4_B_data[60]),
                                      .B_read_data_out_61_oc0 (conv4_B_data[61]),
                                      .B_read_data_out_62_oc0 (conv4_B_data[62]),
                                      .B_read_data_out_63_oc0 (conv4_B_data[63]),
                                      .B_read_data_out_0_oc1 (conv4_B_data[64]),
                                      .B_read_data_out_1_oc1 (conv4_B_data[65]),
                                      .B_read_data_out_2_oc1 (conv4_B_data[66]),
                                      .B_read_data_out_3_oc1 (conv4_B_data[67]),
                                      .B_read_data_out_4_oc1 (conv4_B_data[68]),
                                      .B_read_data_out_5_oc1 (conv4_B_data[69]),
                                      .B_read_data_out_6_oc1 (conv4_B_data[70]),
                                      .B_read_data_out_7_oc1 (conv4_B_data[71]),
                                      .B_read_data_out_8_oc1 (conv4_B_data[72]),
                                      .B_read_data_out_9_oc1 (conv4_B_data[73]),
                                      .B_read_data_out_10_oc1 (conv4_B_data[74]),
                                      .B_read_data_out_11_oc1 (conv4_B_data[75]),
                                      .B_read_data_out_12_oc1 (conv4_B_data[76]),
                                      .B_read_data_out_13_oc1 (conv4_B_data[77]),
                                      .B_read_data_out_14_oc1 (conv4_B_data[78]),
                                      .B_read_data_out_15_oc1 (conv4_B_data[79]),
                                      .B_read_data_out_16_oc1 (conv4_B_data[80]),
                                      .B_read_data_out_17_oc1 (conv4_B_data[81]),
                                      .B_read_data_out_18_oc1 (conv4_B_data[82]),
                                      .B_read_data_out_19_oc1 (conv4_B_data[83]),
                                      .B_read_data_out_20_oc1 (conv4_B_data[84]),
                                      .B_read_data_out_21_oc1 (conv4_B_data[85]),
                                      .B_read_data_out_22_oc1 (conv4_B_data[86]),
                                      .B_read_data_out_23_oc1 (conv4_B_data[87]),
                                      .B_read_data_out_24_oc1 (conv4_B_data[88]),
                                      .B_read_data_out_25_oc1 (conv4_B_data[89]),
                                      .B_read_data_out_26_oc1 (conv4_B_data[90]),
                                      .B_read_data_out_27_oc1 (conv4_B_data[91]),
                                      .B_read_data_out_28_oc1 (conv4_B_data[92]),
                                      .B_read_data_out_29_oc1 (conv4_B_data[93]),
                                      .B_read_data_out_30_oc1 (conv4_B_data[94]),
                                      .B_read_data_out_31_oc1 (conv4_B_data[95]),
                                      .B_read_data_out_32_oc1 (conv4_B_data[96]),
                                      .B_read_data_out_33_oc1 (conv4_B_data[97]),
                                      .B_read_data_out_34_oc1 (conv4_B_data[98]),
                                      .B_read_data_out_35_oc1 (conv4_B_data[99]),
                                      .B_read_data_out_36_oc1 (conv4_B_data[100]),
                                      .B_read_data_out_37_oc1 (conv4_B_data[101]),
                                      .B_read_data_out_38_oc1 (conv4_B_data[102]),
                                      .B_read_data_out_39_oc1 (conv4_B_data[103]),
                                      .B_read_data_out_40_oc1 (conv4_B_data[104]),
                                      .B_read_data_out_41_oc1 (conv4_B_data[105]),
                                      .B_read_data_out_42_oc1 (conv4_B_data[106]),
                                      .B_read_data_out_43_oc1 (conv4_B_data[107]),
                                      .B_read_data_out_44_oc1 (conv4_B_data[108]),
                                      .B_read_data_out_45_oc1 (conv4_B_data[109]),
                                      .B_read_data_out_46_oc1 (conv4_B_data[110]),
                                      .B_read_data_out_47_oc1 (conv4_B_data[111]),
                                      .B_read_data_out_48_oc1 (conv4_B_data[112]),
                                      .B_read_data_out_49_oc1 (conv4_B_data[113]),
                                      .B_read_data_out_50_oc1 (conv4_B_data[114]),
                                      .B_read_data_out_51_oc1 (conv4_B_data[115]),
                                      .B_read_data_out_52_oc1 (conv4_B_data[116]),
                                      .B_read_data_out_53_oc1 (conv4_B_data[117]),
                                      .B_read_data_out_54_oc1 (conv4_B_data[118]),
                                      .B_read_data_out_55_oc1 (conv4_B_data[119]),
                                      .B_read_data_out_56_oc1 (conv4_B_data[120]),
                                      .B_read_data_out_57_oc1 (conv4_B_data[121]),
                                      .B_read_data_out_58_oc1 (conv4_B_data[122]),
                                      .B_read_data_out_59_oc1 (conv4_B_data[123]),
                                      .B_read_data_out_60_oc1 (conv4_B_data[124]),
                                      .B_read_data_out_61_oc1 (conv4_B_data[125]),
                                      .B_read_data_out_62_oc1 (conv4_B_data[126]),
                                      .B_read_data_out_63_oc1 (conv4_B_data[127]),
                                      .C_read_en (conv4_C_read_en),
                                      .C_read_address (conv4_C_addr),
                                      .C_read_data_out_oc0 (conv4_C_data[0]),
                                      .C_read_data_out_oc1 (conv4_C_data[1]),
                                      .RES_write_en_even (conv4_out_we0),
                                      .RES_write_address_even (conv4_out_addr0),
                                      .RES_write_data_in_even (conv4_out_data0),
                                      .RES_write_en_odd (conv4_out_we1),
                                      .RES_write_address_odd (conv4_out_addr1),
                                      .RES_write_data_in_odd (conv4_out_data1));

    wire mp2_can_read, mp2_done_reading, mp2_A_read_en;
    wire [13:0] mp2_A_addr;
    wire [15:0] mp2_A_data;
    ping_pong_buffer_tdp #(
                             .DATA_WIDTH (16),
                             .DEPTH_BITS (14)
                         ) conv4_to_mp2 (
                             .clk                      (ACLK),
                             .rst                      (~ARESETN),
                             .producer_write_en_even   (conv4_out_we0),
                             .producer_write_addr_even (conv4_out_addr0),
                             .producer_write_data_even (conv4_out_data0),
                             .producer_write_en_odd    (conv4_out_we1),
                             .producer_write_addr_odd  (conv4_out_addr1),
                             .producer_write_data_odd  (conv4_out_data1),
                             .producer_frame_done      (conv4_done_writing),
                             .producer_ready           (conv4_can_write),
                             .consumer_read_en         (mp2_A_read_en),
                             .consumer_read_addr       (mp2_A_addr),
                             .consumer_read_data       (mp2_A_data),
                             .consumer_frame_done      (mp2_done_reading),
                             .consumer_ready           (mp2_can_read)
                         );
    wire mp2_can_write, mp2_done_writing;
    wire [63:0] mp2_we;
    wire [5:0] mp2_wchannel;
    wire [5:0] mp2_waddr;
    wire [15:0] mp2_wdata;
    max_pool_param_ping_pong #(
                                 .C_DIM          (64),
                                 .IN_H           (16),
                                 .IN_W           (16),
                                 .A_depth_bits   (14),
                                 .RES_depth_bits (6)
                             ) mp2 (
                                 .clk                 (ACLK),
                                 .rst                 (ARESETN),
                                 .consumer_ready      (mp2_can_read),
                                 .producer_ready      (mp2_can_write),
                                 .consumer_frame_done (mp2_done_reading),
                                 .producer_frame_done (mp2_done_writing),
                                 .A_read_en           (mp2_A_read_en),
                                 .A_read_address      (mp2_A_addr),
                                 .A_read_data_out     (mp2_A_data),
                                 .RES_write_en        (mp2_we),
                                 .RES_write_channel   (mp2_wchannel),
                                 .RES_write_address   (mp2_waddr),
                                 .RES_write_data_in   (mp2_wdata)
                             );
    wire conv5_can_read, conv5_done_reading, conv5_A_read_en;
    wire [5:0] conv5_A_addr;
    wire [1023:0] conv5_A_data_bus;
    ping_pong_buffer_array #(
                               .DATA_WIDTH   (16),
                               .DEPTH_BITS   (6),
                               .NUM_CHANNELS (64),
                               .MEM_TYPE     ("LUTRAM")
                           ) mp2_to_conv5 (
                               .clk                     (ACLK),
                               .rst                     (~ARESETN),
                               .producer_write_en_bus   (mp2_we),
                               .producer_write_addr     (mp2_waddr),
                               .producer_write_data_bus ({64{mp2_wdata}}),
                               .producer_frame_done     (mp2_done_writing),
                               .producer_ready          (mp2_can_write),
                               .consumer_read_en        (conv5_A_read_en),
                               .consumer_read_addr      (conv5_A_addr),
                               .consumer_read_data_bus  (conv5_A_data_bus),
                               .consumer_frame_done     (conv5_done_reading),
                               .consumer_ready          (conv5_can_read)
                           );
    wire [15:0] conv5_A_data[0:63];
    genvar g_conv5_A_data;
generate for (g_conv5_A_data = 0; g_conv5_A_data<64; g_conv5_A_data = g_conv5_A_data + 1) begin : CONV5_A_DATA_UNPACK
            assign conv5_A_data[g_conv5_A_data] = conv5_A_data_bus[g_conv5_A_data*16 +:16];
        end
    endgenerate

    wire conv5_can_write, conv5_done_writing, conv5_out_we;
    wire [12:0] conv5_out_addr;
    wire [15:0] conv5_out_data;
    conv5_relu_unrolled_ping_pong #(.IC_DIM (64), .OC_DIM (128),
                                    .OUT_UNROLL (1), .IN_H (8), .IN_W (8),
                                    .A_depth_bits (6), .B_depth_bits (11),
                                    .C_depth_bits (7), .RES_depth_bits (13))conv5_inst(
                                      .clk (ACLK),
                                      .rst (ARESETN),
                                      .consumer_ready (conv5_can_read),
                                      .producer_ready (conv5_can_write),
                                      .consumer_frame_done (conv5_done_reading),
                                      .producer_frame_done (conv5_done_writing),
                                      .A_read_en (conv5_A_read_en),
                                      .A_read_address (conv5_A_addr),
                                      .A_read_data_out_0 (conv5_A_data[0]),
                                      .A_read_data_out_1 (conv5_A_data[1]),
                                      .A_read_data_out_2 (conv5_A_data[2]),
                                      .A_read_data_out_3 (conv5_A_data[3]),
                                      .A_read_data_out_4 (conv5_A_data[4]),
                                      .A_read_data_out_5 (conv5_A_data[5]),
                                      .A_read_data_out_6 (conv5_A_data[6]),
                                      .A_read_data_out_7 (conv5_A_data[7]),
                                      .A_read_data_out_8 (conv5_A_data[8]),
                                      .A_read_data_out_9 (conv5_A_data[9]),
                                      .A_read_data_out_10 (conv5_A_data[10]),
                                      .A_read_data_out_11 (conv5_A_data[11]),
                                      .A_read_data_out_12 (conv5_A_data[12]),
                                      .A_read_data_out_13 (conv5_A_data[13]),
                                      .A_read_data_out_14 (conv5_A_data[14]),
                                      .A_read_data_out_15 (conv5_A_data[15]),
                                      .A_read_data_out_16 (conv5_A_data[16]),
                                      .A_read_data_out_17 (conv5_A_data[17]),
                                      .A_read_data_out_18 (conv5_A_data[18]),
                                      .A_read_data_out_19 (conv5_A_data[19]),
                                      .A_read_data_out_20 (conv5_A_data[20]),
                                      .A_read_data_out_21 (conv5_A_data[21]),
                                      .A_read_data_out_22 (conv5_A_data[22]),
                                      .A_read_data_out_23 (conv5_A_data[23]),
                                      .A_read_data_out_24 (conv5_A_data[24]),
                                      .A_read_data_out_25 (conv5_A_data[25]),
                                      .A_read_data_out_26 (conv5_A_data[26]),
                                      .A_read_data_out_27 (conv5_A_data[27]),
                                      .A_read_data_out_28 (conv5_A_data[28]),
                                      .A_read_data_out_29 (conv5_A_data[29]),
                                      .A_read_data_out_30 (conv5_A_data[30]),
                                      .A_read_data_out_31 (conv5_A_data[31]),
                                      .A_read_data_out_32 (conv5_A_data[32]),
                                      .A_read_data_out_33 (conv5_A_data[33]),
                                      .A_read_data_out_34 (conv5_A_data[34]),
                                      .A_read_data_out_35 (conv5_A_data[35]),
                                      .A_read_data_out_36 (conv5_A_data[36]),
                                      .A_read_data_out_37 (conv5_A_data[37]),
                                      .A_read_data_out_38 (conv5_A_data[38]),
                                      .A_read_data_out_39 (conv5_A_data[39]),
                                      .A_read_data_out_40 (conv5_A_data[40]),
                                      .A_read_data_out_41 (conv5_A_data[41]),
                                      .A_read_data_out_42 (conv5_A_data[42]),
                                      .A_read_data_out_43 (conv5_A_data[43]),
                                      .A_read_data_out_44 (conv5_A_data[44]),
                                      .A_read_data_out_45 (conv5_A_data[45]),
                                      .A_read_data_out_46 (conv5_A_data[46]),
                                      .A_read_data_out_47 (conv5_A_data[47]),
                                      .A_read_data_out_48 (conv5_A_data[48]),
                                      .A_read_data_out_49 (conv5_A_data[49]),
                                      .A_read_data_out_50 (conv5_A_data[50]),
                                      .A_read_data_out_51 (conv5_A_data[51]),
                                      .A_read_data_out_52 (conv5_A_data[52]),
                                      .A_read_data_out_53 (conv5_A_data[53]),
                                      .A_read_data_out_54 (conv5_A_data[54]),
                                      .A_read_data_out_55 (conv5_A_data[55]),
                                      .A_read_data_out_56 (conv5_A_data[56]),
                                      .A_read_data_out_57 (conv5_A_data[57]),
                                      .A_read_data_out_58 (conv5_A_data[58]),
                                      .A_read_data_out_59 (conv5_A_data[59]),
                                      .A_read_data_out_60 (conv5_A_data[60]),
                                      .A_read_data_out_61 (conv5_A_data[61]),
                                      .A_read_data_out_62 (conv5_A_data[62]),
                                      .A_read_data_out_63 (conv5_A_data[63]),
                                      .B_read_en (conv5_B_read_en),
                                      .B_read_address (conv5_B_addr),
                                      .B_read_data_out_0_oc0 (conv5_B_data[0]),
                                      .B_read_data_out_1_oc0 (conv5_B_data[1]),
                                      .B_read_data_out_2_oc0 (conv5_B_data[2]),
                                      .B_read_data_out_3_oc0 (conv5_B_data[3]),
                                      .B_read_data_out_4_oc0 (conv5_B_data[4]),
                                      .B_read_data_out_5_oc0 (conv5_B_data[5]),
                                      .B_read_data_out_6_oc0 (conv5_B_data[6]),
                                      .B_read_data_out_7_oc0 (conv5_B_data[7]),
                                      .B_read_data_out_8_oc0 (conv5_B_data[8]),
                                      .B_read_data_out_9_oc0 (conv5_B_data[9]),
                                      .B_read_data_out_10_oc0 (conv5_B_data[10]),
                                      .B_read_data_out_11_oc0 (conv5_B_data[11]),
                                      .B_read_data_out_12_oc0 (conv5_B_data[12]),
                                      .B_read_data_out_13_oc0 (conv5_B_data[13]),
                                      .B_read_data_out_14_oc0 (conv5_B_data[14]),
                                      .B_read_data_out_15_oc0 (conv5_B_data[15]),
                                      .B_read_data_out_16_oc0 (conv5_B_data[16]),
                                      .B_read_data_out_17_oc0 (conv5_B_data[17]),
                                      .B_read_data_out_18_oc0 (conv5_B_data[18]),
                                      .B_read_data_out_19_oc0 (conv5_B_data[19]),
                                      .B_read_data_out_20_oc0 (conv5_B_data[20]),
                                      .B_read_data_out_21_oc0 (conv5_B_data[21]),
                                      .B_read_data_out_22_oc0 (conv5_B_data[22]),
                                      .B_read_data_out_23_oc0 (conv5_B_data[23]),
                                      .B_read_data_out_24_oc0 (conv5_B_data[24]),
                                      .B_read_data_out_25_oc0 (conv5_B_data[25]),
                                      .B_read_data_out_26_oc0 (conv5_B_data[26]),
                                      .B_read_data_out_27_oc0 (conv5_B_data[27]),
                                      .B_read_data_out_28_oc0 (conv5_B_data[28]),
                                      .B_read_data_out_29_oc0 (conv5_B_data[29]),
                                      .B_read_data_out_30_oc0 (conv5_B_data[30]),
                                      .B_read_data_out_31_oc0 (conv5_B_data[31]),
                                      .B_read_data_out_32_oc0 (conv5_B_data[32]),
                                      .B_read_data_out_33_oc0 (conv5_B_data[33]),
                                      .B_read_data_out_34_oc0 (conv5_B_data[34]),
                                      .B_read_data_out_35_oc0 (conv5_B_data[35]),
                                      .B_read_data_out_36_oc0 (conv5_B_data[36]),
                                      .B_read_data_out_37_oc0 (conv5_B_data[37]),
                                      .B_read_data_out_38_oc0 (conv5_B_data[38]),
                                      .B_read_data_out_39_oc0 (conv5_B_data[39]),
                                      .B_read_data_out_40_oc0 (conv5_B_data[40]),
                                      .B_read_data_out_41_oc0 (conv5_B_data[41]),
                                      .B_read_data_out_42_oc0 (conv5_B_data[42]),
                                      .B_read_data_out_43_oc0 (conv5_B_data[43]),
                                      .B_read_data_out_44_oc0 (conv5_B_data[44]),
                                      .B_read_data_out_45_oc0 (conv5_B_data[45]),
                                      .B_read_data_out_46_oc0 (conv5_B_data[46]),
                                      .B_read_data_out_47_oc0 (conv5_B_data[47]),
                                      .B_read_data_out_48_oc0 (conv5_B_data[48]),
                                      .B_read_data_out_49_oc0 (conv5_B_data[49]),
                                      .B_read_data_out_50_oc0 (conv5_B_data[50]),
                                      .B_read_data_out_51_oc0 (conv5_B_data[51]),
                                      .B_read_data_out_52_oc0 (conv5_B_data[52]),
                                      .B_read_data_out_53_oc0 (conv5_B_data[53]),
                                      .B_read_data_out_54_oc0 (conv5_B_data[54]),
                                      .B_read_data_out_55_oc0 (conv5_B_data[55]),
                                      .B_read_data_out_56_oc0 (conv5_B_data[56]),
                                      .B_read_data_out_57_oc0 (conv5_B_data[57]),
                                      .B_read_data_out_58_oc0 (conv5_B_data[58]),
                                      .B_read_data_out_59_oc0 (conv5_B_data[59]),
                                      .B_read_data_out_60_oc0 (conv5_B_data[60]),
                                      .B_read_data_out_61_oc0 (conv5_B_data[61]),
                                      .B_read_data_out_62_oc0 (conv5_B_data[62]),
                                      .B_read_data_out_63_oc0 (conv5_B_data[63]),
                                      .C_read_en (conv5_C_read_en),
                                      .C_read_address (conv5_C_addr),
                                      .C_read_data_out_oc0 (conv5_C_data),
                                      .RES_write_en (conv5_out_we),
                                      .RES_write_address (conv5_out_addr),
                                      .RES_write_data (conv5_out_data));

    wire conv6_can_read, conv6_done_reading, conv6_A_read_en;
    wire [6:0] conv6_A_addr;
    wire [1023:0] conv6_A_data_bus;
    // Conv5 OU1 folds channels 0..63 and 64..127 into 64 parallel LUTRAM
    // banks. Address bit 6 selects the input-channel group consumed by Conv6.
    wire [63:0] conv5_out_we_bus =
         conv5_out_we ? (64'b1 << conv5_out_addr[11:6]) : 64'b0;
    ping_pong_buffer_array #(
                               .DATA_WIDTH   (16),
                               .DEPTH_BITS   (7),
                               .NUM_CHANNELS (64),
                               .MEM_TYPE     ("LUTRAM")
                           ) conv5_to_conv6 (
                               .clk                     (ACLK),
                               .rst                     (~ARESETN),
                               .producer_write_en_bus   (conv5_out_we_bus),
                               .producer_write_addr     ({conv5_out_addr[12], conv5_out_addr[5:0]}),
                               .producer_write_data_bus ({64{conv5_out_data}}),
                               .producer_frame_done     (conv5_done_writing),
                               .producer_ready          (conv5_can_write),
                               .consumer_read_en        (conv6_A_read_en),
                               .consumer_read_addr      (conv6_A_addr),
                               .consumer_read_data_bus  (conv6_A_data_bus),
                               .consumer_frame_done     (conv6_done_reading),
                               .consumer_ready          (conv6_can_read)
                           );
    wire [15:0] conv6_A_data[0:63];
    genvar g_conv6_A_data;
generate for (g_conv6_A_data = 0; g_conv6_A_data<64; g_conv6_A_data = g_conv6_A_data + 1) begin : CONV6_A_DATA_UNPACK
            assign conv6_A_data[g_conv6_A_data] = conv6_A_data_bus[g_conv6_A_data*16 +:16];
        end
    endgenerate

    wire conv6_can_write, conv6_done_writing, conv6_out_we0, conv6_out_we1;
    wire [12:0] conv6_out_addr0, conv6_out_addr1;
    wire [15:0] conv6_out_data0, conv6_out_data1;
    conv6_relu_unrolled_ping_pong #(.IC_DIM (128), .OC_DIM (128),
                                    .OUT_UNROLL (2), .IN_H (8), .IN_W (8),
                                    .A_depth_bits (7), .B_depth_bits (11),
                                    .C_depth_bits (6), .RES_depth_bits (13))conv6_inst(
                                      .clk (ACLK),
                                      .rst (ARESETN),
                                      .consumer_ready (conv6_can_read),
                                      .producer_ready (conv6_can_write),
                                      .consumer_frame_done (conv6_done_reading),
                                      .producer_frame_done (conv6_done_writing),
                                      .A_read_en (conv6_A_read_en),
                                      .A_read_address (conv6_A_addr),
                                      .A_read_data_out_0 (conv6_A_data[0]),
                                      .A_read_data_out_1 (conv6_A_data[1]),
                                      .A_read_data_out_2 (conv6_A_data[2]),
                                      .A_read_data_out_3 (conv6_A_data[3]),
                                      .A_read_data_out_4 (conv6_A_data[4]),
                                      .A_read_data_out_5 (conv6_A_data[5]),
                                      .A_read_data_out_6 (conv6_A_data[6]),
                                      .A_read_data_out_7 (conv6_A_data[7]),
                                      .A_read_data_out_8 (conv6_A_data[8]),
                                      .A_read_data_out_9 (conv6_A_data[9]),
                                      .A_read_data_out_10 (conv6_A_data[10]),
                                      .A_read_data_out_11 (conv6_A_data[11]),
                                      .A_read_data_out_12 (conv6_A_data[12]),
                                      .A_read_data_out_13 (conv6_A_data[13]),
                                      .A_read_data_out_14 (conv6_A_data[14]),
                                      .A_read_data_out_15 (conv6_A_data[15]),
                                      .A_read_data_out_16 (conv6_A_data[16]),
                                      .A_read_data_out_17 (conv6_A_data[17]),
                                      .A_read_data_out_18 (conv6_A_data[18]),
                                      .A_read_data_out_19 (conv6_A_data[19]),
                                      .A_read_data_out_20 (conv6_A_data[20]),
                                      .A_read_data_out_21 (conv6_A_data[21]),
                                      .A_read_data_out_22 (conv6_A_data[22]),
                                      .A_read_data_out_23 (conv6_A_data[23]),
                                      .A_read_data_out_24 (conv6_A_data[24]),
                                      .A_read_data_out_25 (conv6_A_data[25]),
                                      .A_read_data_out_26 (conv6_A_data[26]),
                                      .A_read_data_out_27 (conv6_A_data[27]),
                                      .A_read_data_out_28 (conv6_A_data[28]),
                                      .A_read_data_out_29 (conv6_A_data[29]),
                                      .A_read_data_out_30 (conv6_A_data[30]),
                                      .A_read_data_out_31 (conv6_A_data[31]),
                                      .A_read_data_out_32 (conv6_A_data[32]),
                                      .A_read_data_out_33 (conv6_A_data[33]),
                                      .A_read_data_out_34 (conv6_A_data[34]),
                                      .A_read_data_out_35 (conv6_A_data[35]),
                                      .A_read_data_out_36 (conv6_A_data[36]),
                                      .A_read_data_out_37 (conv6_A_data[37]),
                                      .A_read_data_out_38 (conv6_A_data[38]),
                                      .A_read_data_out_39 (conv6_A_data[39]),
                                      .A_read_data_out_40 (conv6_A_data[40]),
                                      .A_read_data_out_41 (conv6_A_data[41]),
                                      .A_read_data_out_42 (conv6_A_data[42]),
                                      .A_read_data_out_43 (conv6_A_data[43]),
                                      .A_read_data_out_44 (conv6_A_data[44]),
                                      .A_read_data_out_45 (conv6_A_data[45]),
                                      .A_read_data_out_46 (conv6_A_data[46]),
                                      .A_read_data_out_47 (conv6_A_data[47]),
                                      .A_read_data_out_48 (conv6_A_data[48]),
                                      .A_read_data_out_49 (conv6_A_data[49]),
                                      .A_read_data_out_50 (conv6_A_data[50]),
                                      .A_read_data_out_51 (conv6_A_data[51]),
                                      .A_read_data_out_52 (conv6_A_data[52]),
                                      .A_read_data_out_53 (conv6_A_data[53]),
                                      .A_read_data_out_54 (conv6_A_data[54]),
                                      .A_read_data_out_55 (conv6_A_data[55]),
                                      .A_read_data_out_56 (conv6_A_data[56]),
                                      .A_read_data_out_57 (conv6_A_data[57]),
                                      .A_read_data_out_58 (conv6_A_data[58]),
                                      .A_read_data_out_59 (conv6_A_data[59]),
                                      .A_read_data_out_60 (conv6_A_data[60]),
                                      .A_read_data_out_61 (conv6_A_data[61]),
                                      .A_read_data_out_62 (conv6_A_data[62]),
                                      .A_read_data_out_63 (conv6_A_data[63]),
                                      .B_read_en (conv6_B_read_en),
                                      .B_read_address (conv6_B_addr),
                                      .B_read_data_out_0_oc0 (conv6_B_data[0]),
                                      .B_read_data_out_1_oc0 (conv6_B_data[1]),
                                      .B_read_data_out_2_oc0 (conv6_B_data[2]),
                                      .B_read_data_out_3_oc0 (conv6_B_data[3]),
                                      .B_read_data_out_4_oc0 (conv6_B_data[4]),
                                      .B_read_data_out_5_oc0 (conv6_B_data[5]),
                                      .B_read_data_out_6_oc0 (conv6_B_data[6]),
                                      .B_read_data_out_7_oc0 (conv6_B_data[7]),
                                      .B_read_data_out_8_oc0 (conv6_B_data[8]),
                                      .B_read_data_out_9_oc0 (conv6_B_data[9]),
                                      .B_read_data_out_10_oc0 (conv6_B_data[10]),
                                      .B_read_data_out_11_oc0 (conv6_B_data[11]),
                                      .B_read_data_out_12_oc0 (conv6_B_data[12]),
                                      .B_read_data_out_13_oc0 (conv6_B_data[13]),
                                      .B_read_data_out_14_oc0 (conv6_B_data[14]),
                                      .B_read_data_out_15_oc0 (conv6_B_data[15]),
                                      .B_read_data_out_16_oc0 (conv6_B_data[16]),
                                      .B_read_data_out_17_oc0 (conv6_B_data[17]),
                                      .B_read_data_out_18_oc0 (conv6_B_data[18]),
                                      .B_read_data_out_19_oc0 (conv6_B_data[19]),
                                      .B_read_data_out_20_oc0 (conv6_B_data[20]),
                                      .B_read_data_out_21_oc0 (conv6_B_data[21]),
                                      .B_read_data_out_22_oc0 (conv6_B_data[22]),
                                      .B_read_data_out_23_oc0 (conv6_B_data[23]),
                                      .B_read_data_out_24_oc0 (conv6_B_data[24]),
                                      .B_read_data_out_25_oc0 (conv6_B_data[25]),
                                      .B_read_data_out_26_oc0 (conv6_B_data[26]),
                                      .B_read_data_out_27_oc0 (conv6_B_data[27]),
                                      .B_read_data_out_28_oc0 (conv6_B_data[28]),
                                      .B_read_data_out_29_oc0 (conv6_B_data[29]),
                                      .B_read_data_out_30_oc0 (conv6_B_data[30]),
                                      .B_read_data_out_31_oc0 (conv6_B_data[31]),
                                      .B_read_data_out_32_oc0 (conv6_B_data[32]),
                                      .B_read_data_out_33_oc0 (conv6_B_data[33]),
                                      .B_read_data_out_34_oc0 (conv6_B_data[34]),
                                      .B_read_data_out_35_oc0 (conv6_B_data[35]),
                                      .B_read_data_out_36_oc0 (conv6_B_data[36]),
                                      .B_read_data_out_37_oc0 (conv6_B_data[37]),
                                      .B_read_data_out_38_oc0 (conv6_B_data[38]),
                                      .B_read_data_out_39_oc0 (conv6_B_data[39]),
                                      .B_read_data_out_40_oc0 (conv6_B_data[40]),
                                      .B_read_data_out_41_oc0 (conv6_B_data[41]),
                                      .B_read_data_out_42_oc0 (conv6_B_data[42]),
                                      .B_read_data_out_43_oc0 (conv6_B_data[43]),
                                      .B_read_data_out_44_oc0 (conv6_B_data[44]),
                                      .B_read_data_out_45_oc0 (conv6_B_data[45]),
                                      .B_read_data_out_46_oc0 (conv6_B_data[46]),
                                      .B_read_data_out_47_oc0 (conv6_B_data[47]),
                                      .B_read_data_out_48_oc0 (conv6_B_data[48]),
                                      .B_read_data_out_49_oc0 (conv6_B_data[49]),
                                      .B_read_data_out_50_oc0 (conv6_B_data[50]),
                                      .B_read_data_out_51_oc0 (conv6_B_data[51]),
                                      .B_read_data_out_52_oc0 (conv6_B_data[52]),
                                      .B_read_data_out_53_oc0 (conv6_B_data[53]),
                                      .B_read_data_out_54_oc0 (conv6_B_data[54]),
                                      .B_read_data_out_55_oc0 (conv6_B_data[55]),
                                      .B_read_data_out_56_oc0 (conv6_B_data[56]),
                                      .B_read_data_out_57_oc0 (conv6_B_data[57]),
                                      .B_read_data_out_58_oc0 (conv6_B_data[58]),
                                      .B_read_data_out_59_oc0 (conv6_B_data[59]),
                                      .B_read_data_out_60_oc0 (conv6_B_data[60]),
                                      .B_read_data_out_61_oc0 (conv6_B_data[61]),
                                      .B_read_data_out_62_oc0 (conv6_B_data[62]),
                                      .B_read_data_out_63_oc0 (conv6_B_data[63]),
                                      .B_read_data_out_0_oc1 (conv6_B_data[64]),
                                      .B_read_data_out_1_oc1 (conv6_B_data[65]),
                                      .B_read_data_out_2_oc1 (conv6_B_data[66]),
                                      .B_read_data_out_3_oc1 (conv6_B_data[67]),
                                      .B_read_data_out_4_oc1 (conv6_B_data[68]),
                                      .B_read_data_out_5_oc1 (conv6_B_data[69]),
                                      .B_read_data_out_6_oc1 (conv6_B_data[70]),
                                      .B_read_data_out_7_oc1 (conv6_B_data[71]),
                                      .B_read_data_out_8_oc1 (conv6_B_data[72]),
                                      .B_read_data_out_9_oc1 (conv6_B_data[73]),
                                      .B_read_data_out_10_oc1 (conv6_B_data[74]),
                                      .B_read_data_out_11_oc1 (conv6_B_data[75]),
                                      .B_read_data_out_12_oc1 (conv6_B_data[76]),
                                      .B_read_data_out_13_oc1 (conv6_B_data[77]),
                                      .B_read_data_out_14_oc1 (conv6_B_data[78]),
                                      .B_read_data_out_15_oc1 (conv6_B_data[79]),
                                      .B_read_data_out_16_oc1 (conv6_B_data[80]),
                                      .B_read_data_out_17_oc1 (conv6_B_data[81]),
                                      .B_read_data_out_18_oc1 (conv6_B_data[82]),
                                      .B_read_data_out_19_oc1 (conv6_B_data[83]),
                                      .B_read_data_out_20_oc1 (conv6_B_data[84]),
                                      .B_read_data_out_21_oc1 (conv6_B_data[85]),
                                      .B_read_data_out_22_oc1 (conv6_B_data[86]),
                                      .B_read_data_out_23_oc1 (conv6_B_data[87]),
                                      .B_read_data_out_24_oc1 (conv6_B_data[88]),
                                      .B_read_data_out_25_oc1 (conv6_B_data[89]),
                                      .B_read_data_out_26_oc1 (conv6_B_data[90]),
                                      .B_read_data_out_27_oc1 (conv6_B_data[91]),
                                      .B_read_data_out_28_oc1 (conv6_B_data[92]),
                                      .B_read_data_out_29_oc1 (conv6_B_data[93]),
                                      .B_read_data_out_30_oc1 (conv6_B_data[94]),
                                      .B_read_data_out_31_oc1 (conv6_B_data[95]),
                                      .B_read_data_out_32_oc1 (conv6_B_data[96]),
                                      .B_read_data_out_33_oc1 (conv6_B_data[97]),
                                      .B_read_data_out_34_oc1 (conv6_B_data[98]),
                                      .B_read_data_out_35_oc1 (conv6_B_data[99]),
                                      .B_read_data_out_36_oc1 (conv6_B_data[100]),
                                      .B_read_data_out_37_oc1 (conv6_B_data[101]),
                                      .B_read_data_out_38_oc1 (conv6_B_data[102]),
                                      .B_read_data_out_39_oc1 (conv6_B_data[103]),
                                      .B_read_data_out_40_oc1 (conv6_B_data[104]),
                                      .B_read_data_out_41_oc1 (conv6_B_data[105]),
                                      .B_read_data_out_42_oc1 (conv6_B_data[106]),
                                      .B_read_data_out_43_oc1 (conv6_B_data[107]),
                                      .B_read_data_out_44_oc1 (conv6_B_data[108]),
                                      .B_read_data_out_45_oc1 (conv6_B_data[109]),
                                      .B_read_data_out_46_oc1 (conv6_B_data[110]),
                                      .B_read_data_out_47_oc1 (conv6_B_data[111]),
                                      .B_read_data_out_48_oc1 (conv6_B_data[112]),
                                      .B_read_data_out_49_oc1 (conv6_B_data[113]),
                                      .B_read_data_out_50_oc1 (conv6_B_data[114]),
                                      .B_read_data_out_51_oc1 (conv6_B_data[115]),
                                      .B_read_data_out_52_oc1 (conv6_B_data[116]),
                                      .B_read_data_out_53_oc1 (conv6_B_data[117]),
                                      .B_read_data_out_54_oc1 (conv6_B_data[118]),
                                      .B_read_data_out_55_oc1 (conv6_B_data[119]),
                                      .B_read_data_out_56_oc1 (conv6_B_data[120]),
                                      .B_read_data_out_57_oc1 (conv6_B_data[121]),
                                      .B_read_data_out_58_oc1 (conv6_B_data[122]),
                                      .B_read_data_out_59_oc1 (conv6_B_data[123]),
                                      .B_read_data_out_60_oc1 (conv6_B_data[124]),
                                      .B_read_data_out_61_oc1 (conv6_B_data[125]),
                                      .B_read_data_out_62_oc1 (conv6_B_data[126]),
                                      .B_read_data_out_63_oc1 (conv6_B_data[127]),
                                      .C_read_en (conv6_C_read_en),
                                      .C_read_address (conv6_C_addr),
                                      .C_read_data_out_oc0 (conv6_C_data[0]),
                                      .C_read_data_out_oc1 (conv6_C_data[1]),
                                      .RES_write_en_even (conv6_out_we0),
                                      .RES_write_address_even (conv6_out_addr0),
                                      .RES_write_data_even (conv6_out_data0),
                                      .RES_write_en_odd (conv6_out_we1),
                                      .RES_write_address_odd (conv6_out_addr1),
                                      .RES_write_data_odd (conv6_out_data1));

    // --- STAGE 10: Conv6 -> Global Average Pool ---
    wire gap_can_read, gap_done_reading, gap_A_read_en;
    wire [12:0] gap_A_addr;
    wire [15:0] gap_A_data;
    ping_pong_buffer_tdp #(
                             .DATA_WIDTH (16),
                             .DEPTH_BITS (13)
                         ) conv6_to_gap (
                             .clk                      (ACLK),
                             .rst                      (~ARESETN),
                             .producer_write_en_even   (conv6_out_we0),
                             .producer_write_addr_even (conv6_out_addr0),
                             .producer_write_data_even (conv6_out_data0),
                             .producer_write_en_odd    (conv6_out_we1),
                             .producer_write_addr_odd  (conv6_out_addr1),
                             .producer_write_data_odd  (conv6_out_data1),
                             .producer_frame_done      (conv6_done_writing),
                             .producer_ready           (conv6_can_write),
                             .consumer_read_en         (gap_A_read_en),
                             .consumer_read_addr       (gap_A_addr),
                             .consumer_read_data       (gap_A_data),
                             .consumer_frame_done      (gap_done_reading),
                             .consumer_ready           (gap_can_read)
                         );
    wire gap_can_write, gap_done_writing, gap_we;
    wire [6:0] gap_waddr;
    wire [15:0] gap_wdata;
    avg_pool_ping_pong #(
                           .C_DIM          (128),
                           .IN_H           (8),
                           .IN_W           (8),
                           .A_depth_bits   (13),
                           .RES_depth_bits (7)
                       ) gap (
                           .clk                 (ACLK),
                           .rst                 (ARESETN),
                           .consumer_ready      (gap_can_read),
                           .producer_ready      (gap_can_write),
                           .consumer_frame_done (gap_done_reading),
                           .producer_frame_done (gap_done_writing),
                           .A_read_en           (gap_A_read_en),
                           .A_read_address      (gap_A_addr),
                           .A_read_data_out     (gap_A_data),
                           .RES_write_en        (gap_we),
                           .RES_write_address   (gap_waddr),
                           .RES_write_data_in   (gap_wdata)
                       );
    // --- STAGE 11: Global Average Pool -> FC ---
    // Two independent 128x16 frame banks use asynchronous distributed RAM.
    // The FC core delays this address until its BRAM weight reaches Multiply.
    wire fc_can_read, fc_done_reading, fc_A_read_en;
    wire [6:0] fc_A_addr;
    wire [15:0] fc_A_data;
    ping_pong_buffer #(
                         .DATA_WIDTH (16),
                         .DEPTH_BITS (7),
                         .MEM_TYPE   ("LUTRAM")
                     ) gap_to_fc (
                         .clk                 (ACLK),
                         .rst                 (~ARESETN),
                         .producer_write_en   (gap_we),
                         .producer_write_addr (gap_waddr),
                         .producer_write_data (gap_wdata),
                         .producer_frame_done (gap_done_writing),
                         .producer_ready      (gap_can_write),
                         .consumer_read_en    (fc_A_read_en),
                         .consumer_read_addr  (fc_A_addr),
                         .consumer_read_data  (fc_A_data),
                         .consumer_frame_done (fc_done_reading),
                         .consumer_ready      (fc_can_read)
                     );
    wire fc_can_write, fc_done_writing, fc_we;
    wire [0:0] fc_waddr;
    wire [15:0] fc_wdata;
    fully_connected_ping_pong #(
                                  .IN_F           (128),
                                  .OUT_F          (10),
                                  .A_depth_bits   (7),
                                  .B_depth_bits   (11),
                                  .C_depth_bits   (4),
                                  .RES_depth_bits (1)
                              ) fc (
                                  .clk                 (ACLK),
                                  .rst                 (ARESETN),
                                  .consumer_ready      (fc_can_read),
                                  .producer_ready      (fc_can_write),
                                  .consumer_frame_done (fc_done_reading),
                                  .producer_frame_done (fc_done_writing),
                                  .A_read_en           (fc_A_read_en),
                                  .A_read_address      (fc_A_addr),
                                  .A_read_data_out     (fc_A_data),
                                  .B_read_en           (fc_B_read_en),
                                  .B_read_address      (fc_B_addr),
                                  .B_read_data_out     (fc_B_data),
                                  .C_read_en           (fc_C_read_en),
                                  .C_read_address      (fc_C_addr),
                                  .C_read_data_out     (fc_C_data),
                                  .RES_write_en        (fc_we),
                                  .RES_write_address   (fc_waddr),
                                  .RES_write_data_in   (fc_wdata)
                              );
    // --- STAGE 12: FC -> Transmitter ---
    // Each frame reserves two LUTRAM words; the fused FC/argmax writes word 0.
    wire tx_can_read, tx_done_reading, tx_read_en;
    wire [3:0] tx_read_addr;
    wire [15:0] tx_read_data;
    ping_pong_buffer #(
                         .DATA_WIDTH (16),
                         .DEPTH_BITS (1),
                         .MEM_TYPE   ("LUTRAM")
                     ) fc_to_tx (
                         .clk                 (ACLK),
                         .rst                 (~ARESETN),
                         .producer_write_en   (fc_we),
                         .producer_write_addr (fc_waddr),
                         .producer_write_data (fc_wdata),
                         .producer_frame_done (fc_done_writing),
                         .producer_ready      (fc_can_write),
                         .consumer_read_en    (tx_read_en),
                         .consumer_read_addr  (tx_read_addr[0:0]),
                         .consumer_read_data  (tx_read_data),
                         .consumer_frame_done (tx_done_reading),
                         .consumer_ready      (tx_can_read)
                     );
    axi_transmitter #(
                        .NUM_OUTPUT_WORDS (1),
                        .NUM_IMAGES       (NUM_IMAGES),
                        .LUTRAM_INPUT     (1)
                    ) tx (
                        .clk           (ACLK),
                        .rst_n         (ARESETN),
                        .buffer_ready  (tx_can_read),
                        .read_en       (tx_read_en),
                        .read_addr     (tx_read_addr),
                        .read_data     (tx_read_data),
                        .frame_done    (tx_done_reading),
                        .batch_done    (tx_batch_done),
                        .m_axis_tready (M_AXIS_TREADY),
                        .m_axis_tdata  (M_AXIS_TDATA),
                        .m_axis_tvalid (M_AXIS_TVALID),
                        .m_axis_tlast  (M_AXIS_TLAST)
                    );
endmodule

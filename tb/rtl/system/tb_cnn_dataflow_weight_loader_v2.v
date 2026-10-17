`timescale 1ns / 1ps

module tb_cnn_dataflow_weight_loader_v2;
    localparam TOTAL=288298;
    reg clk=0,rst_n=0;
    wire s_ready;
    reg[15:0]s_data=0;
    reg s_last=0,s_valid=0;
    wire m_valid,m_last;
    wire[15:0]m_data;
    reg[15:0]parameter_words[0:TOTAL-1];
    integer index,errors;

    cnn_dataflow#(.NUM_IMAGES(1))dut(
                    .ACLK(clk),.ARESETN(rst_n),.S_AXIS_TREADY(s_ready),
                    .S_AXIS_TDATA(s_data),.S_AXIS_TLAST(s_last),.S_AXIS_TVALID(s_valid),
                    .M_AXIS_TVALID(m_valid),.M_AXIS_TDATA(m_data),.M_AXIS_TLAST(m_last),
                    .M_AXIS_TREADY(1'b1));
    always #5 clk=~clk;

    // Inject a selected stream position through the real LOAD_WEIGHTS decode
    // and its one-cycle write-data/enable/address pipeline.
    task load_selected_word;
        input integer stream_index;
        begin
            @(negedge clk);
            dut.state=3'b010;
            dut.read_counter=stream_index;
            s_data=parameter_words[stream_index];
            s_valid=1;
            @(posedge clk);
            @(negedge clk);
            s_valid=0;
            @(posedge clk);
            @(negedge clk);
        end
    endtask

    initial begin
        // Exact parameter order consumed by cnn_accelerator.cpp.
        $readmemh("tb/data/conv1_w.mem",parameter_words,0,863);
        $readmemh("tb/data/conv2_w.mem",parameter_words,864,10079);
        $readmemh("tb/data/conv3_w.mem",parameter_words,10080,28511);
        $readmemh("tb/data/conv4_w.mem",parameter_words,28512,65375);
        $readmemh("tb/data/conv5_w.mem",parameter_words,65376,139103);
        $readmemh("tb/data/conv6_w.mem",parameter_words,139104,286559);
        $readmemh("tb/data/fc_w.mem",parameter_words,286560,287839);
        $readmemh("tb/data/conv1_b.mem",parameter_words,287840,287871);
        $readmemh("tb/data/conv2_b.mem",parameter_words,287872,287903);
        $readmemh("tb/data/conv3_b.mem",parameter_words,287904,287967);
        $readmemh("tb/data/conv4_b.mem",parameter_words,287968,288031);
        $readmemh("tb/data/conv5_b.mem",parameter_words,288032,288159);
        $readmemh("tb/data/conv6_b.mem",parameter_words,288160,288287);
        $readmemh("tb/data/fc_b.mem",parameter_words,288288,288297);

        errors=0;
        repeat(4)@(posedge clk);
        rst_n<=1;

        dut.W1_chunk_cnt=0;
        dut.W1_ram_sel=0;
        dut.W1_base_addr=0;
        load_selected_word(0);
        // Conv1 groups logical LUTRAM banks 0..3 behind one packed interface.
        dut.W1_chunk_cnt=0;
        dut.W1_ram_sel=1;
        dut.W1_base_addr=0;
        load_selected_word(9);
        dut.W1_chunk_cnt=0;
        dut.W1_ram_sel=2;
        dut.W1_base_addr=0;
        load_selected_word(18);
        dut.W1_chunk_cnt=0;
        dut.W1_ram_sel=3;
        dut.W1_base_addr=0;
        load_selected_word(27);
        dut.W1_chunk_cnt=8;
        dut.W1_ram_sel=11;
        dut.W1_base_addr=63;
        load_selected_word(863);
        dut.W2_chunk_cnt=0;
        dut.W2_ram_sel=0;
        dut.W2_base_addr=0;
        load_selected_word(864);
        dut.W2_chunk_cnt=0;
        dut.W2_ram_sel=32;
        dut.W2_base_addr=0;
        load_selected_word(1152);
        dut.W2_chunk_cnt=0;
        dut.W2_ram_sel=64;
        dut.W2_base_addr=0;
        load_selected_word(1440);
        dut.W2_chunk_cnt=0;
        dut.W2_ram_sel=96;
        dut.W2_base_addr=0;
        load_selected_word(1728);
        dut.W2_chunk_cnt=8;
        dut.W2_ram_sel=127;
        dut.W2_base_addr=63;
        load_selected_word(10079);
        // Conv3 OU2 packs logical banks 0/16/32/48 into physical BRAM 0.
        // Each logical bank is declared 512x16 and uses 288 addresses.
        dut.W3_chunk_cnt=0;
        dut.W3_ram_sel=0;
        dut.W3_base_addr=0;
        load_selected_word(10080);
        dut.W3_chunk_cnt=0;
        dut.W3_ram_sel=16;
        dut.W3_base_addr=0;
        load_selected_word(10224);
        dut.W3_chunk_cnt=0;
        dut.W3_ram_sel=32;
        dut.W3_base_addr=0;
        load_selected_word(10368);
        dut.W3_chunk_cnt=0;
        dut.W3_ram_sel=48;
        dut.W3_base_addr=0;
        load_selected_word(10512);
        dut.W3_chunk_cnt=8;
        dut.W3_ram_sel=63;
        dut.W3_base_addr=279;
        load_selected_word(28511);
        // Conv4 OU2 packs logical banks 0/32/64/96 into physical BRAM 0.
        // Each logical bank is declared 512x16 and uses 288 addresses.
        dut.W4_chunk_cnt=0;
        dut.W4_ram_sel=0;
        dut.W4_base_addr=0;
        load_selected_word(28512);
        dut.W4_chunk_cnt=0;
        dut.W4_ram_sel=32;
        dut.W4_base_addr=0;
        load_selected_word(28800);
        dut.W4_chunk_cnt=0;
        dut.W4_ram_sel=64;
        dut.W4_base_addr=0;
        load_selected_word(29088);
        dut.W4_chunk_cnt=0;
        dut.W4_ram_sel=96;
        dut.W4_base_addr=0;
        load_selected_word(29376);
        dut.W4_chunk_cnt=8;
        dut.W4_ram_sel=127;
        dut.W4_base_addr=279;
        load_selected_word(65375);
        // Conv5 URAM 0 contains logical banks 0..7 in eight 18-bit slots.
        dut.W5_chunk_cnt=0;
        dut.W5_ram_sel=0;
        dut.W5_base_addr=0;
        load_selected_word(65376);
        dut.W5_chunk_cnt=0;
        dut.W5_ram_sel=1;
        dut.W5_base_addr=0;
        load_selected_word(65385);
        dut.W5_chunk_cnt=0;
        dut.W5_ram_sel=2;
        dut.W5_base_addr=0;
        load_selected_word(65394);
        dut.W5_chunk_cnt=0;
        dut.W5_ram_sel=3;
        dut.W5_base_addr=0;
        load_selected_word(65403);
        dut.W5_chunk_cnt=0;
        dut.W5_ram_sel=4;
        dut.W5_base_addr=0;
        load_selected_word(65412);
        dut.W5_chunk_cnt=0;
        dut.W5_ram_sel=5;
        dut.W5_base_addr=0;
        load_selected_word(65421);
        dut.W5_chunk_cnt=0;
        dut.W5_ram_sel=6;
        dut.W5_base_addr=0;
        load_selected_word(65430);
        dut.W5_chunk_cnt=0;
        dut.W5_ram_sel=7;
        dut.W5_base_addr=0;
        load_selected_word(65439);
        // Exercise the final packed URAM and the full 1152-word bank depth.
        dut.W5_chunk_cnt=0;
        dut.W5_ram_sel=56;
        dut.W5_base_addr=0;
        load_selected_word(65880);
        dut.W5_chunk_cnt=0;
        dut.W5_ram_sel=63;
        dut.W5_base_addr=0;
        load_selected_word(65943);
        dut.W5_chunk_cnt=8;
        dut.W5_ram_sel=63;
        dut.W5_base_addr=1143;
        load_selected_word(139103);

        dut.W6_chunk_cnt=0;
        dut.W6_ic=0;
        dut.W6_lane=0;
        dut.W6_base_addr=0;
        load_selected_word(139104);
        dut.W6_chunk_cnt=0;
        dut.W6_ic=1;
        dut.W6_lane=0;
        dut.W6_base_addr=0;
        load_selected_word(139113);
        dut.W6_chunk_cnt=0;
        dut.W6_ic=2;
        dut.W6_lane=0;
        dut.W6_base_addr=0;
        load_selected_word(139122);
        dut.W6_chunk_cnt=0;
        dut.W6_ic=3;
        dut.W6_lane=0;
        dut.W6_base_addr=0;
        load_selected_word(139131);
        dut.W6_chunk_cnt=0;
        dut.W6_ic=4;
        dut.W6_lane=0;
        dut.W6_base_addr=0;
        load_selected_word(139140);
        dut.W6_chunk_cnt=0;
        dut.W6_ic=5;
        dut.W6_lane=0;
        dut.W6_base_addr=0;
        load_selected_word(139149);
        dut.W6_chunk_cnt=0;
        dut.W6_ic=6;
        dut.W6_lane=0;
        dut.W6_base_addr=0;
        load_selected_word(139158);
        dut.W6_chunk_cnt=0;
        dut.W6_ic=7;
        dut.W6_lane=0;
        dut.W6_base_addr=0;
        load_selected_word(139167);
        dut.W6_chunk_cnt=0;
        dut.W6_ic=64;
        dut.W6_lane=0;
        dut.W6_base_addr=0;
        load_selected_word(139680);
        dut.W6_chunk_cnt=0;
        dut.W6_ic=0;
        dut.W6_lane=1;
        dut.W6_base_addr=0;
        load_selected_word(140256);
        dut.W6_chunk_cnt=0;
        dut.W6_ic=0;
        dut.W6_lane=0;
        dut.W6_base_addr=18;
        load_selected_word(141408);
        dut.W6_chunk_cnt=8;
        dut.W6_ic=127;
        dut.W6_lane=1;
        dut.W6_base_addr=1134;
        load_selected_word(286559);

        load_selected_word(286560);
        load_selected_word(287839);
        load_selected_word(287871);
        load_selected_word(287903);
        load_selected_word(287967);
        load_selected_word(288031);
        load_selected_word(288159);
        load_selected_word(288287);
        load_selected_word(288297);
        repeat(2)@(posedge clk);

        // Conv1: all four LUTRAM banks and the final lane/group/kernel word.
        if(dut.CONV1_WEIGHT_MEMORIES[0].weight_ram.RAM0[0]!==parameter_words[0])
            errors=errors+1;
        if(dut.CONV1_WEIGHT_MEMORIES[0].weight_ram.RAM1[0]!==parameter_words[9])
            errors=errors+1;
        if(dut.CONV1_WEIGHT_MEMORIES[0].weight_ram.RAM2[0]!==parameter_words[18])
            errors=errors+1;
        if(dut.CONV1_WEIGHT_MEMORIES[0].weight_ram.RAM3[0]!==parameter_words[27])
            errors=errors+1;
        if(dut.CONV1_WEIGHT_MEMORIES[2].weight_ram.RAM3[71]!==parameter_words[863])
            errors=errors+1;
        // Conv2: all four packed slices and the final lane/group/kernel word.
        if(dut.CONV2_WEIGHT_MEMORIES[0].weight_ram.RAM[0][15:0]!==parameter_words[864])
            errors=errors+1;
        if(dut.CONV2_WEIGHT_MEMORIES[0].weight_ram.RAM[0][31:16]!==parameter_words[1152])
            errors=errors+1;
        if(dut.CONV2_WEIGHT_MEMORIES[0].weight_ram.RAM[128][15:0]!==parameter_words[1440])
            errors=errors+1;
        if(dut.CONV2_WEIGHT_MEMORIES[0].weight_ram.RAM[128][31:16]!==parameter_words[1728])
            errors=errors+1;
        if(dut.CONV2_WEIGHT_MEMORIES[31].weight_ram.RAM[199][31:16]!==parameter_words[10079])
            errors=errors+1;
        // Conv3/4/5 end points exercise their full output-bank rotations.
        if(dut.CONV3_WEIGHT_MEMORIES[0].weight_ram.RAM[0][15:0]!==parameter_words[10080])
            errors=errors+1;
        if(dut.CONV3_WEIGHT_MEMORIES[0].weight_ram.RAM[0][31:16]!==parameter_words[10224])
            errors=errors+1;
        if(dut.CONV3_WEIGHT_MEMORIES[0].weight_ram.RAM[512][15:0]!==parameter_words[10368])
            errors=errors+1;
        if(dut.CONV3_WEIGHT_MEMORIES[0].weight_ram.RAM[512][31:16]!==parameter_words[10512])
            errors=errors+1;
        if(dut.CONV3_WEIGHT_MEMORIES[15].weight_ram.RAM[799][31:16]!==parameter_words[28511])
            errors=errors+1;
        if(dut.CONV4_WEIGHT_MEMORIES[0].weight_ram.RAM[0][15:0]!==parameter_words[28512])
            errors=errors+1;
        if(dut.CONV4_WEIGHT_MEMORIES[0].weight_ram.RAM[0][31:16]!==parameter_words[28800])
            errors=errors+1;
        if(dut.CONV4_WEIGHT_MEMORIES[0].weight_ram.RAM[512][15:0]!==parameter_words[29088])
            errors=errors+1;
        if(dut.CONV4_WEIGHT_MEMORIES[0].weight_ram.RAM[512][31:16]!==parameter_words[29376])
            errors=errors+1;
        if(dut.CONV4_WEIGHT_MEMORIES[31].weight_ram.RAM[799][31:16]!==parameter_words[65375])
            errors=errors+1;
        // XPM UltraRAM contents are opaque to portable hierarchical checks;
        // selected writes above cover all eight lanes of physical URAM 0,
        // physical URAM 7, and the final logical address (1151).
        if(dut.FC_WEIGHT_RAM.RAM[0]!==parameter_words[286560])
            errors=errors+1;
        if(dut.FC_WEIGHT_RAM.RAM[1279]!==parameter_words[287839])
            errors=errors+1;

        // Bias banking follows output-channel unrolling for every convolution.
        if(dut.CONV1_BIAS_MEMORIES[3].bias_ram.RAM[7]!==parameter_words[287871])
            errors=errors+1;
        if(dut.CONV2_BIAS_MEMORIES[3].bias_ram.RAM[7]!==parameter_words[287903])
            errors=errors+1;
        if(dut.CONV3_BIAS_MEMORIES[1].bias_ram.RAM[31]!==parameter_words[287967])
            errors=errors+1;
        if(dut.CONV4_BIAS_MEMORIES[1].bias_ram.RAM[31]!==parameter_words[288031])
            errors=errors+1;
        if(dut.CONV5_BIAS_MEMORY.RAM[127]!==parameter_words[288159])
            errors=errors+1;
        if(dut.CONV6_BIAS_MEMORIES[1].bias_ram.RAM[63]!==parameter_words[288287])
            errors=errors+1;
        if(dut.FC_BIAS_RAM.RAM[9]!==parameter_words[288297])
            errors=errors+1;
        if(!dut.weights_loaded)
            errors=errors+1;

        if(errors==0)
            $display("PASS: all Model_version2 parameter groups routed to the expected memories.");
        else
            $display("FAIL: parameter loader routing errors=%0d",errors);
        $finish;
    end
    initial begin
        #5000000;
        $display("FAIL: parameter loader timeout.");
        $finish;
    end
endmodule

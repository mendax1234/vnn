`timescale 1ns / 1ps

module tb_myip_conv2_relu_unrolled ();
    localparam width = 16;
    // 32x32x32 activations + 32x32x3x3 weights + 32 biases.
    localparam NUMBER_OF_INPUT_WORDS = 42016;
    localparam NUMBER_OF_OUTPUT_WORDS = 32768;

    reg ACLK = 0, ARESETN = 0;
    wire S_AXIS_TREADY;
    reg [31:0] S_AXIS_TDATA = 0;
    reg S_AXIS_TLAST = 0, S_AXIS_TVALID = 0;
    wire M_AXIS_TVALID;
    wire [31:0] M_AXIS_TDATA;
    wire M_AXIS_TLAST;
    reg M_AXIS_TREADY = 0;

    reg [width-1:0] test_input_memory [0:NUMBER_OF_INPUT_WORDS-1];
    reg [width-1:0] test_result_expected_memory [0:NUMBER_OF_OUTPUT_WORDS-1];
    reg [width-1:0] result_memory [0:NUMBER_OF_OUTPUT_WORDS-1];
    integer word_cnt;
    integer mismatch_count = 0;
    integer tdp_write_count = 0;
    integer serializer_phase = 0;
    reg [14:0] serializer_base_address;

    myip_conv2_relu_unrolled DUT (
                                 .ACLK          (ACLK),
                                 .ARESETN       (ARESETN),
                                 .S_AXIS_TREADY (S_AXIS_TREADY),
                                 .S_AXIS_TDATA  (S_AXIS_TDATA),
                                 .S_AXIS_TLAST  (S_AXIS_TLAST),
                                 .S_AXIS_TVALID (S_AXIS_TVALID),
                                 .M_AXIS_TVALID (M_AXIS_TVALID),
                                 .M_AXIS_TDATA  (M_AXIS_TDATA),
                                 .M_AXIS_TLAST  (M_AXIS_TLAST),
                                 .M_AXIS_TREADY (M_AXIS_TREADY)
                             );
    always #5 ACLK = ~ACLK;

    // Output-unroll 4 is serialized into two consecutive two-port writes:
    // channels {0,1}, followed by channels {2,3}.
    always @(posedge ACLK) begin
        if (ARESETN && (DUT.RES_write_en_even || DUT.RES_write_en_odd)) begin
            if (!(DUT.RES_write_en_even && DUT.RES_write_en_odd)) begin
                $display("TDP write-enable mismatch at word %0d", tdp_write_count);
                mismatch_count = mismatch_count + 1;
            end
            if (DUT.RES_write_address_odd != DUT.RES_write_address_even + 1024) begin
                $display("TDP pair address mismatch at word %0d", tdp_write_count);
                mismatch_count = mismatch_count + 1;
            end
            if (serializer_phase == 0)
                serializer_base_address = DUT.RES_write_address_even;
            else if (DUT.RES_write_address_even !=
                     serializer_base_address + serializer_phase * 2048) begin
                $display("Conv2 serializer phase/address mismatch at word %0d", tdp_write_count);
                mismatch_count = mismatch_count + 1;
            end
            serializer_phase = (serializer_phase + 1) & 1;
            tdp_write_count = tdp_write_count + 2;
        end
    end

    initial begin
        $display("Loading fused Conv2 + ReLU vectors (IC unroll 32, OC unroll 4)...");
        $readmemh("tb/data/conv_relu/conv2/test_input_conv2_relu.mem", test_input_memory);
        $readmemh("tb/data/conv_relu/conv2/test_result_expected_conv2_relu.mem", test_result_expected_memory);
        repeat (4) @(posedge ACLK);
        ARESETN <= 1'b1;

        word_cnt = 0;
        while (word_cnt < NUMBER_OF_INPUT_WORDS) begin
            @(negedge ACLK);
            S_AXIS_TVALID = 1'b1;
            S_AXIS_TDATA = {16'd0, test_input_memory[word_cnt]};
            S_AXIS_TLAST = (word_cnt == NUMBER_OF_INPUT_WORDS - 1);
            if (S_AXIS_TREADY)
                word_cnt = word_cnt + 1;
        end
        @(negedge ACLK);
        S_AXIS_TVALID = 1'b0;
        S_AXIS_TLAST = 1'b0;

        M_AXIS_TREADY = 1'b1;
        word_cnt = 0;
        while (word_cnt < NUMBER_OF_OUTPUT_WORDS) begin
            @(posedge ACLK);
            if (M_AXIS_TVALID && M_AXIS_TREADY) begin
                result_memory[word_cnt] = M_AXIS_TDATA[width-1:0];
                if ((word_cnt == NUMBER_OF_OUTPUT_WORDS - 1) != M_AXIS_TLAST) begin
                    $display("TLAST alignment error at output %0d", word_cnt);
                    mismatch_count = mismatch_count + 1;
                end
                word_cnt = word_cnt + 1;
            end
        end
        M_AXIS_TREADY = 1'b0;

        if (tdp_write_count != NUMBER_OF_OUTPUT_WORDS || serializer_phase != 0) begin
            $display("TDP serializer count/phase error: words=%0d phase=%0d",
                     tdp_write_count, serializer_phase);
            mismatch_count = mismatch_count + 1;
        end

        for (word_cnt = 0; word_cnt < NUMBER_OF_OUTPUT_WORDS; word_cnt = word_cnt + 1)
            if (result_memory[word_cnt] !== test_result_expected_memory[word_cnt]) begin
                if (mismatch_count < 20)
                    $display("Mismatch at %0d: expected %h, got %h", word_cnt,
                             test_result_expected_memory[word_cnt], result_memory[word_cnt]);
                mismatch_count = mismatch_count + 1;
            end
        if (mismatch_count == 0)
            $display("PASS: fused Conv2 + ReLU produced all %0d expected words.", NUMBER_OF_OUTPUT_WORDS);
        else
            $display("FAIL: fused Conv2 + ReLU had %0d mismatches.", mismatch_count);
        $finish;
    end

    initial begin
        #10000000;
        $display("FAIL: simulation timeout.");
        $finish;
    end

endmodule

`timescale 1ns / 1ps

module tb_myip_conv5_relu_unrolled ();
    localparam width = 16;
    localparam NUMBER_OF_INPUT_WORDS = 64 * 8 * 8
               + 128 * 64 * 3 * 3
               + 128; // 77952
    localparam NUMBER_OF_OUTPUT_WORDS = 128 * 8 * 8; // 8192

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
    integer output_write_count = 0;

    myip_conv5_relu_unrolled DUT (
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

    // OC-unroll 1 writes one channel-major result per cycle.
    always @(posedge ACLK) begin
        if (ARESETN && DUT.RES_write_en)
            output_write_count = output_write_count + 1;
    end

    initial begin
        $display("Loading fused Conv5 + ReLU vectors (OC unroll 1)...");
        $readmemh("tb/data/conv_relu/conv5/test_input_conv5_relu.mem", test_input_memory);
        $readmemh("tb/data/conv_relu/conv5/test_result_expected_conv5_relu.mem",
                  test_result_expected_memory);
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

        if (output_write_count != NUMBER_OF_OUTPUT_WORDS) begin
            $display("Output write count error: words=%0d", output_write_count);
            mismatch_count = mismatch_count + 1;
        end

        for (word_cnt = 0; word_cnt < NUMBER_OF_OUTPUT_WORDS;
                word_cnt = word_cnt + 1)
            if (result_memory[word_cnt]
                    !== test_result_expected_memory[word_cnt]) begin
                if (mismatch_count < 20)
                    $display("Mismatch at %0d: expected %h, got %h", word_cnt,
                             test_result_expected_memory[word_cnt],
                             result_memory[word_cnt]);
                mismatch_count = mismatch_count + 1;
            end

        if (mismatch_count == 0)
            $display("PASS: fused Conv5 + ReLU produced all %0d expected words.",
                     NUMBER_OF_OUTPUT_WORDS);
        else
            $display("FAIL: fused Conv5 + ReLU had %0d mismatches.",
                     mismatch_count);
        $finish;
    end

    initial begin
        #20000000;
        $display("FAIL: simulation timeout.");
        $finish;
    end
endmodule

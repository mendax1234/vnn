`timescale 1ns / 1ps

module tb_cnn_dataflow ();

    reg         ACLK = 0;
    reg         ARESETN;

    // slave in interface
    wire        S_AXIS_TREADY;
    reg  [15:0] S_AXIS_TDATA;
    reg         S_AXIS_TLAST;
    reg         S_AXIS_TVALID;

    // master out interface
    wire        M_AXIS_TVALID;
    wire [15:0] M_AXIS_TDATA;
    wire        M_AXIS_TLAST;
    reg         M_AXIS_TREADY;

    // Memory Parameters
    localparam TOTAL_WEIGHTS          = 288298;
    localparam TOTAL_IMAGE            = 3072;
    localparam NUMBER_OF_TEST_VECTORS = 10;
    localparam TOTAL_INPUTS           = TOTAL_WEIGHTS + (TOTAL_IMAGE * NUMBER_OF_TEST_VECTORS);

    // Instantiate Top-Level CNN Wrapper
    cnn_dataflow #(
                     .NUM_IMAGES(NUMBER_OF_TEST_VECTORS) // Pass the batch size!
                 ) U1 (
                     .ACLK         (ACLK),
                     .ARESETN      (ARESETN),
                     .S_AXIS_TREADY(S_AXIS_TREADY),
                     .S_AXIS_TDATA (S_AXIS_TDATA),
                     .S_AXIS_TLAST (S_AXIS_TLAST),
                     .S_AXIS_TVALID(S_AXIS_TVALID),
                     .M_AXIS_TVALID(M_AXIS_TVALID),
                     .M_AXIS_TDATA (M_AXIS_TDATA),
                     .M_AXIS_TLAST (M_AXIS_TLAST),
                     .M_AXIS_TREADY(M_AXIS_TREADY)
                 );

    localparam width = 16;

    reg [width-1:0] test_input_memory [0:TOTAL_INPUTS-1];
    reg [width-1:0] test_result_expected_memory [0:NUMBER_OF_TEST_VECTORS-1];
    reg [width-1:0] result_memory [0:NUMBER_OF_TEST_VECTORS-1];

    integer word_cnt, test_case_cnt;
    reg success = 1'b1;
    reg simulation_done = 1'b0;

    always #50 ACLK = ~ACLK; // 10MHz Clock

    // Prevent "run all" from spinning forever if a pipeline handshake stalls.
    initial begin
        #1000000000; // 1 simulated second = 10,000,000 clocks
        if (!simulation_done) begin
            $display("TIMEOUT: no complete 10-image result batch was produced.");
            $finish;
        end
    end

    initial begin
        $display("Loading Memory into Testbench...");
        $fflush();
        // Paths are relative to the repository root. For a Vivado GUI project,
        // set these to the absolute locations of the supplied vector files.
        $readmemh("tb/data/system/test_input_10.mem", test_input_memory);
        $readmemh("tb/data/system/test_result_expected_10.mem", test_result_expected_memory);

        ARESETN = 1'b0;
        S_AXIS_TVALID = 1'b0;
        S_AXIS_TDATA = 16'b0;
        S_AXIS_TLAST = 1'b0;
        M_AXIS_TREADY = 1'b0;

        repeat (3) @(posedge ACLK);
        @(negedge ACLK);
        ARESETN = 1'b1;

        // ==========================================
        // PHASE 1: Load Weights & Biases
        // ==========================================
        $display("Streaming %0d weights into FPGA...", TOTAL_WEIGHTS);
        word_cnt = 0;
        while (word_cnt < TOTAL_WEIGHTS) begin
            @(negedge ACLK);
            S_AXIS_TVALID = 1'b1;
            S_AXIS_TDATA = test_input_memory[word_cnt];
            S_AXIS_TLAST = (word_cnt == TOTAL_WEIGHTS - 1);
            @(posedge ACLK);
            if (S_AXIS_TREADY) begin
                word_cnt = word_cnt + 1;
                if ((word_cnt % 50000) == 0) begin
                    $display("[%0t] Loaded %0d/%0d parameter words.",
                             $time, word_cnt, TOTAL_WEIGHTS);
                    $fflush();
                end
            end
        end
        @(negedge ACLK);
        S_AXIS_TVALID = 1'b0;
        S_AXIS_TLAST  = 1'b0;

        $display("Weights loaded! Transitioning to RUN_DATAFLOW state...");
        $fflush();
        repeat (2) @(posedge ACLK);

        // ==========================================
        // PHASE 2: Stream Images and Read Output
        // ==========================================
        $display("Starting Dataflow Pipeline. Streaming %0d images...", NUMBER_OF_TEST_VECTORS);
        fork
            // ---------------------------------------------------------
            // THREAD 1: Send Images (The Producer)
            // ---------------------------------------------------------
            begin: send_thread
                for (test_case_cnt = 0; test_case_cnt < NUMBER_OF_TEST_VECTORS; test_case_cnt = test_case_cnt + 1) begin
                    word_cnt = 0;
                    while (word_cnt < TOTAL_IMAGE) begin
                        @(negedge ACLK);
                        S_AXIS_TVALID = 1'b1;
                        S_AXIS_TDATA = test_input_memory[TOTAL_WEIGHTS + (test_case_cnt * TOTAL_IMAGE) + word_cnt];
                        S_AXIS_TLAST = (word_cnt == TOTAL_IMAGE - 1);
                        @(posedge ACLK);
                        if (S_AXIS_TREADY)
                            word_cnt = word_cnt + 1;
                    end
                    @(negedge ACLK);
                    S_AXIS_TVALID = 1'b0;
                    S_AXIS_TLAST  = 1'b0;
                    $display("[%0t] Finished sending Image %0d into pipeline.", $time, test_case_cnt);
                    $fflush();
                    repeat (2) @(posedge ACLK); // Small gap between images
                end
            end

            // ---------------------------------------------------------
            // THREAD 2: Receive Images (The Consumer)
            // ---------------------------------------------------------
            begin: receive_thread
                integer out_cnt;
                M_AXIS_TREADY = 1'b1;

                for (out_cnt = 0; out_cnt < NUMBER_OF_TEST_VECTORS; out_cnt = out_cnt + 1) begin

                    if (out_cnt == 0)
                        $display("[%0t] Waiting for Image 0 to traverse the Model_version2 pipeline...", $time);

                    // Wait block: hold until a valid handshake occurs
                    @(posedge ACLK);
                    while (!(M_AXIS_TVALID && M_AXIS_TREADY))
                        @(posedge ACLK);

                    // Capture the SINGLE Argmax result
                    result_memory[out_cnt] = M_AXIS_TDATA;
                    $display("[%0t] Received result for Image %0d: Class %0d", $time, out_cnt, M_AXIS_TDATA);
                    $fflush();
                    if (M_AXIS_TLAST !== (out_cnt == NUMBER_OF_TEST_VECTORS - 1)) begin
                        success = 1'b0;
                        $display("FAILURE: TLAST mismatch on output %0d.", out_cnt);
                    end

                end
                M_AXIS_TREADY = 1'b0;
            end
        join

        // ==========================================
        // VERIFICATION
        // ==========================================
        $display("-----------------------------------------");
        for (test_case_cnt = 0; test_case_cnt < NUMBER_OF_TEST_VECTORS; test_case_cnt = test_case_cnt + 1) begin
            if (result_memory[test_case_cnt] !== test_result_expected_memory[test_case_cnt])
                success = 1'b0;
            $display("Image %0d -> Expected: %04X, Got: %04X", test_case_cnt, test_result_expected_memory[test_case_cnt], result_memory[test_case_cnt]);
        end

        if (success)
            $display("SUCCESS: Test Passed for all %0d images!", NUMBER_OF_TEST_VECTORS);
        else
            $display("FAILURE: Test Failed. Outputs do not match expected results.");
        $display("-----------------------------------------");

        simulation_done = 1'b1;
        $finish;
    end

endmodule

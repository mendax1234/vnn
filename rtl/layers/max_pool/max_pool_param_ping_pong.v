`timescale 1ns / 1ps

/* ----------------------------------------------------------------------------------
--  Parameterized RTL implementation of 2x2, stride-2 Max Pool
--  Reads from one channel-major, three-cycle synchronous memory and presents
--  each result with its channel index for a packed downstream memory
----------------------------------------------------------------------------------
*/

module max_pool_param_ping_pong #(
        parameter width          = 16,

        // Input dimensions
        parameter IN_W           = 32,
        parameter IN_H           = 32,
        parameter C_DIM          = 16,

        parameter A_depth_bits   = $clog2(C_DIM * IN_H * IN_W),
        parameter RES_depth_bits = $clog2((IN_W >> 1) * (IN_H >> 1))
    )(
        input      clk,
        input      rst,
        input      consumer_ready,      // HIGH when upstream Input Buffer is full
        input      producer_ready,      // HIGH when downstream Output Buffer has an empty bank
        output reg consumer_frame_done, // Pulses HIGH to tell upstream we finished reading
        output reg producer_frame_done, // Pulses HIGH to tell downstream we finished writing

        // RAM A (Input - Single Port, 3-cycle synchronous read)
        output reg                    A_read_en,
        output     [A_depth_bits-1:0] A_read_address,
        input signed [     width-1:0] A_read_data_out,

        // RAM RES (Output - one result and its packed-memory channel per cycle)
        output reg [     C_DIM-1:0]   RES_write_en,
        output [$clog2(C_DIM)-1:0]    RES_write_channel,
        output [RES_depth_bits-1:0]   RES_write_address,  // Shared Address Bus
        output signed [  width-1:0]   RES_write_data_in   // Shared Data Bus
    );

    localparam OUT_W = IN_W >> 1;
    localparam OUT_H = IN_H >> 1;
    localparam WORDS_PER_RAM = OUT_W * OUT_H;

    // Define the states of state machine
    localparam IDLE    = 4'b1000;
    localparam COMPUTE = 4'b0100;
    localparam FLUSH   = 4'b0010;
    localparam FINISH  = 4'b0001;

    reg [3:0] state = IDLE;
    reg [3:0] n_state = IDLE;

    // Coordinate Registers (Stage 1)
    wire [$clog2(C_DIM)-1:0]     c;
    wire [$clog2(OUT_H)-1:0]     oh;
    wire [$clog2(OUT_W)-1:0]     ow;
    wire                         kh;
    wire                         kw;

    // Pipelined Channel Registers (To track which channel is at which stage)
    reg [$clog2(C_DIM)-1:0]      c_Read2;
    reg [$clog2(C_DIM)-1:0]      c_Read3;
    reg [$clog2(C_DIM)-1:0]      c_Cmp;
    (* max_fanout = "8" *) reg [$clog2(C_DIM)-1:0]      c_Out;

    // Coordinate counter enable registers
    reg en_kw, en_kh, en_ow, en_oh, en_c;

    // Stage 1: Read1 Signals
    reg counter_clear_Read1;
    reg reset_max_Read1;
    reg RES_write_en_Read1;

    // Stage 1.1: Read2 Signals
    reg reset_max_Read2;
    reg RES_write_en_Read2;

    // Stage 1.2: Read3 Signals
    reg reset_max_Read3;
    reg RES_write_en_Read3;

    // Stage 2: Compare
    reg reset_max_Cmp;
    reg RES_write_en_Cmp;
    reg [width-1:0] max_val_Cmp;

    // Stage 3: Output
    (* max_fanout = "8" *) reg RES_write_en_Out;
    (* max_fanout = "16" *) reg [width-1:0] max_val_Out;

    // -------------------------------------------------------------------------
    // FSM Control Logic
    // -------------------------------------------------------------------------
    always @(posedge clk) state <= n_state;

    always @(*) begin
        n_state = state;
        A_read_en = 0;
        counter_clear_Read1 = 0;
        consumer_frame_done = 1'b0;
        producer_frame_done = 1'b0;

        en_kw = 0;
        en_kh = 0;
        en_ow = 0;
        en_oh = 0;
        en_c = 0;
        reset_max_Read1 = 0;
        RES_write_en_Read1 = 0;

        case (state)
            IDLE: begin
                counter_clear_Read1 = 1'b1;
                if (consumer_ready && producer_ready)
                    n_state = COMPUTE;
            end
            COMPUTE: begin
                A_read_en = 1'b1;

                // Flag to tell the datapath when to reset the max tracker
                reset_max_Read1 = (kh == 0) && (kw == 0);

                // Flag to tell the datapath when to write the result
                RES_write_en_Read1  = (kh == 1) && (kw == 1);

                // Cascading Counters
                en_kw = 1'b1;
                en_kh = en_kw && (kw == 1);
                en_ow = en_kh && (kh == 1);
                en_oh = en_ow && (ow == OUT_W - 1);
                en_c  = en_oh && (oh == OUT_H - 1);

                // Check finish condition based on writes
                if (en_c && (c == C_DIM - 1)) begin
                    n_state = FLUSH;
                end
            end
            FLUSH: begin
                // Ensure pipeline has drained. We check if the pipelined channel
                // is the last channel and the per-channel address is complete.
                if (RES_write_en_Out && (c_Out == C_DIM - 1) && (RES_write_address == WORDS_PER_RAM - 1)) begin
                    n_state = FINISH;
                end
            end
            FINISH: begin
                consumer_frame_done = 1'b1; // Tell Input Ping-Pong to flip its read bank
                producer_frame_done = 1'b1; // Tell Output Ping-Pong to flip its write bank
                n_state = IDLE;
            end
        endcase
    end

    // =========================================================================
    // Datapath & Pipeline
    // =========================================================================

    // Stage 1: Read Stage

    // Pipeline Register 1: Read1/Read2
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            reset_max_Read2 <= 0;
            RES_write_en_Read2 <= 0;
            c_Read2 <= 0;
        end
        else begin
            reset_max_Read2 <= reset_max_Read1;
            RES_write_en_Read2 <= RES_write_en_Read1;
            c_Read2 <= c; // Pass the channel down the pipeline
        end
    end

    // Pipeline Register 1.1: Read2/Read3
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            reset_max_Read3 <= 0;
            RES_write_en_Read3 <= 0;
            c_Read3 <= 0;
        end
        else begin
            reset_max_Read3 <= reset_max_Read2;
            RES_write_en_Read3 <= RES_write_en_Read2;
            c_Read3 <= c_Read2; // Pass the channel down the pipeline
        end
    end

    // Pipeline Register 1.2: Read3/Compare
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            reset_max_Cmp <= 0;
            RES_write_en_Cmp <= 0;
            c_Cmp <= 0;
        end
        else begin
            reset_max_Cmp <= reset_max_Read3;
            RES_write_en_Cmp <= RES_write_en_Read3;
            c_Cmp <= c_Read3; // Pass the channel down the pipeline
        end
    end

    // Stage 2: Compare Stage
    always @(*) begin
        if (reset_max_Cmp || ($signed(A_read_data_out) > $signed(max_val_Out))) begin
            max_val_Cmp = A_read_data_out;
        end
        else begin
            max_val_Cmp = max_val_Out;
        end
    end

    // Pipeline Register 2: Compare Output
    always @(posedge clk) begin
        if (!rst || counter_clear_Read1) begin
            max_val_Out <= 0;
            RES_write_en_Out <= 0;
            c_Out <= 0;
        end
        else begin
            max_val_Out <= max_val_Cmp;
            RES_write_en_Out <= RES_write_en_Cmp;
            c_Out <= c_Cmp; // Pass the channel to the output demux
        end
    end

    // Stage 3: Output Stage
    assign RES_write_data_in = max_val_Out;
    assign RES_write_channel = c_Out;

    // Output Demultiplexer
    // Retain the one-hot write enable for compatibility and boundary checking.
    always @(*) begin
        RES_write_en = {C_DIM{1'b0}};
        if (RES_write_en_Out) begin
            RES_write_en[c_Out] = 1'b1;
        end
    end

    // Instantiations: Counters & AGUs
    // Kernel Width Counter (Inner)
    Counter #(
                .ADDR_WIDTH(1),
                .CNT_INIT(0),
                .MAX_VAL(1)
            ) kw_cnt (
                .CLK(clk),
                .RESET(!rst || counter_clear_Read1),
                .ENABLE(en_kw),
                .CNT(kw)
            );

    // Kernel Height Counter
    Counter #(
                .ADDR_WIDTH(1),
                .CNT_INIT(0),
                .MAX_VAL(1)
            ) kh_cnt (
                .CLK(clk),
                .RESET(!rst || counter_clear_Read1),
                .ENABLE(en_kh),
                .CNT(kh)
            );

    // Output Width Counter
    Counter #(
                .ADDR_WIDTH($clog2(OUT_W)),
                .CNT_INIT(0),
                .MAX_VAL(OUT_W - 1)
            ) ow_cnt (
                .CLK(clk),
                .RESET(!rst || counter_clear_Read1),
                .ENABLE(en_ow),
                .CNT(ow)
            );

    // Output Height Counter
    Counter #(
                .ADDR_WIDTH($clog2(OUT_H)),
                .CNT_INIT(0),
                .MAX_VAL(OUT_H - 1)
            ) oh_cnt (
                .CLK(clk),
                .RESET(!rst || counter_clear_Read1),
                .ENABLE(en_oh),
                .CNT(oh)
            );

    // Channel Counter (Outer)
    Counter #(
                .ADDR_WIDTH($clog2(C_DIM)),
                .CNT_INIT(0),
                .MAX_VAL(C_DIM - 1)
            ) c_cnt (
                .CLK(clk),
                .RESET(!rst || counter_clear_Read1),
                .ENABLE(en_c),
                .CNT(c)
            );

    // Input AGU
    maxpool_agu #(
                    .ADDR_WIDTH(A_depth_bits),
                    .C_DIM(C_DIM),
                    .IN_H(IN_H),
                    .IN_W(IN_W)
                ) agu_inst (
                    .c(c),
                    .oh(oh),
                    .ow(ow),
                    .kh(kh),
                    .kw(kw),
                    .read_address(A_read_address)
                );

    // Output Address Counter
    Counter #(
                .ADDR_WIDTH(RES_depth_bits),
                .CNT_INIT(0),
                .MAX_VAL(WORDS_PER_RAM - 1)
            ) res_cnt (
                .CLK(clk),
                .RESET(!rst || counter_clear_Read1),
                .ENABLE(RES_write_en_Out),
                .CNT(RES_write_address)
            );

endmodule

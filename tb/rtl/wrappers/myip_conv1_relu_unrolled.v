`timescale 1ns / 1ps

/* Standalone AXI-stream wrapper used to verify fused Conv1 + ReLU.
 * Input order: image[3][32][32], weights[32][3][3][3], biases[32].
 * Output order: activation[32][32][32], all channel-major.
 *
 * Conv1 computes four output channels concurrently.  Its four results are
 * serialized as two pairs into the two ports of one true dual-port BRAM.
 */
module myip_conv1_relu_unrolled (
        input ACLK,
        input ARESETN,
        output reg S_AXIS_TREADY,
        input [31:0] S_AXIS_TDATA,
        input S_AXIS_TLAST,
        input S_AXIS_TVALID,
        output reg M_AXIS_TVALID,
        output reg [31:0] M_AXIS_TDATA,
        output reg M_AXIS_TLAST,
        input M_AXIS_TREADY
    );

    localparam width = 16;
    localparam NUM_A = 3072;
    localparam NUM_B = 864;
    localparam NUM_C = 32;
    localparam NUM_RES = 32768;

    localparam Idle = 4'b1000;
    localparam Read_Inputs = 4'b0100;
    localparam Compute = 4'b0010;
    localparam Write_Outputs = 4'b0001;

    reg [3:0] state, n_state;
    reg [11:0] read_counter;
    reg [14:0] write_counter;

    reg [2:0] A_we;
    reg [11:0] B_we;
    reg [3:0] C_we;
    reg [9:0] A_waddr;
    reg [6:0] B_waddr;
    reg [2:0] C_waddr;

    reg [3:0] weight_chunk_counter;
    reg [3:0] weight_ram_select;
    reg [6:0] weight_base_address;

    wire A_read_en, B_read_en, C_read_en;
    wire [9:0] A_read_address;
    wire [6:0] B_read_address;
    wire [2:0] C_read_address;
    wire [15:0] A_read_data [0:2];
    wire [15:0] B_read_data [0:11];
    wire [15:0] C_read_data [0:3];
    wire [63:0] B_read_data_packed [0:2];

    wire RES_write_en_even, RES_write_en_odd;
    wire [14:0] RES_write_address_even, RES_write_address_odd;
    wire [15:0] RES_write_data_even, RES_write_data_odd;
    wire consumer_frame_done, producer_frame_done;
    wire core_ready = (state == Compute);
    wire [15:0] output_read_data;
    reg [1:0] output_read_latency;
    reg output_word_valid;
    wire output_read_en = (state == Write_Outputs) && !output_word_valid
         && (output_read_latency == 0);

    always @(posedge ACLK) begin
        if (!ARESETN)
            state <= Idle;
        else
            state <= n_state;
    end

    always @(*) begin
        n_state = state;
        S_AXIS_TREADY = 1'b0;
        M_AXIS_TVALID = 1'b0;
        M_AXIS_TLAST = 1'b0;
        M_AXIS_TDATA = {16'd0, output_read_data};
        A_we = 0;
        B_we = 0;
        C_we = 0;
        A_waddr = 0;
        B_waddr = 0;
        C_waddr = 0;

        case (state)
            Idle: begin
                if (S_AXIS_TVALID)
                    n_state = Read_Inputs;
            end

            Read_Inputs: begin
                S_AXIS_TREADY = 1'b1;
                if (S_AXIS_TVALID) begin
                    if (read_counter < NUM_A) begin
                        if (read_counter < 1024)
                            A_we[0] = 1'b1;
                        else if (read_counter < 2048)
                            A_we[1] = 1'b1;
                        else
                            A_we[2] = 1'b1;
                        A_waddr = read_counter[9:0];
                    end
                    else if (read_counter < NUM_A + NUM_B) begin
                        B_we[weight_ram_select] = 1'b1;
                        B_waddr = weight_base_address + weight_chunk_counter;
                    end
                    else begin
                        C_we[(read_counter - NUM_A - NUM_B) & 3] = 1'b1;
                        C_waddr = (read_counter - NUM_A - NUM_B) >> 2;
                    end

                    if (read_counter == NUM_A + NUM_B + NUM_C - 1)
                        n_state = Compute;
                end
            end

            Compute: begin
                if (producer_frame_done)
                    n_state = Write_Outputs;
            end

            Write_Outputs: begin
                M_AXIS_TVALID = output_word_valid;
                M_AXIS_TLAST = (write_counter == NUM_RES - 1);
                if (output_word_valid && M_AXIS_TREADY
                        && write_counter == NUM_RES - 1)
                    n_state = Idle;
            end
        endcase
    end

    always @(posedge ACLK) begin
        if (!ARESETN) begin
            read_counter <= 0;
            write_counter <= 0;
            weight_chunk_counter <= 0;
            weight_ram_select <= 0;
            weight_base_address <= 0;
            output_read_latency <= 0;
            output_word_valid <= 0;
        end
        else begin
            if (state == Read_Inputs && S_AXIS_TVALID) begin
                if (read_counter >= NUM_A && read_counter < NUM_A + NUM_B) begin
                    if (weight_chunk_counter == 8) begin
                        weight_chunk_counter <= 0;
                        if (weight_ram_select == 11) begin
                            weight_ram_select <= 0;
                            weight_base_address <= weight_base_address + 9;
                        end
                        else
                            weight_ram_select <= weight_ram_select + 1'b1;
                    end
                    else
                        weight_chunk_counter <= weight_chunk_counter + 1'b1;
                end

                if (read_counter == NUM_A + NUM_B + NUM_C - 1) begin
                    read_counter <= 0;
                    weight_chunk_counter <= 0;
                    weight_ram_select <= 0;
                    weight_base_address <= 0;
                end
                else
                    read_counter <= read_counter + 1'b1;
            end

            if (state == Write_Outputs && M_AXIS_TVALID && M_AXIS_TREADY) begin
                if (write_counter == NUM_RES - 1)
                    write_counter <= 0;
                else
                    write_counter <= write_counter + 1'b1;
            end

            if (state != Write_Outputs) begin
                output_read_latency <= 0;
                output_word_valid <= 0;
            end
            else if (output_word_valid) begin
                if (M_AXIS_TREADY)
                    output_word_valid <= 0;
            end
            else if (output_read_latency == 2) begin
                output_read_latency <= 0;
                output_word_valid <= 1;
            end
            else begin
                output_read_latency <= output_read_latency + 1'b1;
            end
        end
    end

    genvar g;
    generate
        for (g = 0; g < 3; g = g + 1) begin : INPUT_MEMORIES
            memory_RAM #(.width(width), .depth_bits(10)) input_RAM (
                           .clk(ACLK), .write_en(A_we[g]), .write_address(A_waddr),
                           .write_data_in(S_AXIS_TDATA[15:0]), .read_en(A_read_en),
                           .read_address(A_read_address), .read_data_out(A_read_data[g])
                       );
        end
        // Four logical 128x16 banks share each asynchronous distributed-RAM
        // module.  The core delays their address to meet the input BRAM data.
        for (g = 0; g < 3; g = g + 1) begin : WEIGHT_MEMORIES
            wire [3:0] packed_write_en = B_we[g*4 +: 4];
            memory_lutram_packed4 #(.width(width), .depth_bits(7)) weight_RAM (
                                      .clk(ACLK), .write_en(packed_write_en), .write_address(B_waddr),
                                      .write_data_in(S_AXIS_TDATA[15:0]), .read_en(B_read_en),
                                      .read_address(B_read_address), .read_data_out(B_read_data_packed[g])
                                  );
            assign B_read_data[g*4]   = B_read_data_packed[g][15:0];
            assign B_read_data[g*4+1] = B_read_data_packed[g][31:16];
            assign B_read_data[g*4+2] = B_read_data_packed[g][47:32];
            assign B_read_data[g*4+3] = B_read_data_packed[g][63:48];
        end
        for (g = 0; g < 4; g = g + 1) begin : BIAS_MEMORIES
            memory_lutram #(.width(width), .depth_bits(3)) bias_RAM (
                              .clk(ACLK), .write_en(C_we[g]), .write_address(C_waddr),
                              .write_data_in(S_AXIS_TDATA[15:0]), .read_en(C_read_en),
                              .read_address(C_read_address), .read_data_out(C_read_data[g])
                          );
        end
    endgenerate

    // Port A writes the even lane of each pair during compute and performs
    // the serialized AXI read after the complete result frame is available.
    memory_bram_tdp #(.width(width), .depth_bits(15)) output_RAM (
                        .clk(ACLK),
                        .port_a_enable((state == Compute) ? RES_write_en_even : output_read_en),
                        .port_a_write_en((state == Compute) ? RES_write_en_even : 1'b0),
                        .port_a_address((state == Compute) ? RES_write_address_even : write_counter),
                        .port_a_data_in(RES_write_data_even),
                        .port_a_data_out(output_read_data),
                        .port_b_enable((state == Compute) ? RES_write_en_odd : 1'b0),
                        .port_b_write_en((state == Compute) ? RES_write_en_odd : 1'b0),
                        .port_b_address(RES_write_address_odd),
                        .port_b_data_in(RES_write_data_odd),
                        .port_b_data_out()
                    );

    conv1_relu_unrolled_ping_pong #(
                                      .A_depth_bits(10), .B_depth_bits(7),
                                      .C_depth_bits(3), .RES_depth_bits(15),
                                      .OC_DIM(32), .OUT_UNROLL(4)
                                  ) core (
                                      .clk(ACLK), .rst(ARESETN),
                                      .consumer_ready(core_ready), .producer_ready(core_ready),
                                      .consumer_frame_done(consumer_frame_done),
                                      .producer_frame_done(producer_frame_done),
                                      .A_read_en(A_read_en), .A_read_address(A_read_address),
                                      .A_read_data_out_0(A_read_data[0]),
                                      .A_read_data_out_1(A_read_data[1]),
                                      .A_read_data_out_2(A_read_data[2]),
                                      .B_read_en(B_read_en), .B_read_address(B_read_address),
                                      .B_read_data_out_0_oc0(B_read_data[0]),
                                      .B_read_data_out_1_oc0(B_read_data[1]),
                                      .B_read_data_out_2_oc0(B_read_data[2]),
                                      .B_read_data_out_0_oc1(B_read_data[3]),
                                      .B_read_data_out_1_oc1(B_read_data[4]),
                                      .B_read_data_out_2_oc1(B_read_data[5]),
                                      .B_read_data_out_0_oc2(B_read_data[6]),
                                      .B_read_data_out_1_oc2(B_read_data[7]),
                                      .B_read_data_out_2_oc2(B_read_data[8]),
                                      .B_read_data_out_0_oc3(B_read_data[9]),
                                      .B_read_data_out_1_oc3(B_read_data[10]),
                                      .B_read_data_out_2_oc3(B_read_data[11]),
                                      .C_read_en(C_read_en), .C_read_address(C_read_address),
                                      .C_read_data_out_oc0(C_read_data[0]),
                                      .C_read_data_out_oc1(C_read_data[1]),
                                      .C_read_data_out_oc2(C_read_data[2]),
                                      .C_read_data_out_oc3(C_read_data[3]),
                                      .RES_write_en_even(RES_write_en_even),
                                      .RES_write_address_even(RES_write_address_even),
                                      .RES_write_data_in_even(RES_write_data_even),
                                      .RES_write_en_odd(RES_write_en_odd),
                                      .RES_write_address_odd(RES_write_address_odd),
                                      .RES_write_data_in_odd(RES_write_data_odd)
                                  );

endmodule

`timescale 1ns/1ps

module decision_tree_9 (
   input wire          clk,
    input wire [1:0]    start,
     input wire [143:0]  features, // Nâng cấp lên 288-bit để chứa 9 feature 32-bit

    output reg          done,
    output reg [1:0]    prediction
);

    // ========================================================================
    // GIẢI NÉN CHUẨN 32-BIT (Dùng cú pháp +: 32 để tránh nhầm lẫn vị trí)
    // ========================================================================
    wire [15:0]  kde_prob_mean           = features[0   +: 16]; 
    wire [15:0] kde_prob_night_mean      = features[16  +: 16]; 
    wire [15:0] dist_to_centroid_mean    = features[32  +: 16]; 
    wire [15:0] step_median              = features[48  +: 16]; 
    wire [15:0] mean_speed               = features[64 +: 16]; 
    wire [15:0] accelerate               = features[80 +: 16]; // Đây là bit [191:160]
    wire [15:0] turning_angle_max        = features[96 +: 16]; 
    wire [15:0] turning_angle_median     = features[112 +: 16]; 
    wire [15:0] is_night                 = features[128 +: 16]; 
    // ... Giữ nguyên logic Always @(*) so sánh 32-bit của bạn ...

    // ... Giữ nguyên logic always @(*) và FSM bên dưới ...
    // ========================================================================
    // Decision Tree Logic (combinational - giữ nguyên từ code bạn gửi)
    // ========================================================================
    reg tree_out;   // 1 = BẤT THƯỜNG, 0 = Bình thường
always @(*) begin
    if (kde_prob_night_mean <= 16'h4498) begin
        if (kde_prob_mean <= 16'h19C8) begin
            if (step_median <= 16'h029D) begin
                if (step_median <= 16'h0294) begin
                    tree_out = 1'b1;
                end else begin
                    tree_out = 1'b0;
                end
            end else begin
                if (kde_prob_mean <= 16'h19A0) begin
                    tree_out = 1'b1;
                end else begin
                    if (dist_to_centroid_mean <= 16'h65DB) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end
        end else begin
            if (step_median <= 16'h3034) begin
                if (kde_prob_night_mean <= 16'h0FF6) begin
                    if (dist_to_centroid_mean <= 16'h5B40) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    if (step_median <= 16'h0D22) begin
                        if (is_night <= 16'h8000) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (step_median <= 16'h0D58) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end else begin
                if (mean_speed <= 16'h66E1) begin
                    if (accelerate <= 16'h09E4) begin
                        tree_out = 1'b0;
                    end else begin
                        if (dist_to_centroid_mean <= 16'h225E) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (turning_angle_median <= 16'h2BCB) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end
        end
    end else begin
        if (mean_speed <= 16'h59AC) begin
            if (kde_prob_mean <= 16'h193C) begin
                tree_out = 1'b1;
            end else begin
                if (accelerate <= 16'h51E2) begin
                    if (accelerate <= 16'h0000) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    if (turning_angle_median <= 16'hEB5B) begin
                        if (accelerate <= 16'h51F9) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (turning_angle_median <= 16'hF4A3) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end
        end else begin
            if (kde_prob_mean <= 16'h5804) begin
                if (is_night <= 16'h8000) begin
                    tree_out = 1'b0;
                end else begin
                    if (kde_prob_night_mean <= 16'h5162) begin
                        tree_out = 1'b0;
                    end else begin
                        if (step_median <= 16'h60F4) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end else begin
                if (kde_prob_mean <= 16'hCC16) begin
                    tree_out = 1'b0;
                end else begin
                    if (kde_prob_mean <= 16'hCF34) begin
                        tree_out = 1'b1;
                    end else begin
                        if (turning_angle_max <= 16'h0AB0) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end
        end
    end
end

    // ========================================================================
    // FSM xử lý start (idle / run / clear) - giống hệt code mẫu bạn gửi
    // ========================================================================
    always @(posedge clk) begin
        if (start == 2'b00 || start == 2'b10) begin   // IDLE hoặc CLEAR
            done       <= 1'b0;
            prediction <= 2'b00;
        end
        else if (start == 2'b01) begin                // RUN
            done       <= 1'b1;
            prediction <= tree_out ? 2'b10 : 2'b01;   // 10 = bất thường
        end
    end

endmodule
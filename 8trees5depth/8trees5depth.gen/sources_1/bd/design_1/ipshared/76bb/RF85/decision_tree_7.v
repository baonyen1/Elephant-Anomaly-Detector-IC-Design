`timescale 1ns/1ps

module decision_tree_7 (
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
    if (kde_prob_night_mean <= 16'h3FE6) begin
        if (kde_prob_mean <= 16'h1999) begin
            tree_out = 1'b1;
        end else begin
            if (accelerate <= 16'h59A6) begin
                if (turning_angle_median <= 16'h1B30) begin
                    if (mean_speed <= 16'h55E9) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b1;
                    end
                end else begin
                    tree_out = 1'b0;
                end
            end else begin
                if (accelerate <= 16'h600F) begin
                    tree_out = 1'b1;
                end else begin
                    if (turning_angle_max <= 16'h071A) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end
        end
    end else begin
        if (step_median <= 16'h31EB) begin
            if (mean_speed <= 16'h4FFA) begin
                if (dist_to_centroid_mean <= 16'h7E72) begin
                    if (step_median <= 16'h10A8) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    if (kde_prob_night_mean <= 16'h469A) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end else begin
                if (turning_angle_median <= 16'hCEA8) begin
                    tree_out = 1'b0;
                end else begin
                    if (mean_speed <= 16'h501F) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end
        end else begin
            if (accelerate <= 16'h54A1) begin
                if (step_median <= 16'h4257) begin
                    tree_out = 1'b0;
                end else begin
                    if (kde_prob_mean <= 16'h4520) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end else begin
                if (kde_prob_night_mean <= 16'h8040) begin
                    if (dist_to_centroid_mean <= 16'h1DD3) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b1;
                    end
                end else begin
                    if (turning_angle_median <= 16'h1BD8) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b1;
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
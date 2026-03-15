`timescale 1ns/1ps

module random_forest_elephant
(
    input  wire        clk,
    input  wire [1:0]  start,           // 00: idle | 01: run | 10: clear

    // ==================== 9 FEATURES (MỖI CÁI 32-BIT) ====================
    input  wire [15:0] kde_prob_mean,    
    input  wire [15:0] kde_prob_night_mean,         
    input  wire [15:0] dist_to_centroid_mean,      
    input  wire [15:0] step_median,                   
    input  wire [15:0] mean_speed,                 
    input  wire [15:0] accelerate,         
    input  wire [15:0] turning_angle_max,       
    input  wire [15:0] turning_angle_median,                       
    input  wire [15:0] is_night,                 
   

    output reg         done,            
    output reg  [1:0]  result           
);

    // ========================================================================
    // Pack 9 features vào 288-bit (9 features * 32 bits = 288)
    // Giữ nguyên thứ tự để các cây con giải nén (unpack) chính xác
    // ========================================================================
    wire [143:0] features;
   assign features = {
  is_night,
  turning_angle_median,
  turning_angle_max,
  accelerate,
  mean_speed,
  step_median,
  dist_to_centroid_mean,
  kde_prob_night_mean,
  kde_prob_mean
};

    // ========================================================================
    // Instantiate 12 decision trees (Cập nhật interface lên 288-bit)
    // ========================================================================
    wire [11:0] t_done;
    wire [1:0]  t_pred [0:11];

    decision_tree_1  t1 (.clk(clk), .start(start), .features(features), .done(t_done[0]),  .prediction(t_pred[0]));
    decision_tree_2  t2 (.clk(clk), .start(start), .features(features), .done(t_done[1]),  .prediction(t_pred[1]));
    decision_tree_3  t3 (.clk(clk), .start(start), .features(features), .done(t_done[2]),  .prediction(t_pred[2]));
    decision_tree_4  t4 (.clk(clk), .start(start), .features(features), .done(t_done[3]),  .prediction(t_pred[3]));
    decision_tree_5  t5 (.clk(clk), .start(start), .features(features), .done(t_done[4]),  .prediction(t_pred[4]));
    decision_tree_6  t6 (.clk(clk), .start(start), .features(features), .done(t_done[5]),  .prediction(t_pred[5]));
    decision_tree_7  t7 (.clk(clk), .start(start), .features(features), .done(t_done[6]),  .prediction(t_pred[6]));
    decision_tree_8  t8 (.clk(clk), .start(start), .features(features), .done(t_done[7]),  .prediction(t_pred[7]));
    decision_tree_9  t9 (.clk(clk), .start(start), .features(features), .done(t_done[8]),  .prediction(t_pred[8]));
    decision_tree_10 t10(.clk(clk), .start(start), .features(features), .done(t_done[9]),  .prediction(t_pred[9]));
    decision_tree_11 t11(.clk(clk), .start(start), .features(features), .done(t_done[10]), .prediction(t_pred[10]));
    decision_tree_12 t12(.clk(clk), .start(start), .features(features), .done(t_done[11]), .prediction(t_pred[11]));

    // ========================================================================
    // Majority Voting (12 cây) - Sử dụng bit [1] để đếm các dự đoán '10' (Bất thường)
    // ========================================================================
   integer i;
reg [3:0] anomaly_cnt;

always @(posedge clk) begin
    if (&t_done) begin                 // tất cả 12 cây xong

        anomaly_cnt = 4'd0;

        for (i = 0; i < 12; i = i + 1)
            if (t_pred[i] == 2'b10)    // đếm anomaly
                anomaly_cnt = anomaly_cnt + 1'b1;

        // majority vote
        result <= (anomaly_cnt >= 4'd7) ? 2'b10 : 2'b01;

        done <= 1'b1;

    end
    else begin
        done   <= 1'b0;
        result <= 2'b00;
    end
end
endmodule
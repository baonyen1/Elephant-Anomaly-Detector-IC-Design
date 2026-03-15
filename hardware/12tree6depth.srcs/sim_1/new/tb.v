`timescale 1ns/1ps

module tb_random_forest_elephant;

reg clk;
reg [1:0] start;

// 9 FEATURES
reg [15:0] kde_prob_mean;
reg [15:0] kde_prob_night_mean;
reg [15:0] dist_to_centroid_mean;
reg [15:0] step_median;
reg [15:0] mean_speed;
reg [15:0] accelerate;
reg [15:0] turning_angle_max;
reg [15:0] turning_angle_median;
reg [15:0] is_night;

wire done;
wire [1:0] result;

random_forest_elephant uut (
    .clk(clk),
    .start(start),

    .kde_prob_mean(kde_prob_mean),
    .kde_prob_night_mean(kde_prob_night_mean),
    .dist_to_centroid_mean(dist_to_centroid_mean),
    .step_median(step_median),
    .mean_speed(mean_speed),
    .accelerate(accelerate),
    .turning_angle_max(turning_angle_max),
    .turning_angle_median(turning_angle_median),
    .is_night(is_night),

    .done(done),
    .result(result)
);

//////////////////////////////////////////////////
// CLOCK
//////////////////////////////////////////////////

always #5 clk = ~clk;

//////////////////////////////////////////////////
// TEST
//////////////////////////////////////////////////

initial begin

clk = 0;
start = 2'b00;

#20;

//////////////////////////////////////////////////
// CASE 1 : NORMAL
//////////////////////////////////////////////////

$display("CASE 1 : NORMAL");

kde_prob_mean        = 16'hA700;
kde_prob_night_mean  = 16'hD400;
dist_to_centroid_mean= 16'h0200;
step_median          = 16'h0100;
mean_speed           = 16'h0400;
accelerate           = 16'h0500;
turning_angle_max    = 16'h0300;
turning_angle_median = 16'h0200;
is_night             = 16'h0000;

start = 2'b01;

#20;

$display("Result = %d", result);

start = 2'b10;
#20;
start = 2'b00;

#40;

//////////////////////////////////////////////////
// CASE 2 : ANOMALY
//////////////////////////////////////////////////

$display("CASE 2 : ANOMALY");

kde_prob_mean        = 16'h0900;
kde_prob_night_mean  = 16'h0700;
dist_to_centroid_mean= 16'h6000;
step_median          = 16'h5000;
mean_speed           = 16'h7000;
accelerate           = 16'h7000;
turning_angle_max    = 16'h8000;
turning_angle_median = 16'h7000;
is_night             = 16'h8000;

start = 2'b01;

#20;

$display("Result = %d", result);

start = 2'b10;
#20;
start = 2'b00;

#40;

//////////////////////////////////////////////////
// CASE 3 : NOISE TEST (FFFF)
//////////////////////////////////////////////////

$display("CASE 3 : NOISE TEST");

kde_prob_mean        = 16'hB400;
kde_prob_night_mean  = 16'h0800;
dist_to_centroid_mean= 16'hFFFF;
step_median          = 16'h0100;
mean_speed           = 16'h1A00;
accelerate           = 16'hFFFF;
turning_angle_max    = 16'h0300;
turning_angle_median = 16'h0200;
is_night             = 16'h0000;

start = 2'b01;

#20;

$display("Result = %d", result);

#50;

$finish;

end

endmodule
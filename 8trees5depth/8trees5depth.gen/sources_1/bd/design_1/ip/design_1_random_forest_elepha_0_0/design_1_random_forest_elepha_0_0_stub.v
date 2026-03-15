// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2024.1 (win64) Build 5076996 Wed May 22 18:37:14 MDT 2024
// Date        : Fri Mar 13 12:33:03 2026
// Host        : DESKTOP-AUH71TB running 64-bit major release  (build 9200)
// Command     : write_verilog -force -mode synth_stub
//               d:/8trees5depth/8trees5depth.gen/sources_1/bd/design_1/ip/design_1_random_forest_elepha_0_0/design_1_random_forest_elepha_0_0_stub.v
// Design      : design_1_random_forest_elepha_0_0
// Purpose     : Stub declaration of top-level module interface
// Device      : xc7z010clg400-1
// --------------------------------------------------------------------------------

// This empty module with port declaration file causes synthesis tools to infer a black box for IP.
// The synthesis directives are for Synopsys Synplify support to prevent IO buffer insertion.
// Please paste the declaration into a Verilog source file or add the file as an additional source.
(* X_CORE_INFO = "random_forest_elephant,Vivado 2024.1" *)
module design_1_random_forest_elepha_0_0(clk, start, kde_prob_mean, kde_prob_night_mean, 
  dist_to_centroid_mean, step_median, mean_speed, accelerate, turning_angle_max, 
  turning_angle_median, is_night, done, result)
/* synthesis syn_black_box black_box_pad_pin="start[1:0],kde_prob_mean[15:0],kde_prob_night_mean[15:0],dist_to_centroid_mean[15:0],step_median[15:0],mean_speed[15:0],accelerate[15:0],turning_angle_max[15:0],turning_angle_median[15:0],is_night[15:0],done,result[1:0]" */
/* synthesis syn_force_seq_prim="clk" */;
  input clk /* synthesis syn_isclock = 1 */;
  input [1:0]start;
  input [15:0]kde_prob_mean;
  input [15:0]kde_prob_night_mean;
  input [15:0]dist_to_centroid_mean;
  input [15:0]step_median;
  input [15:0]mean_speed;
  input [15:0]accelerate;
  input [15:0]turning_angle_max;
  input [15:0]turning_angle_median;
  input [15:0]is_night;
  output done;
  output [1:0]result;
endmodule

// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2024.1 (win64) Build 5076996 Wed May 22 18:37:14 MDT 2024
// Date        : Fri Mar 13 12:33:03 2026
// Host        : DESKTOP-AUH71TB running 64-bit major release  (build 9200)
// Command     : write_verilog -force -mode funcsim
//               d:/8trees5depth/8trees5depth.gen/sources_1/bd/design_1/ip/design_1_random_forest_elepha_0_0/design_1_random_forest_elepha_0_0_sim_netlist.v
// Design      : design_1_random_forest_elepha_0_0
// Purpose     : This verilog netlist is a functional simulation representation of the design and should not be modified
//               or synthesized. This netlist cannot be used for SDF annotated simulation.
// Device      : xc7z010clg400-1
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

(* CHECK_LICENSE_TYPE = "design_1_random_forest_elepha_0_0,random_forest_elephant,{}" *) (* DowngradeIPIdentifiedWarnings = "yes" *) (* IP_DEFINITION_SOURCE = "package_project" *) 
(* X_CORE_INFO = "random_forest_elephant,Vivado 2024.1" *) 
(* NotValidForBitStream *)
module design_1_random_forest_elepha_0_0
   (clk,
    start,
    kde_prob_mean,
    kde_prob_night_mean,
    dist_to_centroid_mean,
    step_median,
    mean_speed,
    accelerate,
    turning_angle_max,
    turning_angle_median,
    is_night,
    done,
    result);
  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME clk, FREQ_HZ 50000000, FREQ_TOLERANCE_HZ 0, PHASE 0.0, CLK_DOMAIN design_1_processing_system7_0_0_FCLK_CLK0, INSERT_VIP 0" *) input clk;
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

  wire [15:0]accelerate;
  wire clk;
  wire [15:0]dist_to_centroid_mean;
  wire done;
  wire [15:0]is_night;
  wire [15:0]kde_prob_mean;
  wire [15:0]kde_prob_night_mean;
  wire [15:0]mean_speed;
  wire n_0_373;
  wire [1:0]result;
  wire [1:0]start;
  wire [15:0]step_median;
  wire [15:0]turning_angle_max;
  wire [15:0]turning_angle_median;

  LUT1 #(
    .INIT(2'h1)) 
    i_373
       (.I0(start[0]),
        .O(n_0_373));
  design_1_random_forest_elepha_0_0_random_forest_elephant inst
       (.accelerate(accelerate),
        .clk(clk),
        .dist_to_centroid_mean(dist_to_centroid_mean),
        .done(done),
        .is_night(is_night),
        .kde_prob_mean(kde_prob_mean),
        .kde_prob_night_mean(kde_prob_night_mean),
        .mean_speed(mean_speed),
        .result(result),
        .start(start),
        .step_median(step_median),
        .turning_angle_max(turning_angle_max),
        .turning_angle_median(turning_angle_median));
endmodule

(* ORIG_REF_NAME = "decision_tree_1" *) 
module design_1_random_forest_elepha_0_0_decision_tree_1
   (t_done,
    accelerate_14_sp_1,
    accelerate_13_sp_1,
    mean_speed_10_sp_1,
    mean_speed_5_sp_1,
    mean_speed_9_sp_1,
    \prediction_reg[1]_0 ,
    \prediction_reg[0]_0 ,
    \prediction_reg[1]_1 ,
    clk,
    \prediction_reg[0]_1 ,
    \prediction_reg[0]_2 ,
    \prediction_reg[0]_3 ,
    \prediction_reg[0]_4 ,
    accelerate,
    \prediction[1]_i_10__1_0 ,
    step_median,
    mean_speed,
    \prediction_reg[1]_i_7_0 ,
    turning_angle_max,
    \prediction[1]_i_12__1_0 ,
    kde_prob_mean,
    \prediction_reg[1]_2 ,
    \prediction_reg[1]_i_6 ,
    \prediction[1]_i_13__3_0 ,
    \prediction[1]_i_13__3_1 ,
    \prediction[1]_i_13__3_2 ,
    \prediction[1]_i_17__0_0 ,
    turning_angle_median,
    kde_prob_night_mean,
    start,
    \prediction[1]_i_10__1_1 ,
    \prediction_reg[1]_3 );
  output [0:0]t_done;
  output accelerate_14_sp_1;
  output accelerate_13_sp_1;
  output mean_speed_10_sp_1;
  output mean_speed_5_sp_1;
  output mean_speed_9_sp_1;
  output \prediction_reg[1]_0 ;
  output \prediction_reg[0]_0 ;
  input \prediction_reg[1]_1 ;
  input clk;
  input \prediction_reg[0]_1 ;
  input \prediction_reg[0]_2 ;
  input \prediction_reg[0]_3 ;
  input \prediction_reg[0]_4 ;
  input [15:0]accelerate;
  input \prediction[1]_i_10__1_0 ;
  input [11:0]step_median;
  input [15:0]mean_speed;
  input \prediction_reg[1]_i_7_0 ;
  input [13:0]turning_angle_max;
  input \prediction[1]_i_12__1_0 ;
  input [15:0]kde_prob_mean;
  input \prediction_reg[1]_2 ;
  input \prediction_reg[1]_i_6 ;
  input \prediction[1]_i_13__3_0 ;
  input \prediction[1]_i_13__3_1 ;
  input \prediction[1]_i_13__3_2 ;
  input \prediction[1]_i_17__0_0 ;
  input [14:0]turning_angle_median;
  input [13:0]kde_prob_night_mean;
  input [0:0]start;
  input \prediction[1]_i_10__1_1 ;
  input \prediction_reg[1]_3 ;

  wire [15:0]accelerate;
  wire accelerate_13_sn_1;
  wire accelerate_14_sn_1;
  wire clk;
  wire done_i_1__0_n_0;
  wire [15:0]kde_prob_mean;
  wire [13:0]kde_prob_night_mean;
  wire [15:0]mean_speed;
  wire mean_speed_10_sn_1;
  wire mean_speed_5_sn_1;
  wire mean_speed_9_sn_1;
  wire \prediction[0]_i_1__4_n_0 ;
  wire \prediction[1]_i_10__1_0 ;
  wire \prediction[1]_i_10__1_1 ;
  wire \prediction[1]_i_10__1_n_0 ;
  wire \prediction[1]_i_11__0_n_0 ;
  wire \prediction[1]_i_12__1_0 ;
  wire \prediction[1]_i_12__1_n_0 ;
  wire \prediction[1]_i_13__3_0 ;
  wire \prediction[1]_i_13__3_1 ;
  wire \prediction[1]_i_13__3_2 ;
  wire \prediction[1]_i_13__5_n_0 ;
  wire \prediction[1]_i_14__5_n_0 ;
  wire \prediction[1]_i_15__5_n_0 ;
  wire \prediction[1]_i_17__0_0 ;
  wire \prediction[1]_i_17__0_n_0 ;
  wire \prediction[1]_i_18__5_n_0 ;
  wire \prediction[1]_i_19__4_n_0 ;
  wire \prediction[1]_i_1__1_n_0 ;
  wire \prediction[1]_i_20__5_n_0 ;
  wire \prediction[1]_i_21_n_0 ;
  wire \prediction[1]_i_22__1_n_0 ;
  wire \prediction[1]_i_22__2_n_0 ;
  wire \prediction[1]_i_23__5_n_0 ;
  wire \prediction[1]_i_24__2_n_0 ;
  wire \prediction[1]_i_25__3_n_0 ;
  wire \prediction[1]_i_26_n_0 ;
  wire \prediction[1]_i_27__3_n_0 ;
  wire \prediction[1]_i_28__0_n_0 ;
  wire \prediction[1]_i_30__2_n_0 ;
  wire \prediction[1]_i_31__2_n_0 ;
  wire \prediction[1]_i_33__2_n_0 ;
  wire \prediction[1]_i_34_n_0 ;
  wire \prediction[1]_i_35__2_n_0 ;
  wire \prediction[1]_i_36__2_n_0 ;
  wire \prediction[1]_i_37__0_n_0 ;
  wire \prediction[1]_i_38__1_n_0 ;
  wire \prediction[1]_i_4__1_n_0 ;
  wire \prediction[1]_i_9__1_n_0 ;
  wire \prediction_reg[0]_0 ;
  wire \prediction_reg[0]_1 ;
  wire \prediction_reg[0]_2 ;
  wire \prediction_reg[0]_3 ;
  wire \prediction_reg[0]_4 ;
  wire \prediction_reg[1]_0 ;
  wire \prediction_reg[1]_1 ;
  wire \prediction_reg[1]_2 ;
  wire \prediction_reg[1]_3 ;
  wire \prediction_reg[1]_i_6 ;
  wire \prediction_reg[1]_i_7_0 ;
  wire \prediction_reg[1]_i_7_n_0 ;
  wire [0:0]start;
  wire [11:0]step_median;
  wire [0:0]t_done;
  wire tree_out;
  wire [13:0]turning_angle_max;
  wire [14:0]turning_angle_median;

  assign accelerate_13_sp_1 = accelerate_13_sn_1;
  assign accelerate_14_sp_1 = accelerate_14_sn_1;
  assign mean_speed_10_sp_1 = mean_speed_10_sn_1;
  assign mean_speed_5_sp_1 = mean_speed_5_sn_1;
  assign mean_speed_9_sp_1 = mean_speed_9_sn_1;
  LUT2 #(
    .INIT(4'hD)) 
    done_i_1__0
       (.I0(start),
        .I1(t_done),
        .O(done_i_1__0_n_0));
  FDRE done_reg
       (.C(clk),
        .CE(1'b1),
        .D(done_i_1__0_n_0),
        .Q(t_done),
        .R(\prediction_reg[1]_1 ));
  LUT6 #(
    .INIT(64'h5555555544440040)) 
    \prediction[0]_i_1__4 
       (.I0(\prediction_reg[1]_i_7_n_0 ),
        .I1(\prediction_reg[0]_1 ),
        .I2(\prediction_reg[0]_2 ),
        .I3(\prediction[1]_i_4__1_n_0 ),
        .I4(\prediction_reg[0]_3 ),
        .I5(\prediction_reg[0]_4 ),
        .O(\prediction[0]_i_1__4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT4 #(
    .INIT(16'h7FFF)) 
    \prediction[0]_i_44 
       (.I0(mean_speed[9]),
        .I1(mean_speed[8]),
        .I2(mean_speed[11]),
        .I3(mean_speed[10]),
        .O(mean_speed_9_sn_1));
  LUT4 #(
    .INIT(16'hFFFE)) 
    \prediction[0]_i_6__2 
       (.I0(accelerate[13]),
        .I1(accelerate[12]),
        .I2(accelerate[15]),
        .I3(accelerate[14]),
        .O(accelerate_13_sn_1));
  LUT6 #(
    .INIT(64'h888888B888888888)) 
    \prediction[1]_i_10__1 
       (.I0(tree_out),
        .I1(\prediction[1]_i_17__0_n_0 ),
        .I2(\prediction[1]_i_18__5_n_0 ),
        .I3(accelerate_13_sn_1),
        .I4(\prediction[1]_i_19__4_n_0 ),
        .I5(accelerate[11]),
        .O(\prediction[1]_i_10__1_n_0 ));
  LUT5 #(
    .INIT(32'hEEEEEEAE)) 
    \prediction[1]_i_11__0 
       (.I0(mean_speed[15]),
        .I1(mean_speed[14]),
        .I2(\prediction[1]_i_20__5_n_0 ),
        .I3(mean_speed[13]),
        .I4(mean_speed[12]),
        .O(\prediction[1]_i_11__0_n_0 ));
  LUT5 #(
    .INIT(32'h000077F7)) 
    \prediction[1]_i_12__1 
       (.I0(turning_angle_max[11]),
        .I1(turning_angle_max[12]),
        .I2(\prediction[1]_i_21_n_0 ),
        .I3(turning_angle_max[10]),
        .I4(turning_angle_max[13]),
        .O(\prediction[1]_i_12__1_n_0 ));
  LUT6 #(
    .INIT(64'h00000000555577F7)) 
    \prediction[1]_i_13__3 
       (.I0(accelerate[14]),
        .I1(accelerate[12]),
        .I2(\prediction[1]_i_22__1_n_0 ),
        .I3(\prediction_reg[1]_i_6 ),
        .I4(accelerate[13]),
        .I5(accelerate[15]),
        .O(accelerate_14_sn_1));
  LUT6 #(
    .INIT(64'h00000000777777F7)) 
    \prediction[1]_i_13__5 
       (.I0(kde_prob_night_mean[11]),
        .I1(kde_prob_night_mean[12]),
        .I2(\prediction[1]_i_22__2_n_0 ),
        .I3(\prediction[1]_i_23__5_n_0 ),
        .I4(kde_prob_night_mean[6]),
        .I5(kde_prob_night_mean[13]),
        .O(\prediction[1]_i_13__5_n_0 ));
  LUT6 #(
    .INIT(64'h1055FFFFFFFFFFFF)) 
    \prediction[1]_i_14__5 
       (.I0(turning_angle_median[12]),
        .I1(turning_angle_median[10]),
        .I2(\prediction[1]_i_24__2_n_0 ),
        .I3(turning_angle_median[11]),
        .I4(turning_angle_median[14]),
        .I5(turning_angle_median[13]),
        .O(\prediction[1]_i_14__5_n_0 ));
  LUT6 #(
    .INIT(64'h00010000FFFFFFFF)) 
    \prediction[1]_i_15__5 
       (.I0(kde_prob_mean[12]),
        .I1(kde_prob_mean[11]),
        .I2(kde_prob_mean[14]),
        .I3(kde_prob_mean[13]),
        .I4(\prediction[1]_i_25__3_n_0 ),
        .I5(kde_prob_mean[15]),
        .O(\prediction[1]_i_15__5_n_0 ));
  LUT6 #(
    .INIT(64'h4500454545004500)) 
    \prediction[1]_i_16 
       (.I0(\prediction[1]_i_10__1_0 ),
        .I1(\prediction[1]_i_26_n_0 ),
        .I2(step_median[11]),
        .I3(mean_speed[15]),
        .I4(\prediction[1]_i_27__3_n_0 ),
        .I5(mean_speed[14]),
        .O(tree_out));
  LUT6 #(
    .INIT(64'h00000000555577F7)) 
    \prediction[1]_i_17__0 
       (.I0(mean_speed[14]),
        .I1(mean_speed[12]),
        .I2(\prediction[1]_i_28__0_n_0 ),
        .I3(mean_speed[11]),
        .I4(mean_speed[13]),
        .I5(mean_speed[15]),
        .O(\prediction[1]_i_17__0_n_0 ));
  LUT6 #(
    .INIT(64'h777F77777F7F7F7F)) 
    \prediction[1]_i_18__5 
       (.I0(accelerate[11]),
        .I1(accelerate[10]),
        .I2(\prediction[1]_i_10__1_1 ),
        .I3(accelerate[6]),
        .I4(\prediction[1]_i_30__2_n_0 ),
        .I5(accelerate[7]),
        .O(\prediction[1]_i_18__5_n_0 ));
  LUT6 #(
    .INIT(64'h000000000DFFFFFF)) 
    \prediction[1]_i_19__4 
       (.I0(accelerate[6]),
        .I1(\prediction[1]_i_31__2_n_0 ),
        .I2(accelerate[7]),
        .I3(accelerate[8]),
        .I4(accelerate[9]),
        .I5(accelerate[10]),
        .O(\prediction[1]_i_19__4_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFF10115555)) 
    \prediction[1]_i_1__1 
       (.I0(\prediction_reg[0]_4 ),
        .I1(\prediction_reg[0]_3 ),
        .I2(\prediction[1]_i_4__1_n_0 ),
        .I3(\prediction_reg[0]_2 ),
        .I4(\prediction_reg[0]_1 ),
        .I5(\prediction_reg[1]_i_7_n_0 ),
        .O(\prediction[1]_i_1__1_n_0 ));
  LUT6 #(
    .INIT(64'hABBBBBBBBBBBBBBB)) 
    \prediction[1]_i_20__5 
       (.I0(mean_speed_9_sn_1),
        .I1(mean_speed_5_sn_1),
        .I2(mean_speed[2]),
        .I3(mean_speed[3]),
        .I4(mean_speed[1]),
        .I5(mean_speed[0]),
        .O(\prediction[1]_i_20__5_n_0 ));
  LUT6 #(
    .INIT(64'h5455FFFFFFFFFFFF)) 
    \prediction[1]_i_21 
       (.I0(turning_angle_max[8]),
        .I1(\prediction[1]_i_33__2_n_0 ),
        .I2(\prediction[1]_i_34_n_0 ),
        .I3(turning_angle_max[5]),
        .I4(\prediction[1]_i_12__1_0 ),
        .I5(turning_angle_max[9]),
        .O(\prediction[1]_i_21_n_0 ));
  LUT6 #(
    .INIT(64'h10555555FFFFFFFF)) 
    \prediction[1]_i_22__1 
       (.I0(\prediction[1]_i_13__3_0 ),
        .I1(accelerate[2]),
        .I2(\prediction[1]_i_13__3_1 ),
        .I3(accelerate[4]),
        .I4(accelerate[3]),
        .I5(\prediction[1]_i_13__3_2 ),
        .O(\prediction[1]_i_22__1_n_0 ));
  LUT6 #(
    .INIT(64'h01010111FFFFFFFF)) 
    \prediction[1]_i_22__2 
       (.I0(kde_prob_night_mean[3]),
        .I1(kde_prob_night_mean[4]),
        .I2(kde_prob_night_mean[2]),
        .I3(kde_prob_night_mean[1]),
        .I4(kde_prob_night_mean[0]),
        .I5(kde_prob_night_mean[5]),
        .O(\prediction[1]_i_22__2_n_0 ));
  LUT4 #(
    .INIT(16'hFFFE)) 
    \prediction[1]_i_23__5 
       (.I0(kde_prob_night_mean[8]),
        .I1(kde_prob_night_mean[7]),
        .I2(kde_prob_night_mean[10]),
        .I3(kde_prob_night_mean[9]),
        .O(\prediction[1]_i_23__5_n_0 ));
  LUT6 #(
    .INIT(64'h5515FFFFFFFFFFFF)) 
    \prediction[1]_i_24__2 
       (.I0(turning_angle_median[7]),
        .I1(turning_angle_median[6]),
        .I2(turning_angle_median[5]),
        .I3(\prediction[1]_i_35__2_n_0 ),
        .I4(turning_angle_median[9]),
        .I5(turning_angle_median[8]),
        .O(\prediction[1]_i_24__2_n_0 ));
  LUT6 #(
    .INIT(64'h01001111FFFFFFFF)) 
    \prediction[1]_i_25__3 
       (.I0(kde_prob_mean[8]),
        .I1(kde_prob_mean[9]),
        .I2(kde_prob_mean[6]),
        .I3(\prediction[1]_i_36__2_n_0 ),
        .I4(kde_prob_mean[7]),
        .I5(kde_prob_mean[10]),
        .O(\prediction[1]_i_25__3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000777777F7)) 
    \prediction[1]_i_26 
       (.I0(step_median[8]),
        .I1(step_median[9]),
        .I2(\prediction[1]_i_37__0_n_0 ),
        .I3(step_median[7]),
        .I4(step_median[6]),
        .I5(step_median[10]),
        .O(\prediction[1]_i_26_n_0 ));
  LUT6 #(
    .INIT(64'h000000000000DDFD)) 
    \prediction[1]_i_27__3 
       (.I0(mean_speed[7]),
        .I1(mean_speed_9_sn_1),
        .I2(\prediction[1]_i_38__1_n_0 ),
        .I3(mean_speed[6]),
        .I4(mean_speed[13]),
        .I5(mean_speed[12]),
        .O(\prediction[1]_i_27__3_n_0 ));
  LUT6 #(
    .INIT(64'h45555555FFFFFFFF)) 
    \prediction[1]_i_28__0 
       (.I0(mean_speed[6]),
        .I1(\prediction[1]_i_17__0_0 ),
        .I2(mean_speed[4]),
        .I3(mean_speed[5]),
        .I4(mean_speed[3]),
        .I5(mean_speed_10_sn_1),
        .O(\prediction[1]_i_28__0_n_0 ));
  LUT6 #(
    .INIT(64'h15555555FFFFFFFF)) 
    \prediction[1]_i_30__2 
       (.I0(accelerate[4]),
        .I1(accelerate[2]),
        .I2(accelerate[3]),
        .I3(accelerate[1]),
        .I4(accelerate[0]),
        .I5(accelerate[5]),
        .O(\prediction[1]_i_30__2_n_0 ));
  LUT6 #(
    .INIT(64'h000000000000557F)) 
    \prediction[1]_i_31__2 
       (.I0(accelerate[3]),
        .I1(accelerate[0]),
        .I2(accelerate[1]),
        .I3(accelerate[2]),
        .I4(accelerate[5]),
        .I5(accelerate[4]),
        .O(\prediction[1]_i_31__2_n_0 ));
  LUT4 #(
    .INIT(16'hFFFE)) 
    \prediction[1]_i_32 
       (.I0(mean_speed[5]),
        .I1(mean_speed[4]),
        .I2(mean_speed[7]),
        .I3(mean_speed[6]),
        .O(mean_speed_5_sn_1));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \prediction[1]_i_32__2 
       (.I0(mean_speed[10]),
        .I1(mean_speed[9]),
        .I2(mean_speed[8]),
        .I3(mean_speed[7]),
        .O(mean_speed_10_sn_1));
  LUT5 #(
    .INIT(32'h00007FFF)) 
    \prediction[1]_i_33__2 
       (.I0(turning_angle_max[0]),
        .I1(turning_angle_max[1]),
        .I2(turning_angle_max[3]),
        .I3(turning_angle_max[2]),
        .I4(turning_angle_max[4]),
        .O(\prediction[1]_i_33__2_n_0 ));
  LUT2 #(
    .INIT(4'h7)) 
    \prediction[1]_i_34 
       (.I0(turning_angle_max[6]),
        .I1(turning_angle_max[7]),
        .O(\prediction[1]_i_34_n_0 ));
  LUT5 #(
    .INIT(32'h00001FFF)) 
    \prediction[1]_i_35__2 
       (.I0(turning_angle_median[0]),
        .I1(turning_angle_median[1]),
        .I2(turning_angle_median[2]),
        .I3(turning_angle_median[3]),
        .I4(turning_angle_median[4]),
        .O(\prediction[1]_i_35__2_n_0 ));
  LUT6 #(
    .INIT(64'h15555555FFFFFFFF)) 
    \prediction[1]_i_36__2 
       (.I0(kde_prob_mean[4]),
        .I1(kde_prob_mean[2]),
        .I2(kde_prob_mean[3]),
        .I3(kde_prob_mean[1]),
        .I4(kde_prob_mean[0]),
        .I5(kde_prob_mean[5]),
        .O(\prediction[1]_i_36__2_n_0 ));
  LUT6 #(
    .INIT(64'h00010101FFFFFFFF)) 
    \prediction[1]_i_37__0 
       (.I0(step_median[2]),
        .I1(step_median[4]),
        .I2(step_median[3]),
        .I3(step_median[1]),
        .I4(step_median[0]),
        .I5(step_median[5]),
        .O(\prediction[1]_i_37__0_n_0 ));
  LUT6 #(
    .INIT(64'h01FFFFFFFFFFFFFF)) 
    \prediction[1]_i_38__1 
       (.I0(mean_speed[0]),
        .I1(mean_speed[1]),
        .I2(mean_speed[2]),
        .I3(mean_speed[4]),
        .I4(mean_speed[5]),
        .I5(mean_speed[3]),
        .O(\prediction[1]_i_38__1_n_0 ));
  LUT6 #(
    .INIT(64'h000000007777777F)) 
    \prediction[1]_i_4__1 
       (.I0(kde_prob_mean[3]),
        .I1(kde_prob_mean[4]),
        .I2(kde_prob_mean[2]),
        .I3(kde_prob_mean[1]),
        .I4(kde_prob_mean[0]),
        .I5(\prediction_reg[1]_2 ),
        .O(\prediction[1]_i_4__1_n_0 ));
  LUT6 #(
    .INIT(64'hB800B800B8FFB800)) 
    \prediction[1]_i_9__1 
       (.I0(\prediction[1]_i_11__0_n_0 ),
        .I1(\prediction[1]_i_12__1_n_0 ),
        .I2(\prediction[1]_i_13__5_n_0 ),
        .I3(\prediction[1]_i_14__5_n_0 ),
        .I4(\prediction[1]_i_15__5_n_0 ),
        .I5(\prediction_reg[1]_i_7_0 ),
        .O(\prediction[1]_i_9__1_n_0 ));
  FDRE \prediction_reg[0] 
       (.C(clk),
        .CE(\prediction_reg[1]_3 ),
        .D(\prediction[0]_i_1__4_n_0 ),
        .Q(\prediction_reg[0]_0 ),
        .R(\prediction_reg[1]_1 ));
  FDRE \prediction_reg[1] 
       (.C(clk),
        .CE(\prediction_reg[1]_3 ),
        .D(\prediction[1]_i_1__1_n_0 ),
        .Q(\prediction_reg[1]_0 ),
        .R(\prediction_reg[1]_1 ));
  MUXF7 \prediction_reg[1]_i_7 
       (.I0(\prediction[1]_i_9__1_n_0 ),
        .I1(\prediction[1]_i_10__1_n_0 ),
        .O(\prediction_reg[1]_i_7_n_0 ),
        .S(accelerate_14_sn_1));
endmodule

(* ORIG_REF_NAME = "decision_tree_2" *) 
module design_1_random_forest_elepha_0_0_decision_tree_2
   (t_done,
    kde_prob_mean_7_sp_1,
    \kde_prob_mean[7]_0 ,
    kde_prob_mean_3_sp_1,
    accelerate_7_sp_1,
    kde_prob_night_mean_8_sp_1,
    dist_to_centroid_mean_12_sp_1,
    kde_prob_night_mean_2_sp_1,
    \prediction_reg[1]_0 ,
    \prediction_reg[0]_0 ,
    \prediction_reg[1]_1 ,
    clk,
    \prediction_reg[0]_1 ,
    \prediction_reg[1]_2 ,
    \prediction_reg[1]_3 ,
    kde_prob_mean,
    \prediction_reg[1]_4 ,
    \prediction_reg[1]_5 ,
    mean_speed,
    step_median,
    \prediction[1]_i_23__0_0 ,
    \prediction[1]_i_23__0_1 ,
    \prediction[1]_i_9__3_0 ,
    \prediction[1]_i_14__0_0 ,
    \prediction[1]_i_25__0_0 ,
    accelerate,
    \prediction_reg[1]_i_4__0_0 ,
    kde_prob_night_mean,
    \prediction_reg[1]_6 ,
    \prediction[1]_i_24__4_0 ,
    \prediction[1]_i_24__4_1 ,
    dist_to_centroid_mean,
    start,
    \prediction[1]_i_5__4_0 ,
    \prediction_reg[1]_7 );
  output [0:0]t_done;
  output kde_prob_mean_7_sp_1;
  output \kde_prob_mean[7]_0 ;
  output kde_prob_mean_3_sp_1;
  output accelerate_7_sp_1;
  output kde_prob_night_mean_8_sp_1;
  output dist_to_centroid_mean_12_sp_1;
  output kde_prob_night_mean_2_sp_1;
  output \prediction_reg[1]_0 ;
  output \prediction_reg[0]_0 ;
  input \prediction_reg[1]_1 ;
  input clk;
  input \prediction_reg[0]_1 ;
  input \prediction_reg[1]_2 ;
  input \prediction_reg[1]_3 ;
  input [15:0]kde_prob_mean;
  input \prediction_reg[1]_4 ;
  input \prediction_reg[1]_5 ;
  input [15:0]mean_speed;
  input [9:0]step_median;
  input \prediction[1]_i_23__0_0 ;
  input \prediction[1]_i_23__0_1 ;
  input \prediction[1]_i_9__3_0 ;
  input \prediction[1]_i_14__0_0 ;
  input \prediction[1]_i_25__0_0 ;
  input [15:0]accelerate;
  input \prediction_reg[1]_i_4__0_0 ;
  input [15:0]kde_prob_night_mean;
  input \prediction_reg[1]_6 ;
  input \prediction[1]_i_24__4_0 ;
  input \prediction[1]_i_24__4_1 ;
  input [15:0]dist_to_centroid_mean;
  input [0:0]start;
  input \prediction[1]_i_5__4_0 ;
  input \prediction_reg[1]_7 ;

  wire [15:0]accelerate;
  wire accelerate_7_sn_1;
  wire clk;
  wire [15:0]dist_to_centroid_mean;
  wire dist_to_centroid_mean_12_sn_1;
  wire done_i_1__1_n_0;
  wire [15:0]kde_prob_mean;
  wire \kde_prob_mean[7]_0 ;
  wire kde_prob_mean_3_sn_1;
  wire kde_prob_mean_7_sn_1;
  wire [15:0]kde_prob_night_mean;
  wire kde_prob_night_mean_2_sn_1;
  wire kde_prob_night_mean_8_sn_1;
  wire [15:0]mean_speed;
  wire \prediction[0]_i_1__3_n_0 ;
  wire \prediction[1]_i_10__2_n_0 ;
  wire \prediction[1]_i_11__3_n_0 ;
  wire \prediction[1]_i_12__5_n_0 ;
  wire \prediction[1]_i_14__0_0 ;
  wire \prediction[1]_i_14__0_n_0 ;
  wire \prediction[1]_i_15_n_0 ;
  wire \prediction[1]_i_16__3_n_0 ;
  wire \prediction[1]_i_18__2_n_0 ;
  wire \prediction[1]_i_19__2_n_0 ;
  wire \prediction[1]_i_1__0_n_0 ;
  wire \prediction[1]_i_20__3_n_0 ;
  wire \prediction[1]_i_23__0_0 ;
  wire \prediction[1]_i_23__0_1 ;
  wire \prediction[1]_i_24__4_0 ;
  wire \prediction[1]_i_24__4_1 ;
  wire \prediction[1]_i_24__4_n_0 ;
  wire \prediction[1]_i_25__0_0 ;
  wire \prediction[1]_i_25__0_n_0 ;
  wire \prediction[1]_i_26__2_n_0 ;
  wire \prediction[1]_i_27_n_0 ;
  wire \prediction[1]_i_28__2_n_0 ;
  wire \prediction[1]_i_29_n_0 ;
  wire \prediction[1]_i_30_n_0 ;
  wire \prediction[1]_i_31_n_0 ;
  wire \prediction[1]_i_33__1_n_0 ;
  wire \prediction[1]_i_34__2_n_0 ;
  wire \prediction[1]_i_36_n_0 ;
  wire \prediction[1]_i_37_n_0 ;
  wire \prediction[1]_i_38__2_n_0 ;
  wire \prediction[1]_i_39_n_0 ;
  wire \prediction[1]_i_3__1_n_0 ;
  wire \prediction[1]_i_40__0_n_0 ;
  wire \prediction[1]_i_41__1_n_0 ;
  wire \prediction[1]_i_42_n_0 ;
  wire \prediction[1]_i_43_n_0 ;
  wire \prediction[1]_i_44_n_0 ;
  wire \prediction[1]_i_45_n_0 ;
  wire \prediction[1]_i_46_n_0 ;
  wire \prediction[1]_i_47_n_0 ;
  wire \prediction[1]_i_48_n_0 ;
  wire \prediction[1]_i_49_n_0 ;
  wire \prediction[1]_i_5__4_0 ;
  wire \prediction[1]_i_5__4_n_0 ;
  wire \prediction[1]_i_7__3_n_0 ;
  wire \prediction[1]_i_8__4_n_0 ;
  wire \prediction[1]_i_9__3_0 ;
  wire \prediction[1]_i_9__3_n_0 ;
  wire \prediction_reg[0]_0 ;
  wire \prediction_reg[0]_1 ;
  wire \prediction_reg[1]_0 ;
  wire \prediction_reg[1]_1 ;
  wire \prediction_reg[1]_2 ;
  wire \prediction_reg[1]_3 ;
  wire \prediction_reg[1]_4 ;
  wire \prediction_reg[1]_5 ;
  wire \prediction_reg[1]_6 ;
  wire \prediction_reg[1]_7 ;
  wire \prediction_reg[1]_i_4__0_0 ;
  wire \prediction_reg[1]_i_4__0_n_0 ;
  wire \prediction_reg[1]_i_6_n_0 ;
  wire [0:0]start;
  wire [9:0]step_median;
  wire [0:0]t_done;
  wire tree_out3_out;

  assign accelerate_7_sp_1 = accelerate_7_sn_1;
  assign dist_to_centroid_mean_12_sp_1 = dist_to_centroid_mean_12_sn_1;
  assign kde_prob_mean_3_sp_1 = kde_prob_mean_3_sn_1;
  assign kde_prob_mean_7_sp_1 = kde_prob_mean_7_sn_1;
  assign kde_prob_night_mean_2_sp_1 = kde_prob_night_mean_2_sn_1;
  assign kde_prob_night_mean_8_sp_1 = kde_prob_night_mean_8_sn_1;
  LUT2 #(
    .INIT(4'hD)) 
    done_i_1__1
       (.I0(start),
        .I1(t_done),
        .O(done_i_1__1_n_0));
  FDRE done_reg
       (.C(clk),
        .CE(1'b1),
        .D(done_i_1__1_n_0),
        .Q(t_done),
        .R(\prediction_reg[1]_1 ));
  LUT6 #(
    .INIT(64'h111D111D111DDD1D)) 
    \prediction[0]_i_1__3 
       (.I0(\prediction_reg[1]_i_6_n_0 ),
        .I1(\prediction[1]_i_5__4_n_0 ),
        .I2(\prediction_reg[1]_i_4__0_n_0 ),
        .I3(\prediction_reg[0]_1 ),
        .I4(\prediction[1]_i_3__1_n_0 ),
        .I5(kde_prob_mean_7_sn_1),
        .O(\prediction[0]_i_1__3_n_0 ));
  LUT2 #(
    .INIT(4'h8)) 
    \prediction[0]_i_22 
       (.I0(accelerate[7]),
        .I1(accelerate[8]),
        .O(accelerate_7_sn_1));
  LUT6 #(
    .INIT(64'h0000000000005557)) 
    \prediction[0]_i_7__0 
       (.I0(kde_prob_mean[3]),
        .I1(kde_prob_mean[2]),
        .I2(kde_prob_mean[1]),
        .I3(kde_prob_mean[0]),
        .I4(kde_prob_mean[5]),
        .I5(kde_prob_mean[4]),
        .O(kde_prob_mean_3_sn_1));
  LUT6 #(
    .INIT(64'h00000000000000F7)) 
    \prediction[1]_i_10__2 
       (.I0(accelerate[10]),
        .I1(\prediction_reg[1]_i_4__0_0 ),
        .I2(\prediction[1]_i_18__2_n_0 ),
        .I3(accelerate[14]),
        .I4(accelerate[15]),
        .I5(accelerate[13]),
        .O(\prediction[1]_i_10__2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFBA00FFFFFFFF)) 
    \prediction[1]_i_11__3 
       (.I0(accelerate[13]),
        .I1(\prediction[1]_i_19__2_n_0 ),
        .I2(accelerate[12]),
        .I3(accelerate[14]),
        .I4(accelerate[15]),
        .I5(\prediction[1]_i_20__3_n_0 ),
        .O(\prediction[1]_i_11__3_n_0 ));
  LUT6 #(
    .INIT(64'h777F7F7F7F7F7F7F)) 
    \prediction[1]_i_12__5 
       (.I0(kde_prob_night_mean[7]),
        .I1(kde_prob_night_mean[6]),
        .I2(\prediction[1]_i_5__4_0 ),
        .I3(kde_prob_night_mean[3]),
        .I4(kde_prob_night_mean[2]),
        .I5(kde_prob_night_mean[1]),
        .O(\prediction[1]_i_12__5_n_0 ));
  LUT6 #(
    .INIT(64'h88888888B8BBB8B8)) 
    \prediction[1]_i_14__0 
       (.I0(tree_out3_out),
        .I1(\prediction[1]_i_24__4_n_0 ),
        .I2(mean_speed[15]),
        .I3(\prediction[1]_i_25__0_n_0 ),
        .I4(mean_speed[14]),
        .I5(\prediction[1]_i_26__2_n_0 ),
        .O(\prediction[1]_i_14__0_n_0 ));
  LUT6 #(
    .INIT(64'h888888888B888B8B)) 
    \prediction[1]_i_15 
       (.I0(\prediction[1]_i_27_n_0 ),
        .I1(\prediction[1]_i_28__2_n_0 ),
        .I2(kde_prob_mean[15]),
        .I3(\prediction[1]_i_29_n_0 ),
        .I4(kde_prob_mean[14]),
        .I5(\prediction[1]_i_30_n_0 ),
        .O(\prediction[1]_i_15_n_0 ));
  LUT6 #(
    .INIT(64'h10115555FFFFFFFF)) 
    \prediction[1]_i_16__3 
       (.I0(mean_speed[6]),
        .I1(mean_speed[4]),
        .I2(\prediction[1]_i_31_n_0 ),
        .I3(mean_speed[3]),
        .I4(mean_speed[5]),
        .I5(\prediction[1]_i_9__3_0 ),
        .O(\prediction[1]_i_16__3_n_0 ));
  LUT6 #(
    .INIT(64'h0000000001FFFFFF)) 
    \prediction[1]_i_18__2 
       (.I0(accelerate[3]),
        .I1(accelerate[5]),
        .I2(accelerate[4]),
        .I3(accelerate[6]),
        .I4(accelerate_7_sn_1),
        .I5(accelerate[9]),
        .O(\prediction[1]_i_18__2_n_0 ));
  LUT6 #(
    .INIT(64'h00000000555577F7)) 
    \prediction[1]_i_19__2 
       (.I0(accelerate[10]),
        .I1(accelerate_7_sn_1),
        .I2(\prediction[1]_i_33__1_n_0 ),
        .I3(accelerate[6]),
        .I4(accelerate[9]),
        .I5(accelerate[11]),
        .O(\prediction[1]_i_19__2_n_0 ));
  LUT3 #(
    .INIT(8'h01)) 
    \prediction[1]_i_19__5 
       (.I0(kde_prob_night_mean[2]),
        .I1(kde_prob_night_mean[1]),
        .I2(kde_prob_night_mean[0]),
        .O(kde_prob_night_mean_2_sn_1));
  LUT6 #(
    .INIT(64'hEFE0FFFFEFE00000)) 
    \prediction[1]_i_1__0 
       (.I0(kde_prob_mean_7_sn_1),
        .I1(\prediction[1]_i_3__1_n_0 ),
        .I2(\prediction_reg[0]_1 ),
        .I3(\prediction_reg[1]_i_4__0_n_0 ),
        .I4(\prediction[1]_i_5__4_n_0 ),
        .I5(\prediction_reg[1]_i_6_n_0 ),
        .O(\prediction[1]_i_1__0_n_0 ));
  LUT6 #(
    .INIT(64'h00000000555577F7)) 
    \prediction[1]_i_20__3 
       (.I0(kde_prob_night_mean[14]),
        .I1(kde_prob_night_mean[10]),
        .I2(\prediction[1]_i_34__2_n_0 ),
        .I3(kde_prob_night_mean_8_sn_1),
        .I4(\prediction_reg[1]_6 ),
        .I5(kde_prob_night_mean[15]),
        .O(\prediction[1]_i_20__3_n_0 ));
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[1]_i_22__4 
       (.I0(dist_to_centroid_mean[12]),
        .I1(dist_to_centroid_mean[13]),
        .O(dist_to_centroid_mean_12_sn_1));
  LUT6 #(
    .INIT(64'h0000000001005555)) 
    \prediction[1]_i_23__0 
       (.I0(kde_prob_mean[15]),
        .I1(kde_prob_mean[12]),
        .I2(kde_prob_mean[13]),
        .I3(\prediction[1]_i_36_n_0 ),
        .I4(kde_prob_mean[14]),
        .I5(\prediction[1]_i_37_n_0 ),
        .O(tree_out3_out));
  LUT6 #(
    .INIT(64'h000000007F7FFF7F)) 
    \prediction[1]_i_24__4 
       (.I0(kde_prob_night_mean[12]),
        .I1(kde_prob_night_mean[14]),
        .I2(kde_prob_night_mean[13]),
        .I3(\prediction[1]_i_38__2_n_0 ),
        .I4(kde_prob_night_mean[11]),
        .I5(kde_prob_night_mean[15]),
        .O(\prediction[1]_i_24__4_n_0 ));
  LUT6 #(
    .INIT(64'h000000000DFFFFFF)) 
    \prediction[1]_i_25__0 
       (.I0(mean_speed[8]),
        .I1(\prediction[1]_i_39_n_0 ),
        .I2(mean_speed[9]),
        .I3(mean_speed[10]),
        .I4(mean_speed[11]),
        .I5(\prediction[1]_i_14__0_0 ),
        .O(\prediction[1]_i_25__0_n_0 ));
  LUT6 #(
    .INIT(64'h000000005555FF7F)) 
    \prediction[1]_i_26__2 
       (.I0(dist_to_centroid_mean[14]),
        .I1(dist_to_centroid_mean[10]),
        .I2(dist_to_centroid_mean[11]),
        .I3(\prediction[1]_i_40__0_n_0 ),
        .I4(dist_to_centroid_mean_12_sn_1),
        .I5(dist_to_centroid_mean[15]),
        .O(\prediction[1]_i_26__2_n_0 ));
  LUT6 #(
    .INIT(64'h00000000555557FF)) 
    \prediction[1]_i_27 
       (.I0(\prediction_reg[1]_3 ),
        .I1(kde_prob_mean[5]),
        .I2(kde_prob_mean[6]),
        .I3(\kde_prob_mean[7]_0 ),
        .I4(\prediction_reg[1]_4 ),
        .I5(\prediction_reg[1]_2 ),
        .O(\prediction[1]_i_27_n_0 ));
  LUT6 #(
    .INIT(64'h00000000555555F7)) 
    \prediction[1]_i_28__2 
       (.I0(kde_prob_night_mean[14]),
        .I1(kde_prob_night_mean[11]),
        .I2(\prediction[1]_i_41__1_n_0 ),
        .I3(kde_prob_night_mean[13]),
        .I4(kde_prob_night_mean[12]),
        .I5(kde_prob_night_mean[15]),
        .O(\prediction[1]_i_28__2_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FFFF7FFF)) 
    \prediction[1]_i_29 
       (.I0(kde_prob_mean[6]),
        .I1(kde_prob_mean[9]),
        .I2(kde_prob_mean[10]),
        .I3(\kde_prob_mean[7]_0 ),
        .I4(kde_prob_mean_3_sn_1),
        .I5(\prediction[1]_i_42_n_0 ),
        .O(\prediction[1]_i_29_n_0 ));
  LUT6 #(
    .INIT(64'h000000005555FF7F)) 
    \prediction[1]_i_2__0 
       (.I0(\prediction_reg[1]_3 ),
        .I1(kde_prob_mean[7]),
        .I2(kde_prob_mean[8]),
        .I3(\prediction[1]_i_7__3_n_0 ),
        .I4(\prediction_reg[1]_4 ),
        .I5(\prediction_reg[1]_2 ),
        .O(kde_prob_mean_7_sn_1));
  LUT5 #(
    .INIT(32'h0000FF7F)) 
    \prediction[1]_i_30 
       (.I0(mean_speed[12]),
        .I1(mean_speed[14]),
        .I2(mean_speed[13]),
        .I3(\prediction[1]_i_43_n_0 ),
        .I4(mean_speed[15]),
        .O(\prediction[1]_i_30_n_0 ));
  LUT3 #(
    .INIT(8'h01)) 
    \prediction[1]_i_31 
       (.I0(mean_speed[2]),
        .I1(mean_speed[1]),
        .I2(mean_speed[0]),
        .O(\prediction[1]_i_31_n_0 ));
  LUT6 #(
    .INIT(64'h01555555FFFFFFFF)) 
    \prediction[1]_i_33__1 
       (.I0(accelerate[4]),
        .I1(accelerate[0]),
        .I2(accelerate[1]),
        .I3(accelerate[3]),
        .I4(accelerate[2]),
        .I5(accelerate[5]),
        .O(\prediction[1]_i_33__1_n_0 ));
  LUT6 #(
    .INIT(64'h777F7777777F777F)) 
    \prediction[1]_i_34__2 
       (.I0(kde_prob_night_mean[7]),
        .I1(kde_prob_night_mean[6]),
        .I2(kde_prob_night_mean[4]),
        .I3(kde_prob_night_mean[5]),
        .I4(kde_prob_night_mean_2_sn_1),
        .I5(kde_prob_night_mean[3]),
        .O(\prediction[1]_i_34__2_n_0 ));
  LUT6 #(
    .INIT(64'h0100FFFFFFFFFFFF)) 
    \prediction[1]_i_36 
       (.I0(kde_prob_mean[7]),
        .I1(kde_prob_mean[9]),
        .I2(kde_prob_mean[8]),
        .I3(\prediction[1]_i_44_n_0 ),
        .I4(kde_prob_mean[11]),
        .I5(kde_prob_mean[10]),
        .O(\prediction[1]_i_36_n_0 ));
  LUT6 #(
    .INIT(64'h00000000555555F7)) 
    \prediction[1]_i_37 
       (.I0(\prediction[1]_i_45_n_0 ),
        .I1(step_median[6]),
        .I2(\prediction[1]_i_46_n_0 ),
        .I3(\prediction[1]_i_23__0_0 ),
        .I4(step_median[7]),
        .I5(\prediction[1]_i_23__0_1 ),
        .O(\prediction[1]_i_37_n_0 ));
  LUT6 #(
    .INIT(64'h45555555FFFFFFFF)) 
    \prediction[1]_i_38__2 
       (.I0(kde_prob_night_mean[8]),
        .I1(\prediction[1]_i_24__4_0 ),
        .I2(kde_prob_night_mean[6]),
        .I3(kde_prob_night_mean[7]),
        .I4(kde_prob_night_mean[5]),
        .I5(\prediction[1]_i_24__4_1 ),
        .O(\prediction[1]_i_38__2_n_0 ));
  LUT6 #(
    .INIT(64'h00000000000000F7)) 
    \prediction[1]_i_39 
       (.I0(mean_speed[3]),
        .I1(mean_speed[4]),
        .I2(\prediction[1]_i_25__0_0 ),
        .I3(mean_speed[6]),
        .I4(mean_speed[7]),
        .I5(mean_speed[5]),
        .O(\prediction[1]_i_39_n_0 ));
  LUT6 #(
    .INIT(64'hEEEEEEEEEAEAAAEA)) 
    \prediction[1]_i_3__1 
       (.I0(\prediction_reg[1]_2 ),
        .I1(\prediction_reg[1]_3 ),
        .I2(\kde_prob_mean[7]_0 ),
        .I3(\prediction[1]_i_8__4_n_0 ),
        .I4(kde_prob_mean[6]),
        .I5(\prediction_reg[1]_4 ),
        .O(\prediction[1]_i_3__1_n_0 ));
  LUT6 #(
    .INIT(64'h000000000000FF7F)) 
    \prediction[1]_i_40__0 
       (.I0(dist_to_centroid_mean[5]),
        .I1(dist_to_centroid_mean[7]),
        .I2(dist_to_centroid_mean[6]),
        .I3(\prediction[1]_i_47_n_0 ),
        .I4(dist_to_centroid_mean[9]),
        .I5(dist_to_centroid_mean[8]),
        .O(\prediction[1]_i_40__0_n_0 ));
  LUT6 #(
    .INIT(64'h000000000000555D)) 
    \prediction[1]_i_41__1 
       (.I0(kde_prob_night_mean[8]),
        .I1(\prediction[1]_i_48_n_0 ),
        .I2(kde_prob_night_mean[7]),
        .I3(kde_prob_night_mean[6]),
        .I4(kde_prob_night_mean[10]),
        .I5(kde_prob_night_mean[9]),
        .O(\prediction[1]_i_41__1_n_0 ));
  LUT3 #(
    .INIT(8'hFE)) 
    \prediction[1]_i_42 
       (.I0(kde_prob_mean[11]),
        .I1(kde_prob_mean[13]),
        .I2(kde_prob_mean[12]),
        .O(\prediction[1]_i_42_n_0 ));
  LUT6 #(
    .INIT(64'h000000005555DFFF)) 
    \prediction[1]_i_43 
       (.I0(mean_speed[10]),
        .I1(\prediction[1]_i_49_n_0 ),
        .I2(mean_speed[7]),
        .I3(mean_speed[8]),
        .I4(mean_speed[9]),
        .I5(mean_speed[11]),
        .O(\prediction[1]_i_43_n_0 ));
  LUT6 #(
    .INIT(64'h01010111FFFFFFFF)) 
    \prediction[1]_i_44 
       (.I0(kde_prob_mean[4]),
        .I1(kde_prob_mean[5]),
        .I2(kde_prob_mean[3]),
        .I3(kde_prob_mean[2]),
        .I4(kde_prob_mean[1]),
        .I5(kde_prob_mean[6]),
        .O(\prediction[1]_i_44_n_0 ));
  LUT2 #(
    .INIT(4'h8)) 
    \prediction[1]_i_45 
       (.I0(step_median[8]),
        .I1(step_median[9]),
        .O(\prediction[1]_i_45_n_0 ));
  LUT6 #(
    .INIT(64'h0000000055557FFF)) 
    \prediction[1]_i_46 
       (.I0(step_median[4]),
        .I1(step_median[0]),
        .I2(step_median[1]),
        .I3(step_median[2]),
        .I4(step_median[3]),
        .I5(step_median[5]),
        .O(\prediction[1]_i_46_n_0 ));
  LUT5 #(
    .INIT(32'h0000777F)) 
    \prediction[1]_i_47 
       (.I0(dist_to_centroid_mean[2]),
        .I1(dist_to_centroid_mean[3]),
        .I2(dist_to_centroid_mean[1]),
        .I3(dist_to_centroid_mean[0]),
        .I4(dist_to_centroid_mean[4]),
        .O(\prediction[1]_i_47_n_0 ));
  LUT6 #(
    .INIT(64'h01115555FFFFFFFF)) 
    \prediction[1]_i_48 
       (.I0(kde_prob_night_mean[4]),
        .I1(kde_prob_night_mean[2]),
        .I2(kde_prob_night_mean[1]),
        .I3(kde_prob_night_mean[0]),
        .I4(kde_prob_night_mean[3]),
        .I5(kde_prob_night_mean[5]),
        .O(\prediction[1]_i_48_n_0 ));
  LUT6 #(
    .INIT(64'h0000000015FFFFFF)) 
    \prediction[1]_i_49 
       (.I0(mean_speed[3]),
        .I1(mean_speed[2]),
        .I2(mean_speed[1]),
        .I3(mean_speed[4]),
        .I4(mean_speed[5]),
        .I5(mean_speed[6]),
        .O(\prediction[1]_i_49_n_0 ));
  LUT2 #(
    .INIT(4'h8)) 
    \prediction[1]_i_5__0 
       (.I0(kde_prob_mean[7]),
        .I1(kde_prob_mean[8]),
        .O(\kde_prob_mean[7]_0 ));
  LUT6 #(
    .INIT(64'h00000000555577F7)) 
    \prediction[1]_i_5__4 
       (.I0(kde_prob_night_mean[14]),
        .I1(kde_prob_night_mean[10]),
        .I2(\prediction[1]_i_12__5_n_0 ),
        .I3(kde_prob_night_mean_8_sn_1),
        .I4(\prediction_reg[1]_6 ),
        .I5(kde_prob_night_mean[15]),
        .O(\prediction[1]_i_5__4_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000001FFF)) 
    \prediction[1]_i_7__3 
       (.I0(kde_prob_mean[1]),
        .I1(kde_prob_mean[2]),
        .I2(kde_prob_mean[3]),
        .I3(kde_prob_mean[4]),
        .I4(kde_prob_mean[6]),
        .I5(kde_prob_mean[5]),
        .O(\prediction[1]_i_7__3_n_0 ));
  LUT6 #(
    .INIT(64'h7FFFFFFFFFFFFFFF)) 
    \prediction[1]_i_8__4 
       (.I0(kde_prob_mean[3]),
        .I1(kde_prob_mean[2]),
        .I2(kde_prob_mean[5]),
        .I3(kde_prob_mean[4]),
        .I4(kde_prob_mean[1]),
        .I5(kde_prob_mean[0]),
        .O(\prediction[1]_i_8__4_n_0 ));
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[1]_i_8__5 
       (.I0(kde_prob_night_mean[8]),
        .I1(kde_prob_night_mean[9]),
        .O(kde_prob_night_mean_8_sn_1));
  LUT6 #(
    .INIT(64'h00000000555577F7)) 
    \prediction[1]_i_9__3 
       (.I0(mean_speed[14]),
        .I1(mean_speed[12]),
        .I2(\prediction[1]_i_16__3_n_0 ),
        .I3(mean_speed[11]),
        .I4(mean_speed[13]),
        .I5(mean_speed[15]),
        .O(\prediction[1]_i_9__3_n_0 ));
  FDRE \prediction_reg[0] 
       (.C(clk),
        .CE(\prediction_reg[1]_7 ),
        .D(\prediction[0]_i_1__3_n_0 ),
        .Q(\prediction_reg[0]_0 ),
        .R(\prediction_reg[1]_1 ));
  FDRE \prediction_reg[1] 
       (.C(clk),
        .CE(\prediction_reg[1]_7 ),
        .D(\prediction[1]_i_1__0_n_0 ),
        .Q(\prediction_reg[1]_0 ),
        .R(\prediction_reg[1]_1 ));
  MUXF7 \prediction_reg[1]_i_4__0 
       (.I0(\prediction[1]_i_10__2_n_0 ),
        .I1(\prediction[1]_i_11__3_n_0 ),
        .O(\prediction_reg[1]_i_4__0_n_0 ),
        .S(\prediction[1]_i_9__3_n_0 ));
  MUXF7 \prediction_reg[1]_i_6 
       (.I0(\prediction[1]_i_14__0_n_0 ),
        .I1(\prediction[1]_i_15_n_0 ),
        .O(\prediction_reg[1]_i_6_n_0 ),
        .S(\prediction_reg[1]_5 ));
endmodule

(* ORIG_REF_NAME = "decision_tree_3" *) 
module design_1_random_forest_elepha_0_0_decision_tree_3
   (kde_prob_mean_13_sp_1,
    turning_angle_median_13_sp_1,
    kde_prob_mean_9_sp_1,
    kde_prob_mean_5_sp_1,
    kde_prob_mean_0_sp_1,
    turning_angle_median_10_sp_1,
    turning_angle_median_6_sp_1,
    D,
    done_reg_0,
    done_reg_1,
    \prediction_reg[1]_0 ,
    clk,
    \prediction_reg[0]_0 ,
    \prediction_reg[1]_1 ,
    dist_to_centroid_mean,
    kde_prob_mean,
    kde_prob_night_mean,
    \prediction[1]_i_4__0_0 ,
    \prediction[1]_i_4__0_1 ,
    \prediction[1]_i_12_0 ,
    \prediction_reg[0]_1 ,
    \prediction_reg[0]_2 ,
    \prediction_reg[0]_3 ,
    \prediction_reg[0]_4 ,
    \prediction[0]_i_2__1_0 ,
    turning_angle_max,
    mean_speed,
    \prediction_reg[0]_5 ,
    \prediction[1]_i_7__0_0 ,
    turning_angle_median,
    accelerate,
    start,
    done_reg_2,
    \result_reg[0] ,
    \result_reg[0]_0 ,
    \result_reg[0]_1 ,
    p_3_in,
    \result[1]_i_2_0 ,
    \result[1]_i_2_1 ,
    \result[1]_i_2_2 ,
    \result[1]_i_2_3 ,
    \prediction_reg[1]_2 );
  output kde_prob_mean_13_sp_1;
  output turning_angle_median_13_sp_1;
  output kde_prob_mean_9_sp_1;
  output kde_prob_mean_5_sp_1;
  output kde_prob_mean_0_sp_1;
  output turning_angle_median_10_sp_1;
  output turning_angle_median_6_sp_1;
  output [1:0]D;
  output done_reg_0;
  input [2:0]done_reg_1;
  input \prediction_reg[1]_0 ;
  input clk;
  input \prediction_reg[0]_0 ;
  input \prediction_reg[1]_1 ;
  input [15:0]dist_to_centroid_mean;
  input [15:0]kde_prob_mean;
  input [8:0]kde_prob_night_mean;
  input \prediction[1]_i_4__0_0 ;
  input \prediction[1]_i_4__0_1 ;
  input \prediction[1]_i_12_0 ;
  input \prediction_reg[0]_1 ;
  input \prediction_reg[0]_2 ;
  input \prediction_reg[0]_3 ;
  input \prediction_reg[0]_4 ;
  input \prediction[0]_i_2__1_0 ;
  input [10:0]turning_angle_max;
  input [15:0]mean_speed;
  input \prediction_reg[0]_5 ;
  input \prediction[1]_i_7__0_0 ;
  input [14:0]turning_angle_median;
  input [11:0]accelerate;
  input [0:0]start;
  input done_reg_2;
  input \result_reg[0] ;
  input \result_reg[0]_0 ;
  input \result_reg[0]_1 ;
  input p_3_in;
  input \result[1]_i_2_0 ;
  input \result[1]_i_2_1 ;
  input \result[1]_i_2_2 ;
  input \result[1]_i_2_3 ;
  input \prediction_reg[1]_2 ;

  wire [1:0]D;
  wire [11:0]accelerate;
  wire clk;
  wire [15:0]dist_to_centroid_mean;
  wire done_i_1__2_n_0;
  wire done_reg_0;
  wire [2:0]done_reg_1;
  wire done_reg_2;
  wire [15:0]kde_prob_mean;
  wire kde_prob_mean_0_sn_1;
  wire kde_prob_mean_13_sn_1;
  wire kde_prob_mean_5_sn_1;
  wire kde_prob_mean_9_sn_1;
  wire [8:0]kde_prob_night_mean;
  wire [15:0]mean_speed;
  wire p_3_in;
  wire \prediction[0]_i_10_n_0 ;
  wire \prediction[0]_i_11__1_n_0 ;
  wire \prediction[0]_i_13__1_n_0 ;
  wire \prediction[0]_i_14__2_n_0 ;
  wire \prediction[0]_i_1__2_n_0 ;
  wire \prediction[0]_i_20__2_n_0 ;
  wire \prediction[0]_i_22__1_n_0 ;
  wire \prediction[0]_i_2__1_0 ;
  wire \prediction[0]_i_2__1_n_0 ;
  wire \prediction[0]_i_4__1_n_0 ;
  wire \prediction[0]_i_5__0_n_0 ;
  wire \prediction[0]_i_7__1_n_0 ;
  wire \prediction[1]_i_10__0_n_0 ;
  wire \prediction[1]_i_11__5_n_0 ;
  wire \prediction[1]_i_12_0 ;
  wire \prediction[1]_i_13__0_n_0 ;
  wire \prediction[1]_i_14__4_n_0 ;
  wire \prediction[1]_i_15__2_n_0 ;
  wire \prediction[1]_i_16__2_n_0 ;
  wire \prediction[1]_i_17__4_n_0 ;
  wire \prediction[1]_i_18__4_n_0 ;
  wire \prediction[1]_i_19_n_0 ;
  wire \prediction[1]_i_20__2_n_0 ;
  wire \prediction[1]_i_21__1_n_0 ;
  wire \prediction[1]_i_24__3_n_0 ;
  wire \prediction[1]_i_25__2_n_0 ;
  wire \prediction[1]_i_2__3_n_0 ;
  wire \prediction[1]_i_3__0_n_0 ;
  wire \prediction[1]_i_4__0_0 ;
  wire \prediction[1]_i_4__0_1 ;
  wire \prediction[1]_i_4__0_n_0 ;
  wire \prediction[1]_i_5__2_n_0 ;
  wire \prediction[1]_i_6__3_n_0 ;
  wire \prediction[1]_i_7__0_0 ;
  wire \prediction[1]_i_7__0_n_0 ;
  wire \prediction[1]_i_8__3_n_0 ;
  wire \prediction[1]_i_9_n_0 ;
  wire \prediction_reg[0]_0 ;
  wire \prediction_reg[0]_1 ;
  wire \prediction_reg[0]_2 ;
  wire \prediction_reg[0]_3 ;
  wire \prediction_reg[0]_4 ;
  wire \prediction_reg[0]_5 ;
  wire \prediction_reg[0]_i_3_n_0 ;
  wire \prediction_reg[1]_0 ;
  wire \prediction_reg[1]_1 ;
  wire \prediction_reg[1]_2 ;
  wire \prediction_reg[1]_i_1__0_n_0 ;
  wire \prediction_reg_n_0_[0] ;
  wire \prediction_reg_n_0_[1] ;
  wire \result[1]_i_2_0 ;
  wire \result[1]_i_2_1 ;
  wire \result[1]_i_2_2 ;
  wire \result[1]_i_2_3 ;
  wire \result[1]_i_2_n_0 ;
  wire \result[1]_i_3_n_0 ;
  wire \result[1]_i_8_n_0 ;
  wire \result_reg[0] ;
  wire \result_reg[0]_0 ;
  wire \result_reg[0]_1 ;
  wire [0:0]start;
  wire [2:2]t_done;
  wire tree_out;
  wire [10:0]turning_angle_max;
  wire [14:0]turning_angle_median;
  wire turning_angle_median_10_sn_1;
  wire turning_angle_median_13_sn_1;
  wire turning_angle_median_6_sn_1;

  assign kde_prob_mean_0_sp_1 = kde_prob_mean_0_sn_1;
  assign kde_prob_mean_13_sp_1 = kde_prob_mean_13_sn_1;
  assign kde_prob_mean_5_sp_1 = kde_prob_mean_5_sn_1;
  assign kde_prob_mean_9_sp_1 = kde_prob_mean_9_sn_1;
  assign turning_angle_median_10_sp_1 = turning_angle_median_10_sn_1;
  assign turning_angle_median_13_sp_1 = turning_angle_median_13_sn_1;
  assign turning_angle_median_6_sp_1 = turning_angle_median_6_sn_1;
  LUT5 #(
    .INIT(32'h00008000)) 
    done_i_1
       (.I0(t_done),
        .I1(done_reg_1[2]),
        .I2(done_reg_1[0]),
        .I3(done_reg_1[1]),
        .I4(done_reg_2),
        .O(done_reg_0));
  LUT2 #(
    .INIT(4'hD)) 
    done_i_1__2
       (.I0(start),
        .I1(t_done),
        .O(done_i_1__2_n_0));
  FDRE done_reg
       (.C(clk),
        .CE(1'b1),
        .D(done_i_1__2_n_0),
        .Q(t_done),
        .R(\prediction_reg[1]_0 ));
  LUT6 #(
    .INIT(64'hAAA8AAAAAAA8AAA8)) 
    \prediction[0]_i_10 
       (.I0(\prediction[0]_i_2__1_0 ),
        .I1(turning_angle_max[8]),
        .I2(turning_angle_max[10]),
        .I3(turning_angle_max[9]),
        .I4(\prediction[0]_i_20__2_n_0 ),
        .I5(turning_angle_max[7]),
        .O(\prediction[0]_i_10_n_0 ));
  LUT6 #(
    .INIT(64'h45444545FFFFFFFF)) 
    \prediction[0]_i_11__1 
       (.I0(turning_angle_median[11]),
        .I1(turning_angle_median_10_sn_1),
        .I2(turning_angle_median[8]),
        .I3(\prediction[1]_i_15__2_n_0 ),
        .I4(turning_angle_median[7]),
        .I5(turning_angle_median_13_sn_1),
        .O(\prediction[0]_i_11__1_n_0 ));
  LUT2 #(
    .INIT(4'h1)) 
    \prediction[0]_i_13__1 
       (.I0(accelerate[1]),
        .I1(accelerate[0]),
        .O(\prediction[0]_i_13__1_n_0 ));
  LUT6 #(
    .INIT(64'h8000000000000000)) 
    \prediction[0]_i_14__2 
       (.I0(accelerate[7]),
        .I1(accelerate[6]),
        .I2(accelerate[10]),
        .I3(accelerate[11]),
        .I4(accelerate[8]),
        .I5(accelerate[9]),
        .O(\prediction[0]_i_14__2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT3 #(
    .INIT(8'h1F)) 
    \prediction[0]_i_18 
       (.I0(kde_prob_mean[0]),
        .I1(kde_prob_mean[1]),
        .I2(kde_prob_mean[2]),
        .O(kde_prob_mean_0_sn_1));
  LUT6 #(
    .INIT(64'h1D001D1D1DFF1D1D)) 
    \prediction[0]_i_1__2 
       (.I0(\prediction[0]_i_2__1_n_0 ),
        .I1(\prediction_reg[0]_0 ),
        .I2(\prediction_reg[0]_i_3_n_0 ),
        .I3(\prediction[0]_i_4__1_n_0 ),
        .I4(\prediction[0]_i_5__0_n_0 ),
        .I5(\prediction[1]_i_4__0_n_0 ),
        .O(\prediction[0]_i_1__2_n_0 ));
  LUT6 #(
    .INIT(64'h000000000000FF7F)) 
    \prediction[0]_i_20__2 
       (.I0(turning_angle_max[0]),
        .I1(turning_angle_max[2]),
        .I2(turning_angle_max[1]),
        .I3(\prediction[0]_i_22__1_n_0 ),
        .I4(turning_angle_max[6]),
        .I5(turning_angle_max[5]),
        .O(\prediction[0]_i_20__2_n_0 ));
  LUT2 #(
    .INIT(4'h7)) 
    \prediction[0]_i_21__1 
       (.I0(turning_angle_median[9]),
        .I1(turning_angle_median[10]),
        .O(turning_angle_median_10_sn_1));
  LUT2 #(
    .INIT(4'h7)) 
    \prediction[0]_i_22__1 
       (.I0(turning_angle_max[3]),
        .I1(turning_angle_max[4]),
        .O(\prediction[0]_i_22__1_n_0 ));
  LUT6 #(
    .INIT(64'h04FF04FF04FF0400)) 
    \prediction[0]_i_2__1 
       (.I0(\prediction_reg[0]_1 ),
        .I1(\prediction[0]_i_7__1_n_0 ),
        .I2(\prediction_reg[0]_2 ),
        .I3(\prediction_reg[0]_3 ),
        .I4(\prediction_reg[0]_4 ),
        .I5(\prediction[0]_i_10_n_0 ),
        .O(\prediction[0]_i_2__1_n_0 ));
  LUT3 #(
    .INIT(8'hFE)) 
    \prediction[0]_i_4__1 
       (.I0(mean_speed[13]),
        .I1(mean_speed[15]),
        .I2(mean_speed[14]),
        .O(\prediction[0]_i_4__1_n_0 ));
  LUT6 #(
    .INIT(64'h10115555FFFFFFFF)) 
    \prediction[0]_i_5__0 
       (.I0(mean_speed[11]),
        .I1(\prediction[1]_i_6__3_n_0 ),
        .I2(\prediction_reg[0]_5 ),
        .I3(mean_speed[3]),
        .I4(mean_speed[10]),
        .I5(mean_speed[12]),
        .O(\prediction[0]_i_5__0_n_0 ));
  LUT6 #(
    .INIT(64'h10115555FFFFFFFF)) 
    \prediction[0]_i_7__1 
       (.I0(accelerate[5]),
        .I1(accelerate[3]),
        .I2(\prediction[0]_i_13__1_n_0 ),
        .I3(accelerate[2]),
        .I4(accelerate[4]),
        .I5(\prediction[0]_i_14__2_n_0 ),
        .O(\prediction[0]_i_7__1_n_0 ));
  LUT6 #(
    .INIT(64'h0000000055557FFF)) 
    \prediction[1]_i_10__0 
       (.I0(\prediction[1]_i_4__0_1 ),
        .I1(kde_prob_mean[2]),
        .I2(kde_prob_mean[3]),
        .I3(kde_prob_mean[4]),
        .I4(kde_prob_mean_5_sn_1),
        .I5(kde_prob_mean_9_sn_1),
        .O(\prediction[1]_i_10__0_n_0 ));
  LUT6 #(
    .INIT(64'h7FFFFFFF7FFF7FFF)) 
    \prediction[1]_i_11__5 
       (.I0(dist_to_centroid_mean[13]),
        .I1(dist_to_centroid_mean[14]),
        .I2(dist_to_centroid_mean[11]),
        .I3(dist_to_centroid_mean[12]),
        .I4(dist_to_centroid_mean[10]),
        .I5(\prediction[1]_i_17__4_n_0 ),
        .O(\prediction[1]_i_11__5_n_0 ));
  LUT6 #(
    .INIT(64'h1011555500000000)) 
    \prediction[1]_i_12 
       (.I0(kde_prob_night_mean[8]),
        .I1(\prediction[1]_i_4__0_0 ),
        .I2(\prediction[1]_i_18__4_n_0 ),
        .I3(kde_prob_night_mean[6]),
        .I4(kde_prob_night_mean[7]),
        .I5(\prediction[1]_i_19_n_0 ),
        .O(tree_out));
  LUT6 #(
    .INIT(64'h000000005555555D)) 
    \prediction[1]_i_13__0 
       (.I0(\prediction[1]_i_4__0_1 ),
        .I1(\prediction[1]_i_20__2_n_0 ),
        .I2(kde_prob_mean[5]),
        .I3(kde_prob_mean[6]),
        .I4(kde_prob_mean[4]),
        .I5(kde_prob_mean_9_sn_1),
        .O(\prediction[1]_i_13__0_n_0 ));
  LUT6 #(
    .INIT(64'h00000000777777F7)) 
    \prediction[1]_i_14__4 
       (.I0(dist_to_centroid_mean[10]),
        .I1(dist_to_centroid_mean[11]),
        .I2(\prediction[1]_i_21__1_n_0 ),
        .I3(dist_to_centroid_mean[9]),
        .I4(dist_to_centroid_mean[8]),
        .I5(\prediction[1]_i_7__0_0 ),
        .O(\prediction[1]_i_14__4_n_0 ));
  LUT6 #(
    .INIT(64'h0000000055557FFF)) 
    \prediction[1]_i_15__2 
       (.I0(turning_angle_median[4]),
        .I1(turning_angle_median[0]),
        .I2(turning_angle_median[1]),
        .I3(turning_angle_median[2]),
        .I4(turning_angle_median[3]),
        .I5(turning_angle_median_6_sn_1),
        .O(\prediction[1]_i_15__2_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000000002)) 
    \prediction[1]_i_16__2 
       (.I0(kde_prob_mean_0_sn_1),
        .I1(kde_prob_mean_5_sn_1),
        .I2(kde_prob_mean[8]),
        .I3(kde_prob_mean[7]),
        .I4(kde_prob_mean[3]),
        .I5(kde_prob_mean[4]),
        .O(\prediction[1]_i_16__2_n_0 ));
  LUT6 #(
    .INIT(64'h0100FFFFFFFFFFFF)) 
    \prediction[1]_i_17__4 
       (.I0(dist_to_centroid_mean[5]),
        .I1(dist_to_centroid_mean[7]),
        .I2(dist_to_centroid_mean[6]),
        .I3(\prediction[1]_i_24__3_n_0 ),
        .I4(dist_to_centroid_mean[9]),
        .I5(dist_to_centroid_mean[8]),
        .O(\prediction[1]_i_17__4_n_0 ));
  LUT6 #(
    .INIT(64'h0000000055557FFF)) 
    \prediction[1]_i_18__4 
       (.I0(kde_prob_night_mean[4]),
        .I1(kde_prob_night_mean[0]),
        .I2(kde_prob_night_mean[1]),
        .I3(kde_prob_night_mean[2]),
        .I4(kde_prob_night_mean[3]),
        .I5(kde_prob_night_mean[5]),
        .O(\prediction[1]_i_18__4_n_0 ));
  LUT6 #(
    .INIT(64'h00000000555577F7)) 
    \prediction[1]_i_19 
       (.I0(\prediction_reg[1]_1 ),
        .I1(\prediction[1]_i_4__0_1 ),
        .I2(\prediction[1]_i_12_0 ),
        .I3(kde_prob_mean[6]),
        .I4(kde_prob_mean_9_sn_1),
        .I5(kde_prob_mean_13_sn_1),
        .O(\prediction[1]_i_19_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT4 #(
    .INIT(16'h01FF)) 
    \prediction[1]_i_20__2 
       (.I0(kde_prob_mean[0]),
        .I1(kde_prob_mean[1]),
        .I2(kde_prob_mean[2]),
        .I3(kde_prob_mean[3]),
        .O(\prediction[1]_i_20__2_n_0 ));
  LUT6 #(
    .INIT(64'h00015555FFFFFFFF)) 
    \prediction[1]_i_21__1 
       (.I0(dist_to_centroid_mean[4]),
        .I1(dist_to_centroid_mean[0]),
        .I2(dist_to_centroid_mean[1]),
        .I3(dist_to_centroid_mean[2]),
        .I4(dist_to_centroid_mean[3]),
        .I5(\prediction[1]_i_25__2_n_0 ),
        .O(\prediction[1]_i_21__1_n_0 ));
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[1]_i_23__4 
       (.I0(turning_angle_median[5]),
        .I1(turning_angle_median[6]),
        .O(turning_angle_median_6_sn_1));
  LUT5 #(
    .INIT(32'h0001FFFF)) 
    \prediction[1]_i_24__3 
       (.I0(dist_to_centroid_mean[0]),
        .I1(dist_to_centroid_mean[1]),
        .I2(dist_to_centroid_mean[3]),
        .I3(dist_to_centroid_mean[2]),
        .I4(dist_to_centroid_mean[4]),
        .O(\prediction[1]_i_24__3_n_0 ));
  LUT3 #(
    .INIT(8'h80)) 
    \prediction[1]_i_25__2 
       (.I0(dist_to_centroid_mean[5]),
        .I1(dist_to_centroid_mean[7]),
        .I2(dist_to_centroid_mean[6]),
        .O(\prediction[1]_i_25__2_n_0 ));
  LUT3 #(
    .INIT(8'hFE)) 
    \prediction[1]_i_2__1 
       (.I0(kde_prob_mean[13]),
        .I1(kde_prob_mean[15]),
        .I2(kde_prob_mean[14]),
        .O(kde_prob_mean_13_sn_1));
  LUT6 #(
    .INIT(64'h00000000555577F7)) 
    \prediction[1]_i_2__3 
       (.I0(mean_speed[12]),
        .I1(mean_speed[10]),
        .I2(\prediction[1]_i_5__2_n_0 ),
        .I3(\prediction[1]_i_6__3_n_0 ),
        .I4(mean_speed[11]),
        .I5(\prediction[0]_i_4__1_n_0 ),
        .O(\prediction[1]_i_2__3_n_0 ));
  LUT6 #(
    .INIT(64'hBA8AFFFFBA8A0000)) 
    \prediction[1]_i_3__0 
       (.I0(\prediction[1]_i_7__0_n_0 ),
        .I1(\prediction[1]_i_8__3_n_0 ),
        .I2(turning_angle_median_13_sn_1),
        .I3(\prediction[1]_i_9_n_0 ),
        .I4(\prediction_reg[0]_0 ),
        .I5(\prediction[0]_i_2__1_n_0 ),
        .O(\prediction[1]_i_3__0_n_0 ));
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[1]_i_3__2 
       (.I0(kde_prob_mean[9]),
        .I1(kde_prob_mean[10]),
        .O(kde_prob_mean_9_sn_1));
  LUT6 #(
    .INIT(64'hFFFF0DFF00000D00)) 
    \prediction[1]_i_4__0 
       (.I0(\prediction_reg[1]_1 ),
        .I1(\prediction[1]_i_10__0_n_0 ),
        .I2(kde_prob_mean_13_sn_1),
        .I3(\prediction[1]_i_11__5_n_0 ),
        .I4(dist_to_centroid_mean[15]),
        .I5(tree_out),
        .O(\prediction[1]_i_4__0_n_0 ));
  LUT4 #(
    .INIT(16'h15FF)) 
    \prediction[1]_i_5__2 
       (.I0(mean_speed[2]),
        .I1(mean_speed[1]),
        .I2(mean_speed[0]),
        .I3(mean_speed[3]),
        .O(\prediction[1]_i_5__2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFFE)) 
    \prediction[1]_i_6__3 
       (.I0(mean_speed[5]),
        .I1(mean_speed[4]),
        .I2(mean_speed[8]),
        .I3(mean_speed[9]),
        .I4(mean_speed[6]),
        .I5(mean_speed[7]),
        .O(\prediction[1]_i_6__3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFF45FF4545)) 
    \prediction[1]_i_7__0 
       (.I0(kde_prob_mean_13_sn_1),
        .I1(\prediction[1]_i_13__0_n_0 ),
        .I2(\prediction_reg[1]_1 ),
        .I3(\prediction[1]_i_14__4_n_0 ),
        .I4(dist_to_centroid_mean[14]),
        .I5(dist_to_centroid_mean[15]),
        .O(\prediction[1]_i_7__0_n_0 ));
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[1]_i_8__1 
       (.I0(kde_prob_mean[5]),
        .I1(kde_prob_mean[6]),
        .O(kde_prob_mean_5_sn_1));
  LUT6 #(
    .INIT(64'h000000000DFFFFFF)) 
    \prediction[1]_i_8__3 
       (.I0(turning_angle_median[7]),
        .I1(\prediction[1]_i_15__2_n_0 ),
        .I2(turning_angle_median[8]),
        .I3(turning_angle_median[9]),
        .I4(turning_angle_median[10]),
        .I5(turning_angle_median[11]),
        .O(\prediction[1]_i_8__3_n_0 ));
  LUT6 #(
    .INIT(64'h000000005555FF7F)) 
    \prediction[1]_i_9 
       (.I0(kde_prob_mean[12]),
        .I1(kde_prob_mean[9]),
        .I2(kde_prob_mean[10]),
        .I3(\prediction[1]_i_16__2_n_0 ),
        .I4(kde_prob_mean[11]),
        .I5(kde_prob_mean_13_sn_1),
        .O(\prediction[1]_i_9_n_0 ));
  LUT3 #(
    .INIT(8'h80)) 
    \prediction[1]_i_9__4 
       (.I0(turning_angle_median[12]),
        .I1(turning_angle_median[14]),
        .I2(turning_angle_median[13]),
        .O(turning_angle_median_13_sn_1));
  FDRE \prediction_reg[0] 
       (.C(clk),
        .CE(\prediction_reg[1]_2 ),
        .D(\prediction[0]_i_1__2_n_0 ),
        .Q(\prediction_reg_n_0_[0] ),
        .R(\prediction_reg[1]_0 ));
  MUXF7 \prediction_reg[0]_i_3 
       (.I0(\prediction[1]_i_9_n_0 ),
        .I1(\prediction[1]_i_7__0_n_0 ),
        .O(\prediction_reg[0]_i_3_n_0 ),
        .S(\prediction[0]_i_11__1_n_0 ));
  FDRE \prediction_reg[1] 
       (.C(clk),
        .CE(\prediction_reg[1]_2 ),
        .D(\prediction_reg[1]_i_1__0_n_0 ),
        .Q(\prediction_reg_n_0_[1] ),
        .R(\prediction_reg[1]_0 ));
  MUXF7 \prediction_reg[1]_i_1__0 
       (.I0(\prediction[1]_i_3__0_n_0 ),
        .I1(\prediction[1]_i_4__0_n_0 ),
        .O(\prediction_reg[1]_i_1__0_n_0 ),
        .S(\prediction[1]_i_2__3_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000008000)) 
    \result[0]_i_1 
       (.I0(t_done),
        .I1(done_reg_1[2]),
        .I2(done_reg_1[0]),
        .I3(done_reg_1[1]),
        .I4(done_reg_2),
        .I5(\result[1]_i_2_n_0 ),
        .O(D[0]));
  LUT6 #(
    .INIT(64'h0000800000000000)) 
    \result[1]_i_1 
       (.I0(t_done),
        .I1(done_reg_1[2]),
        .I2(done_reg_1[0]),
        .I3(done_reg_1[1]),
        .I4(done_reg_2),
        .I5(\result[1]_i_2_n_0 ),
        .O(D[1]));
  LUT6 #(
    .INIT(64'hEEE8E888E8888880)) 
    \result[1]_i_2 
       (.I0(\result[1]_i_3_n_0 ),
        .I1(\result_reg[0] ),
        .I2(\result_reg[0]_0 ),
        .I3(\result_reg[0]_1 ),
        .I4(p_3_in),
        .I5(\result[1]_i_8_n_0 ),
        .O(\result[1]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h02002F2202000200)) 
    \result[1]_i_3 
       (.I0(\prediction_reg_n_0_[1] ),
        .I1(\prediction_reg_n_0_[0] ),
        .I2(\result[1]_i_2_2 ),
        .I3(\result[1]_i_2_3 ),
        .I4(\result[1]_i_2_0 ),
        .I5(\result[1]_i_2_1 ),
        .O(\result[1]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h2D22D2DD2D222D22)) 
    \result[1]_i_8 
       (.I0(\prediction_reg_n_0_[1] ),
        .I1(\prediction_reg_n_0_[0] ),
        .I2(\result[1]_i_2_0 ),
        .I3(\result[1]_i_2_1 ),
        .I4(\result[1]_i_2_2 ),
        .I5(\result[1]_i_2_3 ),
        .O(\result[1]_i_8_n_0 ));
endmodule

(* ORIG_REF_NAME = "decision_tree_4" *) 
module design_1_random_forest_elepha_0_0_decision_tree_4
   (done_reg_0,
    start_0_sp_1,
    kde_prob_mean_6_sp_1,
    step_median_14_sp_1,
    kde_prob_mean_0_sp_1,
    step_median_1_sp_1,
    step_median_6_sp_1,
    turning_angle_max_10_sp_1,
    \accelerate[11] ,
    turning_angle_median_11_sp_1,
    mean_speed_0_sp_1,
    kde_prob_night_mean_11_sp_1,
    kde_prob_night_mean_2_sp_1,
    \prediction_reg[0]_0 ,
    \prediction_reg[0]_1 ,
    \prediction_reg[1]_0 ,
    clk,
    kde_prob_night_mean,
    \prediction_reg[1]_1 ,
    \prediction_reg[0]_2 ,
    kde_prob_mean,
    \prediction_reg[0]_3 ,
    \prediction_reg[0]_4 ,
    \prediction_reg[0]_5 ,
    \prediction_reg[0]_6 ,
    \prediction_reg[0]_7 ,
    mean_speed,
    \prediction[1]_i_24_0 ,
    step_median,
    turning_angle_max,
    accelerate,
    \prediction[1]_i_15__1_0 ,
    \prediction[1]_i_15__1_1 ,
    \prediction[1]_i_14_0 ,
    turning_angle_median,
    \prediction[1]_i_15__1_2 ,
    \prediction_reg[1]_2 ,
    \prediction[1]_i_4__3_0 ,
    start,
    \result[1]_i_2 ,
    \result[1]_i_2_0 ,
    \result[1]_i_2_1 ,
    \result[1]_i_2_2 ,
    \prediction_reg[1]_3 );
  output [0:0]done_reg_0;
  output start_0_sp_1;
  output kde_prob_mean_6_sp_1;
  output step_median_14_sp_1;
  output kde_prob_mean_0_sp_1;
  output step_median_1_sp_1;
  output step_median_6_sp_1;
  output turning_angle_max_10_sp_1;
  output \accelerate[11] ;
  output turning_angle_median_11_sp_1;
  output mean_speed_0_sp_1;
  output kde_prob_night_mean_11_sp_1;
  output kde_prob_night_mean_2_sp_1;
  output \prediction_reg[0]_0 ;
  output \prediction_reg[0]_1 ;
  output \prediction_reg[1]_0 ;
  input clk;
  input [15:0]kde_prob_night_mean;
  input \prediction_reg[1]_1 ;
  input \prediction_reg[0]_2 ;
  input [15:0]kde_prob_mean;
  input \prediction_reg[0]_3 ;
  input \prediction_reg[0]_4 ;
  input \prediction_reg[0]_5 ;
  input \prediction_reg[0]_6 ;
  input \prediction_reg[0]_7 ;
  input [13:0]mean_speed;
  input \prediction[1]_i_24_0 ;
  input [15:0]step_median;
  input [15:0]turning_angle_max;
  input [9:0]accelerate;
  input \prediction[1]_i_15__1_0 ;
  input \prediction[1]_i_15__1_1 ;
  input \prediction[1]_i_14_0 ;
  input [15:0]turning_angle_median;
  input \prediction[1]_i_15__1_2 ;
  input \prediction_reg[1]_2 ;
  input \prediction[1]_i_4__3_0 ;
  input [1:0]start;
  input \result[1]_i_2 ;
  input \result[1]_i_2_0 ;
  input \result[1]_i_2_1 ;
  input \result[1]_i_2_2 ;
  input \prediction_reg[1]_3 ;

  wire [9:0]accelerate;
  wire \accelerate[11] ;
  wire clk;
  wire done_i_1__3_n_0;
  wire [0:0]done_reg_0;
  wire [15:0]kde_prob_mean;
  wire kde_prob_mean_0_sn_1;
  wire kde_prob_mean_6_sn_1;
  wire [15:0]kde_prob_night_mean;
  wire kde_prob_night_mean_11_sn_1;
  wire kde_prob_night_mean_2_sn_1;
  wire [13:0]mean_speed;
  wire mean_speed_0_sn_1;
  wire \prediction[0]_i_11_n_0 ;
  wire \prediction[0]_i_12__0_n_0 ;
  wire \prediction[0]_i_13__0_n_0 ;
  wire \prediction[0]_i_14__1_n_0 ;
  wire \prediction[0]_i_15__0_n_0 ;
  wire \prediction[0]_i_17__0_n_0 ;
  wire \prediction[0]_i_18__2_n_0 ;
  wire \prediction[0]_i_19__0_n_0 ;
  wire \prediction[0]_i_1__5_n_0 ;
  wire \prediction[0]_i_21__2_n_0 ;
  wire \prediction[0]_i_22__2_n_0 ;
  wire \prediction[0]_i_23_n_0 ;
  wire \prediction[0]_i_2__2_n_0 ;
  wire \prediction[0]_i_3_n_0 ;
  wire \prediction[0]_i_5_n_0 ;
  wire \prediction[0]_i_6__0_n_0 ;
  wire \prediction[0]_i_8__0_n_0 ;
  wire \prediction[0]_i_9__0_n_0 ;
  wire \prediction[1]_i_10_n_0 ;
  wire \prediction[1]_i_12__0_n_0 ;
  wire \prediction[1]_i_14_0 ;
  wire \prediction[1]_i_15__1_0 ;
  wire \prediction[1]_i_15__1_1 ;
  wire \prediction[1]_i_15__1_2 ;
  wire \prediction[1]_i_16__0_n_0 ;
  wire \prediction[1]_i_17__3_n_0 ;
  wire \prediction[1]_i_21__3_n_0 ;
  wire \prediction[1]_i_22_n_0 ;
  wire \prediction[1]_i_23__1_n_0 ;
  wire \prediction[1]_i_24_0 ;
  wire \prediction[1]_i_24_n_0 ;
  wire \prediction[1]_i_25_n_0 ;
  wire \prediction[1]_i_26__0_n_0 ;
  wire \prediction[1]_i_27__1_n_0 ;
  wire \prediction[1]_i_28_n_0 ;
  wire \prediction[1]_i_29__2_n_0 ;
  wire \prediction[1]_i_31__1_n_0 ;
  wire \prediction[1]_i_33_n_0 ;
  wire \prediction[1]_i_35_n_0 ;
  wire \prediction[1]_i_36__0_n_0 ;
  wire \prediction[1]_i_41_n_0 ;
  wire \prediction[1]_i_42__0_n_0 ;
  wire \prediction[1]_i_43__0_n_0 ;
  wire \prediction[1]_i_4__3_0 ;
  wire \prediction[1]_i_4__3_n_0 ;
  wire \prediction[1]_i_5_n_0 ;
  wire \prediction[1]_i_6_n_0 ;
  wire \prediction[1]_i_7__4_n_0 ;
  wire \prediction_reg[0]_0 ;
  wire \prediction_reg[0]_1 ;
  wire \prediction_reg[0]_2 ;
  wire \prediction_reg[0]_3 ;
  wire \prediction_reg[0]_4 ;
  wire \prediction_reg[0]_5 ;
  wire \prediction_reg[0]_6 ;
  wire \prediction_reg[0]_7 ;
  wire \prediction_reg[1]_0 ;
  wire \prediction_reg[1]_1 ;
  wire \prediction_reg[1]_2 ;
  wire \prediction_reg[1]_3 ;
  wire \prediction_reg[1]_i_3_n_0 ;
  wire \result[1]_i_2 ;
  wire \result[1]_i_2_0 ;
  wire \result[1]_i_2_1 ;
  wire \result[1]_i_2_2 ;
  wire [1:0]start;
  wire start_0_sn_1;
  wire [15:0]step_median;
  wire step_median_14_sn_1;
  wire step_median_1_sn_1;
  wire step_median_6_sn_1;
  wire tree_out;
  wire tree_out3_out;
  wire tree_out5_out;
  wire [15:0]turning_angle_max;
  wire turning_angle_max_10_sn_1;
  wire [15:0]turning_angle_median;
  wire turning_angle_median_11_sn_1;

  assign kde_prob_mean_0_sp_1 = kde_prob_mean_0_sn_1;
  assign kde_prob_mean_6_sp_1 = kde_prob_mean_6_sn_1;
  assign kde_prob_night_mean_11_sp_1 = kde_prob_night_mean_11_sn_1;
  assign kde_prob_night_mean_2_sp_1 = kde_prob_night_mean_2_sn_1;
  assign mean_speed_0_sp_1 = mean_speed_0_sn_1;
  assign start_0_sp_1 = start_0_sn_1;
  assign step_median_14_sp_1 = step_median_14_sn_1;
  assign step_median_1_sp_1 = step_median_1_sn_1;
  assign step_median_6_sp_1 = step_median_6_sn_1;
  assign turning_angle_max_10_sp_1 = turning_angle_max_10_sn_1;
  assign turning_angle_median_11_sp_1 = turning_angle_median_11_sn_1;
  LUT2 #(
    .INIT(4'hD)) 
    done_i_1__3
       (.I0(start[1]),
        .I1(done_reg_0),
        .O(done_i_1__3_n_0));
  FDRE done_reg
       (.C(clk),
        .CE(1'b1),
        .D(done_i_1__3_n_0),
        .Q(done_reg_0),
        .R(start_0_sn_1));
  LUT6 #(
    .INIT(64'h000000005555555D)) 
    \prediction[0]_i_11 
       (.I0(kde_prob_mean[8]),
        .I1(\prediction[0]_i_18__2_n_0 ),
        .I2(kde_prob_mean[6]),
        .I3(kde_prob_mean[7]),
        .I4(kde_prob_mean[5]),
        .I5(\prediction_reg[0]_5 ),
        .O(\prediction[0]_i_11_n_0 ));
  LUT6 #(
    .INIT(64'h0100FFFFFFFFFFFF)) 
    \prediction[0]_i_12__0 
       (.I0(\prediction[0]_i_19__0_n_0 ),
        .I1(step_median[5]),
        .I2(step_median[4]),
        .I3(step_median_1_sn_1),
        .I4(step_median[7]),
        .I5(step_median[6]),
        .O(\prediction[0]_i_12__0_n_0 ));
  LUT3 #(
    .INIT(8'h07)) 
    \prediction[0]_i_12__1 
       (.I0(mean_speed[0]),
        .I1(mean_speed[1]),
        .I2(mean_speed[2]),
        .O(mean_speed_0_sn_1));
  LUT6 #(
    .INIT(64'h000000000000005D)) 
    \prediction[0]_i_13__0 
       (.I0(\prediction[0]_i_21__2_n_0 ),
        .I1(\prediction[0]_i_22__2_n_0 ),
        .I2(turning_angle_max[6]),
        .I3(turning_angle_max[14]),
        .I4(turning_angle_max[15]),
        .I5(turning_angle_max[13]),
        .O(\prediction[0]_i_13__0_n_0 ));
  LUT6 #(
    .INIT(64'h0000000077777FFF)) 
    \prediction[0]_i_14__1 
       (.I0(turning_angle_median[3]),
        .I1(turning_angle_median[4]),
        .I2(turning_angle_median[0]),
        .I3(turning_angle_median[1]),
        .I4(turning_angle_median[2]),
        .I5(turning_angle_median[5]),
        .O(\prediction[0]_i_14__1_n_0 ));
  LUT6 #(
    .INIT(64'h8000000000000000)) 
    \prediction[0]_i_15__0 
       (.I0(turning_angle_median[7]),
        .I1(turning_angle_median[8]),
        .I2(turning_angle_median[6]),
        .I3(turning_angle_median_11_sn_1),
        .I4(turning_angle_median[9]),
        .I5(turning_angle_median[10]),
        .O(\prediction[0]_i_15__0_n_0 ));
  LUT2 #(
    .INIT(4'h8)) 
    \prediction[0]_i_16 
       (.I0(turning_angle_max[10]),
        .I1(turning_angle_max[11]),
        .O(turning_angle_max_10_sn_1));
  LUT6 #(
    .INIT(64'h0000000000000057)) 
    \prediction[0]_i_17__0 
       (.I0(turning_angle_max[2]),
        .I1(turning_angle_max[1]),
        .I2(turning_angle_max[0]),
        .I3(turning_angle_max[5]),
        .I4(turning_angle_max[6]),
        .I5(\prediction[0]_i_23_n_0 ),
        .O(\prediction[0]_i_17__0_n_0 ));
  LUT5 #(
    .INIT(32'h1555FFFF)) 
    \prediction[0]_i_18__2 
       (.I0(kde_prob_mean[3]),
        .I1(kde_prob_mean[2]),
        .I2(kde_prob_mean[1]),
        .I3(kde_prob_mean[0]),
        .I4(kde_prob_mean[4]),
        .O(\prediction[0]_i_18__2_n_0 ));
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[0]_i_19__0 
       (.I0(step_median[2]),
        .I1(step_median[3]),
        .O(\prediction[0]_i_19__0_n_0 ));
  LUT6 #(
    .INIT(64'h4545457575754575)) 
    \prediction[0]_i_1__5 
       (.I0(\prediction[1]_i_5_n_0 ),
        .I1(kde_prob_night_mean[15]),
        .I2(\prediction[0]_i_2__2_n_0 ),
        .I3(\prediction[0]_i_3_n_0 ),
        .I4(kde_prob_mean_6_sn_1),
        .I5(\prediction[0]_i_5_n_0 ),
        .O(\prediction[0]_i_1__5_n_0 ));
  LUT2 #(
    .INIT(4'h7)) 
    \prediction[0]_i_20 
       (.I0(step_median[1]),
        .I1(step_median[0]),
        .O(step_median_1_sn_1));
  LUT6 #(
    .INIT(64'h8000000000000000)) 
    \prediction[0]_i_21__2 
       (.I0(turning_angle_max[8]),
        .I1(turning_angle_max[7]),
        .I2(turning_angle_max[11]),
        .I3(turning_angle_max[12]),
        .I4(turning_angle_max[9]),
        .I5(turning_angle_max[10]),
        .O(\prediction[0]_i_21__2_n_0 ));
  LUT6 #(
    .INIT(64'h777777777777777F)) 
    \prediction[0]_i_22__2 
       (.I0(turning_angle_max[5]),
        .I1(turning_angle_max[4]),
        .I2(turning_angle_max[0]),
        .I3(turning_angle_max[1]),
        .I4(turning_angle_max[3]),
        .I5(turning_angle_max[2]),
        .O(\prediction[0]_i_22__2_n_0 ));
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[0]_i_23 
       (.I0(turning_angle_max[3]),
        .I1(turning_angle_max[4]),
        .O(\prediction[0]_i_23_n_0 ));
  LUT6 #(
    .INIT(64'h01005555FFFFFFFF)) 
    \prediction[0]_i_2__2 
       (.I0(kde_prob_night_mean_11_sn_1),
        .I1(kde_prob_night_mean[8]),
        .I2(kde_prob_night_mean[9]),
        .I3(\prediction[1]_i_7__4_n_0 ),
        .I4(kde_prob_night_mean[10]),
        .I5(kde_prob_night_mean[14]),
        .O(\prediction[0]_i_2__2_n_0 ));
  LUT6 #(
    .INIT(64'h00A200A2FFAE00A2)) 
    \prediction[0]_i_3 
       (.I0(tree_out),
        .I1(step_median[13]),
        .I2(\prediction[0]_i_6__0_n_0 ),
        .I3(step_median_14_sn_1),
        .I4(\prediction[1]_i_17__3_n_0 ),
        .I5(\prediction_reg[1]_1 ),
        .O(\prediction[0]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h000000005555FF7F)) 
    \prediction[0]_i_4 
       (.I0(\prediction_reg[0]_2 ),
        .I1(kde_prob_mean[6]),
        .I2(\prediction_reg[0]_3 ),
        .I3(\prediction_reg[0]_4 ),
        .I4(\prediction_reg[0]_5 ),
        .I5(\prediction_reg[0]_6 ),
        .O(kde_prob_mean_6_sn_1));
  LUT6 #(
    .INIT(64'h8A8A8A8ABABA8ABA)) 
    \prediction[0]_i_5 
       (.I0(\prediction[0]_i_8__0_n_0 ),
        .I1(\prediction[0]_i_9__0_n_0 ),
        .I2(\prediction_reg[0]_7 ),
        .I3(\prediction_reg[0]_2 ),
        .I4(\prediction[0]_i_11_n_0 ),
        .I5(\prediction_reg[0]_6 ),
        .O(\prediction[0]_i_5_n_0 ));
  LUT6 #(
    .INIT(64'h000000007F7FFF7F)) 
    \prediction[0]_i_6__0 
       (.I0(step_median[9]),
        .I1(step_median[11]),
        .I2(step_median[10]),
        .I3(\prediction[0]_i_12__0_n_0 ),
        .I4(step_median[8]),
        .I5(step_median[12]),
        .O(\prediction[0]_i_6__0_n_0 ));
  LUT6 #(
    .INIT(64'hFFFEFFFFFFFEFFFE)) 
    \prediction[0]_i_8__0 
       (.I0(\prediction[0]_i_13__0_n_0 ),
        .I1(turning_angle_median[13]),
        .I2(turning_angle_median[15]),
        .I3(turning_angle_median[14]),
        .I4(\prediction[0]_i_14__1_n_0 ),
        .I5(\prediction[0]_i_15__0_n_0 ),
        .O(\prediction[0]_i_8__0_n_0 ));
  LUT6 #(
    .INIT(64'h000000005555FF7F)) 
    \prediction[0]_i_9__0 
       (.I0(turning_angle_max_10_sn_1),
        .I1(turning_angle_max[7]),
        .I2(turning_angle_max[8]),
        .I3(\prediction[0]_i_17__0_n_0 ),
        .I4(turning_angle_max[9]),
        .I5(turning_angle_max[12]),
        .O(\prediction[0]_i_9__0_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \prediction[1]_i_1 
       (.I0(start[0]),
        .O(start_0_sn_1));
  LUT6 #(
    .INIT(64'h01005555FFFFFFFF)) 
    \prediction[1]_i_10 
       (.I0(\prediction_reg[0]_5 ),
        .I1(kde_prob_mean[6]),
        .I2(kde_prob_mean[7]),
        .I3(kde_prob_mean_0_sn_1),
        .I4(kde_prob_mean[8]),
        .I5(\prediction_reg[0]_2 ),
        .O(\prediction[1]_i_10_n_0 ));
  LUT6 #(
    .INIT(64'h5455545444444444)) 
    \prediction[1]_i_11 
       (.I0(\prediction[1]_i_21__3_n_0 ),
        .I1(step_median_14_sn_1),
        .I2(step_median[12]),
        .I3(\prediction[1]_i_22_n_0 ),
        .I4(step_median[11]),
        .I5(step_median[13]),
        .O(tree_out3_out));
  LUT6 #(
    .INIT(64'h0100FFFFFFFFFFFF)) 
    \prediction[1]_i_12__0 
       (.I0(step_median[9]),
        .I1(step_median[11]),
        .I2(step_median[10]),
        .I3(\prediction[1]_i_23__1_n_0 ),
        .I4(step_median[13]),
        .I5(step_median[12]),
        .O(\prediction[1]_i_12__0_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[1]_i_13__1 
       (.I0(step_median[14]),
        .I1(step_median[15]),
        .O(step_median_14_sn_1));
  LUT6 #(
    .INIT(64'h4500454545004500)) 
    \prediction[1]_i_14 
       (.I0(kde_prob_mean[15]),
        .I1(\prediction[1]_i_24_n_0 ),
        .I2(kde_prob_mean[14]),
        .I3(mean_speed[13]),
        .I4(\prediction[1]_i_25_n_0 ),
        .I5(mean_speed[12]),
        .O(tree_out5_out));
  LUT5 #(
    .INIT(32'h20230000)) 
    \prediction[1]_i_15__1 
       (.I0(\prediction[1]_i_26__0_n_0 ),
        .I1(accelerate[9]),
        .I2(accelerate[7]),
        .I3(\prediction[1]_i_27__1_n_0 ),
        .I4(accelerate[8]),
        .O(tree_out));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT5 #(
    .INIT(32'h0000005D)) 
    \prediction[1]_i_16__0 
       (.I0(step_median[13]),
        .I1(\prediction[1]_i_28_n_0 ),
        .I2(step_median[12]),
        .I3(step_median[15]),
        .I4(step_median[14]),
        .O(\prediction[1]_i_16__0_n_0 ));
  LUT2 #(
    .INIT(4'h8)) 
    \prediction[1]_i_17__2 
       (.I0(accelerate[5]),
        .I1(accelerate[6]),
        .O(\accelerate[11] ));
  LUT6 #(
    .INIT(64'h1115FFFFFFFFFFFF)) 
    \prediction[1]_i_17__3 
       (.I0(\prediction[1]_i_29__2_n_0 ),
        .I1(turning_angle_median[4]),
        .I2(turning_angle_median[3]),
        .I3(turning_angle_median[2]),
        .I4(turning_angle_median_11_sn_1),
        .I5(turning_angle_median[10]),
        .O(\prediction[1]_i_17__3_n_0 ));
  LUT6 #(
    .INIT(64'h1FFFFFFFFFFFFFFF)) 
    \prediction[1]_i_20 
       (.I0(kde_prob_mean[0]),
        .I1(kde_prob_mean[1]),
        .I2(kde_prob_mean[4]),
        .I3(kde_prob_mean[5]),
        .I4(kde_prob_mean[2]),
        .I5(kde_prob_mean[3]),
        .O(kde_prob_mean_0_sn_1));
  LUT6 #(
    .INIT(64'h45FFFFFFFFFFFFFF)) 
    \prediction[1]_i_21__3 
       (.I0(\prediction[1]_i_31__1_n_0 ),
        .I1(kde_prob_night_mean_2_sn_1),
        .I2(kde_prob_night_mean[5]),
        .I3(kde_prob_night_mean[14]),
        .I4(kde_prob_night_mean[15]),
        .I5(kde_prob_night_mean[13]),
        .O(\prediction[1]_i_21__3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000555577F7)) 
    \prediction[1]_i_22 
       (.I0(step_median[9]),
        .I1(step_median[7]),
        .I2(\prediction[1]_i_33_n_0 ),
        .I3(step_median[6]),
        .I4(step_median[8]),
        .I5(step_median[10]),
        .O(\prediction[1]_i_22_n_0 ));
  LUT6 #(
    .INIT(64'h15FFFFFFFFFFFFFF)) 
    \prediction[1]_i_23__1 
       (.I0(step_median[2]),
        .I1(step_median[1]),
        .I2(step_median[0]),
        .I3(step_median[3]),
        .I4(step_median[4]),
        .I5(step_median_6_sn_1),
        .O(\prediction[1]_i_23__1_n_0 ));
  LUT6 #(
    .INIT(64'h000000005555555D)) 
    \prediction[1]_i_24 
       (.I0(kde_prob_mean[12]),
        .I1(\prediction[1]_i_35_n_0 ),
        .I2(kde_prob_mean[10]),
        .I3(kde_prob_mean[11]),
        .I4(kde_prob_mean[9]),
        .I5(kde_prob_mean[13]),
        .O(\prediction[1]_i_24_n_0 ));
  LUT6 #(
    .INIT(64'h00000000000077F7)) 
    \prediction[1]_i_25 
       (.I0(mean_speed[9]),
        .I1(mean_speed[10]),
        .I2(\prediction[1]_i_36__0_n_0 ),
        .I3(mean_speed[8]),
        .I4(\prediction[1]_i_14_0 ),
        .I5(mean_speed[11]),
        .O(\prediction[1]_i_25_n_0 ));
  LUT5 #(
    .INIT(32'h0000555D)) 
    \prediction[1]_i_26__0 
       (.I0(accelerate[2]),
        .I1(\prediction[1]_i_15__1_0 ),
        .I2(accelerate[1]),
        .I3(accelerate[0]),
        .I4(\prediction[1]_i_15__1_1 ),
        .O(\prediction[1]_i_26__0_n_0 ));
  LUT6 #(
    .INIT(64'h10115555FFFFFFFF)) 
    \prediction[1]_i_27__1 
       (.I0(accelerate[4]),
        .I1(accelerate[2]),
        .I2(\prediction[1]_i_15__1_2 ),
        .I3(accelerate[1]),
        .I4(accelerate[3]),
        .I5(\accelerate[11] ),
        .O(\prediction[1]_i_27__1_n_0 ));
  LUT6 #(
    .INIT(64'h4555FFFFFFFFFFFF)) 
    \prediction[1]_i_28 
       (.I0(step_median[8]),
        .I1(\prediction[1]_i_41_n_0 ),
        .I2(step_median[7]),
        .I3(step_median[6]),
        .I4(\prediction[1]_i_42__0_n_0 ),
        .I5(step_median[9]),
        .O(\prediction[1]_i_28_n_0 ));
  LUT5 #(
    .INIT(32'hFFFFFFFE)) 
    \prediction[1]_i_29__2 
       (.I0(turning_angle_median[5]),
        .I1(turning_angle_median[8]),
        .I2(turning_angle_median[9]),
        .I3(turning_angle_median[6]),
        .I4(turning_angle_median[7]),
        .O(\prediction[1]_i_29__2_n_0 ));
  LUT2 #(
    .INIT(4'h8)) 
    \prediction[1]_i_30__0 
       (.I0(turning_angle_median[11]),
        .I1(turning_angle_median[12]),
        .O(turning_angle_median_11_sn_1));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFFE)) 
    \prediction[1]_i_31__1 
       (.I0(kde_prob_night_mean[7]),
        .I1(kde_prob_night_mean[8]),
        .I2(kde_prob_night_mean[6]),
        .I3(kde_prob_night_mean[11]),
        .I4(kde_prob_night_mean[12]),
        .I5(\prediction[1]_i_43__0_n_0 ),
        .O(\prediction[1]_i_31__1_n_0 ));
  LUT5 #(
    .INIT(32'h0000777F)) 
    \prediction[1]_i_32__1 
       (.I0(kde_prob_night_mean[2]),
        .I1(kde_prob_night_mean[3]),
        .I2(kde_prob_night_mean[1]),
        .I3(kde_prob_night_mean[0]),
        .I4(kde_prob_night_mean[4]),
        .O(kde_prob_night_mean_2_sn_1));
  LUT6 #(
    .INIT(64'h01115555FFFFFFFF)) 
    \prediction[1]_i_33 
       (.I0(step_median[4]),
        .I1(step_median[2]),
        .I2(step_median[1]),
        .I3(step_median[0]),
        .I4(step_median[3]),
        .I5(step_median[5]),
        .O(\prediction[1]_i_33_n_0 ));
  LUT4 #(
    .INIT(16'h8000)) 
    \prediction[1]_i_34__0 
       (.I0(step_median[6]),
        .I1(step_median[5]),
        .I2(step_median[8]),
        .I3(step_median[7]),
        .O(step_median_6_sn_1));
  LUT6 #(
    .INIT(64'h01001111FFFFFFFF)) 
    \prediction[1]_i_35 
       (.I0(kde_prob_mean[5]),
        .I1(kde_prob_mean[6]),
        .I2(kde_prob_mean[3]),
        .I3(\prediction[1]_i_24_0 ),
        .I4(kde_prob_mean[4]),
        .I5(\prediction_reg[0]_3 ),
        .O(\prediction[1]_i_35_n_0 ));
  LUT6 #(
    .INIT(64'h01000101FFFFFFFF)) 
    \prediction[1]_i_36__0 
       (.I0(mean_speed[4]),
        .I1(mean_speed[6]),
        .I2(mean_speed[5]),
        .I3(mean_speed_0_sn_1),
        .I4(mean_speed[3]),
        .I5(mean_speed[7]),
        .O(\prediction[1]_i_36__0_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000000007)) 
    \prediction[1]_i_41 
       (.I0(step_median[0]),
        .I1(step_median[1]),
        .I2(step_median[4]),
        .I3(step_median[5]),
        .I4(step_median[2]),
        .I5(step_median[3]),
        .O(\prediction[1]_i_41_n_0 ));
  LUT2 #(
    .INIT(4'h8)) 
    \prediction[1]_i_42__0 
       (.I0(step_median[10]),
        .I1(step_median[11]),
        .O(\prediction[1]_i_42__0_n_0 ));
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[1]_i_43__0 
       (.I0(kde_prob_night_mean[9]),
        .I1(kde_prob_night_mean[10]),
        .O(\prediction[1]_i_43__0_n_0 ));
  LUT6 #(
    .INIT(64'h00000000555577F7)) 
    \prediction[1]_i_4__3 
       (.I0(kde_prob_night_mean[14]),
        .I1(kde_prob_night_mean[10]),
        .I2(\prediction[1]_i_7__4_n_0 ),
        .I3(\prediction_reg[1]_2 ),
        .I4(kde_prob_night_mean_11_sn_1),
        .I5(kde_prob_night_mean[15]),
        .O(\prediction[1]_i_4__3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF4FF0000F400)) 
    \prediction[1]_i_5 
       (.I0(\prediction_reg[0]_6 ),
        .I1(\prediction[1]_i_10_n_0 ),
        .I2(tree_out3_out),
        .I3(\prediction[1]_i_12__0_n_0 ),
        .I4(step_median_14_sn_1),
        .I5(tree_out5_out),
        .O(\prediction[1]_i_5_n_0 ));
  LUT6 #(
    .INIT(64'hB888B888B8BBB888)) 
    \prediction[1]_i_6 
       (.I0(\prediction[0]_i_5_n_0 ),
        .I1(kde_prob_mean_6_sn_1),
        .I2(tree_out),
        .I3(\prediction[1]_i_16__0_n_0 ),
        .I4(\prediction[1]_i_17__3_n_0 ),
        .I5(\prediction_reg[1]_1 ),
        .O(\prediction[1]_i_6_n_0 ));
  LUT6 #(
    .INIT(64'h10111111FFFFFFFF)) 
    \prediction[1]_i_7__4 
       (.I0(kde_prob_night_mean[5]),
        .I1(kde_prob_night_mean[6]),
        .I2(\prediction[1]_i_4__3_0 ),
        .I3(kde_prob_night_mean[4]),
        .I4(kde_prob_night_mean[3]),
        .I5(kde_prob_night_mean[7]),
        .O(\prediction[1]_i_7__4_n_0 ));
  LUT3 #(
    .INIT(8'hFE)) 
    \prediction[1]_i_9__5 
       (.I0(kde_prob_night_mean[11]),
        .I1(kde_prob_night_mean[13]),
        .I2(kde_prob_night_mean[12]),
        .O(kde_prob_night_mean_11_sn_1));
  FDRE \prediction_reg[0] 
       (.C(clk),
        .CE(\prediction_reg[1]_3 ),
        .D(\prediction[0]_i_1__5_n_0 ),
        .Q(\prediction_reg[0]_1 ),
        .R(start_0_sn_1));
  FDRE \prediction_reg[1] 
       (.C(clk),
        .CE(\prediction_reg[1]_3 ),
        .D(\prediction_reg[1]_i_3_n_0 ),
        .Q(\prediction_reg[1]_0 ),
        .R(start_0_sn_1));
  MUXF7 \prediction_reg[1]_i_3 
       (.I0(\prediction[1]_i_5_n_0 ),
        .I1(\prediction[1]_i_6_n_0 ),
        .O(\prediction_reg[1]_i_3_n_0 ),
        .S(\prediction[1]_i_4__3_n_0 ));
  LUT6 #(
    .INIT(64'h04004F4404000400)) 
    \result[1]_i_4 
       (.I0(\prediction_reg[0]_1 ),
        .I1(\prediction_reg[1]_0 ),
        .I2(\result[1]_i_2 ),
        .I3(\result[1]_i_2_0 ),
        .I4(\result[1]_i_2_1 ),
        .I5(\result[1]_i_2_2 ),
        .O(\prediction_reg[0]_0 ));
endmodule

(* ORIG_REF_NAME = "decision_tree_5" *) 
module design_1_random_forest_elepha_0_0_decision_tree_5
   (t_done,
    accelerate_5_sp_1,
    mean_speed_12_sp_1,
    kde_prob_night_mean_8_sp_1,
    accelerate_1_sp_1,
    \prediction_reg[1]_0 ,
    \prediction_reg[0]_0 ,
    \prediction_reg[1]_1 ,
    clk,
    kde_prob_night_mean,
    \prediction_reg[0]_1 ,
    dist_to_centroid_mean,
    \prediction[1]_i_4_0 ,
    accelerate,
    \prediction_reg[1]_2 ,
    \prediction_reg[1]_3 ,
    \prediction_reg[1]_4 ,
    \prediction[1]_i_2_0 ,
    kde_prob_mean,
    \prediction[1]_i_2_1 ,
    \prediction[1]_i_2_2 ,
    step_median,
    mean_speed,
    \prediction[1]_i_20__1_0 ,
    \prediction_reg[1]_5 ,
    start,
    \prediction_reg[1]_6 );
  output [0:0]t_done;
  output accelerate_5_sp_1;
  output mean_speed_12_sp_1;
  output kde_prob_night_mean_8_sp_1;
  output accelerate_1_sp_1;
  output \prediction_reg[1]_0 ;
  output \prediction_reg[0]_0 ;
  input \prediction_reg[1]_1 ;
  input clk;
  input [15:0]kde_prob_night_mean;
  input \prediction_reg[0]_1 ;
  input [15:0]dist_to_centroid_mean;
  input \prediction[1]_i_4_0 ;
  input [15:0]accelerate;
  input \prediction_reg[1]_2 ;
  input \prediction_reg[1]_3 ;
  input \prediction_reg[1]_4 ;
  input \prediction[1]_i_2_0 ;
  input [4:0]kde_prob_mean;
  input \prediction[1]_i_2_1 ;
  input \prediction[1]_i_2_2 ;
  input [15:0]step_median;
  input [15:0]mean_speed;
  input \prediction[1]_i_20__1_0 ;
  input \prediction_reg[1]_5 ;
  input [0:0]start;
  input \prediction_reg[1]_6 ;

  wire [15:0]accelerate;
  wire accelerate_1_sn_1;
  wire accelerate_5_sn_1;
  wire clk;
  wire [15:0]dist_to_centroid_mean;
  wire done_i_1__4_n_0;
  wire [4:0]kde_prob_mean;
  wire [15:0]kde_prob_night_mean;
  wire kde_prob_night_mean_8_sn_1;
  wire [15:0]mean_speed;
  wire mean_speed_12_sn_1;
  wire \prediction[0]_i_1_n_0 ;
  wire \prediction[1]_i_14__3_n_0 ;
  wire \prediction[1]_i_15__3_n_0 ;
  wire \prediction[1]_i_16__4_n_0 ;
  wire \prediction[1]_i_17__5_n_0 ;
  wire \prediction[1]_i_18_n_0 ;
  wire \prediction[1]_i_19__3_n_0 ;
  wire \prediction[1]_i_20__1_0 ;
  wire \prediction[1]_i_22__5_n_0 ;
  wire \prediction[1]_i_23_n_0 ;
  wire \prediction[1]_i_24__1_n_0 ;
  wire \prediction[1]_i_25__4_n_0 ;
  wire \prediction[1]_i_26__3_n_0 ;
  wire \prediction[1]_i_27__2_n_0 ;
  wire \prediction[1]_i_28__3_n_0 ;
  wire \prediction[1]_i_29__1_n_0 ;
  wire \prediction[1]_i_2_0 ;
  wire \prediction[1]_i_2_1 ;
  wire \prediction[1]_i_2_2 ;
  wire \prediction[1]_i_2_n_0 ;
  wire \prediction[1]_i_30__1_n_0 ;
  wire \prediction[1]_i_31__0_n_0 ;
  wire \prediction[1]_i_32__0_n_0 ;
  wire \prediction[1]_i_33__0_n_0 ;
  wire \prediction[1]_i_34__1_n_0 ;
  wire \prediction[1]_i_35__1_n_0 ;
  wire \prediction[1]_i_36__1_n_0 ;
  wire \prediction[1]_i_38_n_0 ;
  wire \prediction[1]_i_39__1_n_0 ;
  wire \prediction[1]_i_40__1_n_0 ;
  wire \prediction[1]_i_41__0_n_0 ;
  wire \prediction[1]_i_44__0_n_0 ;
  wire \prediction[1]_i_4_0 ;
  wire \prediction[1]_i_5__3_n_0 ;
  wire \prediction[1]_i_6__2_n_0 ;
  wire \prediction[1]_i_7__1_n_0 ;
  wire \prediction[1]_i_8__0_n_0 ;
  wire \prediction[1]_i_9__0_n_0 ;
  wire \prediction_reg[0]_0 ;
  wire \prediction_reg[0]_1 ;
  wire \prediction_reg[1]_0 ;
  wire \prediction_reg[1]_1 ;
  wire \prediction_reg[1]_2 ;
  wire \prediction_reg[1]_3 ;
  wire \prediction_reg[1]_4 ;
  wire \prediction_reg[1]_5 ;
  wire \prediction_reg[1]_6 ;
  wire [0:0]start;
  wire [15:0]step_median;
  wire [0:0]t_done;
  wire tree_out;
  wire tree_out1_out;
  wire tree_out4_out;
  wire tree_out__0;

  assign accelerate_1_sp_1 = accelerate_1_sn_1;
  assign accelerate_5_sp_1 = accelerate_5_sn_1;
  assign kde_prob_night_mean_8_sp_1 = kde_prob_night_mean_8_sn_1;
  assign mean_speed_12_sp_1 = mean_speed_12_sn_1;
  LUT2 #(
    .INIT(4'hD)) 
    done_i_1__4
       (.I0(start),
        .I1(t_done),
        .O(done_i_1__4_n_0));
  FDRE done_reg
       (.C(clk),
        .CE(1'b1),
        .D(done_i_1__4_n_0),
        .Q(t_done),
        .R(\prediction_reg[1]_1 ));
  LUT6 #(
    .INIT(64'h00004575FFFF4575)) 
    \prediction[0]_i_1 
       (.I0(\prediction[1]_i_6__2_n_0 ),
        .I1(kde_prob_night_mean[15]),
        .I2(\prediction[1]_i_5__3_n_0 ),
        .I3(tree_out1_out),
        .I4(\prediction_reg[0]_1 ),
        .I5(\prediction[1]_i_2_n_0 ),
        .O(\prediction[0]_i_1_n_0 ));
  LUT2 #(
    .INIT(4'h7)) 
    \prediction[0]_i_21__0 
       (.I0(accelerate[1]),
        .I1(accelerate[0]),
        .O(accelerate_1_sn_1));
  LUT6 #(
    .INIT(64'hBBBBB8BB8888B888)) 
    \prediction[1]_i_1 
       (.I0(\prediction[1]_i_2_n_0 ),
        .I1(\prediction_reg[0]_1 ),
        .I2(tree_out1_out),
        .I3(\prediction[1]_i_5__3_n_0 ),
        .I4(kde_prob_night_mean[15]),
        .I5(\prediction[1]_i_6__2_n_0 ),
        .O(tree_out));
  LUT6 #(
    .INIT(64'hFFFFFFFFEFEEAAAA)) 
    \prediction[1]_i_13 
       (.I0(\prediction[1]_i_4_0 ),
        .I1(accelerate[13]),
        .I2(\prediction[1]_i_24__1_n_0 ),
        .I3(accelerate[12]),
        .I4(accelerate[14]),
        .I5(accelerate[15]),
        .O(tree_out__0));
  LUT6 #(
    .INIT(64'h1011FFFFFFFFFFFF)) 
    \prediction[1]_i_14__3 
       (.I0(dist_to_centroid_mean[7]),
        .I1(dist_to_centroid_mean[8]),
        .I2(\prediction[1]_i_25__4_n_0 ),
        .I3(dist_to_centroid_mean[6]),
        .I4(\prediction[1]_i_26__3_n_0 ),
        .I5(dist_to_centroid_mean[9]),
        .O(\prediction[1]_i_14__3_n_0 ));
  LUT6 #(
    .INIT(64'h000000001FFFFFFF)) 
    \prediction[1]_i_15__3 
       (.I0(kde_prob_night_mean[2]),
        .I1(kde_prob_night_mean[3]),
        .I2(kde_prob_night_mean[6]),
        .I3(kde_prob_night_mean[5]),
        .I4(kde_prob_night_mean[4]),
        .I5(kde_prob_night_mean[7]),
        .O(\prediction[1]_i_15__3_n_0 ));
  LUT6 #(
    .INIT(64'hEEEEEEEEAAAAEAAA)) 
    \prediction[1]_i_16__4 
       (.I0(mean_speed[15]),
        .I1(mean_speed[14]),
        .I2(mean_speed[11]),
        .I3(mean_speed[12]),
        .I4(\prediction[1]_i_27__2_n_0 ),
        .I5(mean_speed[13]),
        .O(\prediction[1]_i_16__4_n_0 ));
  LUT6 #(
    .INIT(64'h000000005555DFFF)) 
    \prediction[1]_i_17__5 
       (.I0(kde_prob_night_mean[14]),
        .I1(\prediction[1]_i_28__3_n_0 ),
        .I2(kde_prob_night_mean[11]),
        .I3(kde_prob_night_mean[12]),
        .I4(kde_prob_night_mean[13]),
        .I5(kde_prob_night_mean[15]),
        .O(\prediction[1]_i_17__5_n_0 ));
  LUT6 #(
    .INIT(64'hEEEEEEEEEAEAAAEA)) 
    \prediction[1]_i_18 
       (.I0(accelerate[15]),
        .I1(accelerate[14]),
        .I2(accelerate[12]),
        .I3(\prediction[1]_i_29__1_n_0 ),
        .I4(accelerate[11]),
        .I5(accelerate[13]),
        .O(\prediction[1]_i_18_n_0 ));
  LUT6 #(
    .INIT(64'h10115555FFFFFFFF)) 
    \prediction[1]_i_19__3 
       (.I0(kde_prob_night_mean[13]),
        .I1(kde_prob_night_mean[7]),
        .I2(\prediction[1]_i_30__1_n_0 ),
        .I3(kde_prob_night_mean[6]),
        .I4(\prediction[1]_i_31__0_n_0 ),
        .I5(kde_prob_night_mean[14]),
        .O(\prediction[1]_i_19__3_n_0 ));
  LUT6 #(
    .INIT(64'h000000000F0FBF8F)) 
    \prediction[1]_i_2 
       (.I0(\prediction[1]_i_7__1_n_0 ),
        .I1(\prediction[1]_i_8__0_n_0 ),
        .I2(\prediction_reg[1]_2 ),
        .I3(\prediction[1]_i_9__0_n_0 ),
        .I4(\prediction_reg[1]_3 ),
        .I5(\prediction_reg[1]_4 ),
        .O(\prediction[1]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hEFAA0000EFAAEFAA)) 
    \prediction[1]_i_20__1 
       (.I0(accelerate[15]),
        .I1(accelerate[13]),
        .I2(\prediction[1]_i_32__0_n_0 ),
        .I3(accelerate[14]),
        .I4(mean_speed[15]),
        .I5(\prediction[1]_i_33__0_n_0 ),
        .O(tree_out4_out));
  LUT6 #(
    .INIT(64'h01005555FFFFFFFF)) 
    \prediction[1]_i_22__5 
       (.I0(step_median[9]),
        .I1(step_median[6]),
        .I2(step_median[7]),
        .I3(\prediction[1]_i_34__1_n_0 ),
        .I4(step_median[8]),
        .I5(step_median[10]),
        .O(\prediction[1]_i_22__5_n_0 ));
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[1]_i_23 
       (.I0(kde_prob_mean[2]),
        .I1(kde_prob_mean[3]),
        .O(\prediction[1]_i_23_n_0 ));
  LUT6 #(
    .INIT(64'h00000000000000F7)) 
    \prediction[1]_i_24__1 
       (.I0(accelerate[7]),
        .I1(accelerate[8]),
        .I2(accelerate_5_sn_1),
        .I3(accelerate[10]),
        .I4(accelerate[11]),
        .I5(accelerate[9]),
        .O(\prediction[1]_i_24__1_n_0 ));
  LUT6 #(
    .INIT(64'h000000007FFFFFFF)) 
    \prediction[1]_i_25__4 
       (.I0(dist_to_centroid_mean[0]),
        .I1(dist_to_centroid_mean[2]),
        .I2(dist_to_centroid_mean[1]),
        .I3(dist_to_centroid_mean[4]),
        .I4(dist_to_centroid_mean[3]),
        .I5(dist_to_centroid_mean[5]),
        .O(\prediction[1]_i_25__4_n_0 ));
  LUT2 #(
    .INIT(4'h8)) 
    \prediction[1]_i_26__3 
       (.I0(dist_to_centroid_mean[10]),
        .I1(dist_to_centroid_mean[11]),
        .O(\prediction[1]_i_26__3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000555577F7)) 
    \prediction[1]_i_27__2 
       (.I0(mean_speed[9]),
        .I1(mean_speed[7]),
        .I2(\prediction[1]_i_35__1_n_0 ),
        .I3(mean_speed[6]),
        .I4(mean_speed[8]),
        .I5(mean_speed[10]),
        .O(\prediction[1]_i_27__2_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000001FFF)) 
    \prediction[1]_i_28__3 
       (.I0(kde_prob_night_mean[1]),
        .I1(\prediction[1]_i_36__1_n_0 ),
        .I2(kde_prob_night_mean[4]),
        .I3(kde_prob_night_mean[5]),
        .I4(kde_prob_night_mean_8_sn_1),
        .I5(kde_prob_night_mean[6]),
        .O(\prediction[1]_i_28__3_n_0 ));
  LUT6 #(
    .INIT(64'h01115555FFFFFFFF)) 
    \prediction[1]_i_29__1 
       (.I0(accelerate[9]),
        .I1(\prediction[1]_i_38_n_0 ),
        .I2(accelerate[1]),
        .I3(accelerate[0]),
        .I4(accelerate[8]),
        .I5(accelerate[10]),
        .O(\prediction[1]_i_29__1_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000000057)) 
    \prediction[1]_i_30__1 
       (.I0(kde_prob_night_mean[2]),
        .I1(kde_prob_night_mean[1]),
        .I2(kde_prob_night_mean[0]),
        .I3(kde_prob_night_mean[4]),
        .I4(kde_prob_night_mean[5]),
        .I5(kde_prob_night_mean[3]),
        .O(\prediction[1]_i_30__1_n_0 ));
  LUT5 #(
    .INIT(32'h80000000)) 
    \prediction[1]_i_31__0 
       (.I0(kde_prob_night_mean[8]),
        .I1(kde_prob_night_mean[11]),
        .I2(kde_prob_night_mean[12]),
        .I3(kde_prob_night_mean[9]),
        .I4(kde_prob_night_mean[10]),
        .O(\prediction[1]_i_31__0_n_0 ));
  LUT6 #(
    .INIT(64'h10FFFFFFFFFFFFFF)) 
    \prediction[1]_i_32__0 
       (.I0(accelerate[7]),
        .I1(accelerate[8]),
        .I2(\prediction[1]_i_39__1_n_0 ),
        .I3(\prediction[1]_i_20__1_0 ),
        .I4(accelerate[9]),
        .I5(accelerate[10]),
        .O(\prediction[1]_i_32__0_n_0 ));
  LUT6 #(
    .INIT(64'h01005555FFFFFFFF)) 
    \prediction[1]_i_33__0 
       (.I0(mean_speed_12_sn_1),
        .I1(mean_speed[9]),
        .I2(mean_speed[10]),
        .I3(\prediction[1]_i_40__1_n_0 ),
        .I4(mean_speed[11]),
        .I5(mean_speed[14]),
        .O(\prediction[1]_i_33__0_n_0 ));
  LUT6 #(
    .INIT(64'h00011111FFFFFFFF)) 
    \prediction[1]_i_34__1 
       (.I0(step_median[3]),
        .I1(step_median[4]),
        .I2(step_median[0]),
        .I3(step_median[1]),
        .I4(step_median[2]),
        .I5(step_median[5]),
        .O(\prediction[1]_i_34__1_n_0 ));
  LUT6 #(
    .INIT(64'h00000001FFFFFFFF)) 
    \prediction[1]_i_35__1 
       (.I0(mean_speed[0]),
        .I1(mean_speed[2]),
        .I2(mean_speed[1]),
        .I3(mean_speed[4]),
        .I4(mean_speed[3]),
        .I5(mean_speed[5]),
        .O(\prediction[1]_i_35__1_n_0 ));
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[1]_i_36__1 
       (.I0(kde_prob_night_mean[2]),
        .I1(kde_prob_night_mean[3]),
        .O(\prediction[1]_i_36__1_n_0 ));
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[1]_i_37__1 
       (.I0(mean_speed[12]),
        .I1(mean_speed[13]),
        .O(mean_speed_12_sn_1));
  LUT4 #(
    .INIT(16'hFFFE)) 
    \prediction[1]_i_37__2 
       (.I0(kde_prob_night_mean[8]),
        .I1(kde_prob_night_mean[7]),
        .I2(kde_prob_night_mean[10]),
        .I3(kde_prob_night_mean[9]),
        .O(kde_prob_night_mean_8_sn_1));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFFE)) 
    \prediction[1]_i_38 
       (.I0(accelerate[3]),
        .I1(accelerate[2]),
        .I2(accelerate[6]),
        .I3(accelerate[7]),
        .I4(accelerate[4]),
        .I5(accelerate[5]),
        .O(\prediction[1]_i_38_n_0 ));
  LUT6 #(
    .INIT(64'h10FFFFFFFFFFFFFF)) 
    \prediction[1]_i_39__1 
       (.I0(accelerate[2]),
        .I1(accelerate[3]),
        .I2(accelerate_1_sn_1),
        .I3(accelerate[5]),
        .I4(accelerate[6]),
        .I5(accelerate[4]),
        .O(\prediction[1]_i_39__1_n_0 ));
  LUT6 #(
    .INIT(64'hAAAAAAA8AAAAAAAA)) 
    \prediction[1]_i_4 
       (.I0(tree_out__0),
        .I1(dist_to_centroid_mean[13]),
        .I2(dist_to_centroid_mean[12]),
        .I3(dist_to_centroid_mean[15]),
        .I4(dist_to_centroid_mean[14]),
        .I5(\prediction[1]_i_14__3_n_0 ),
        .O(tree_out1_out));
  LUT6 #(
    .INIT(64'h000000007F7F7FFF)) 
    \prediction[1]_i_40 
       (.I0(\prediction[1]_i_44__0_n_0 ),
        .I1(accelerate[5]),
        .I2(accelerate[4]),
        .I3(accelerate[1]),
        .I4(accelerate[0]),
        .I5(accelerate[6]),
        .O(accelerate_5_sn_1));
  LUT5 #(
    .INIT(32'h777F7777)) 
    \prediction[1]_i_40__1 
       (.I0(mean_speed[8]),
        .I1(mean_speed[7]),
        .I2(mean_speed[5]),
        .I3(mean_speed[6]),
        .I4(\prediction[1]_i_41__0_n_0 ),
        .O(\prediction[1]_i_40__1_n_0 ));
  LUT5 #(
    .INIT(32'h0111FFFF)) 
    \prediction[1]_i_41__0 
       (.I0(mean_speed[2]),
        .I1(mean_speed[3]),
        .I2(mean_speed[1]),
        .I3(mean_speed[0]),
        .I4(mean_speed[4]),
        .O(\prediction[1]_i_41__0_n_0 ));
  LUT2 #(
    .INIT(4'h8)) 
    \prediction[1]_i_44__0 
       (.I0(accelerate[2]),
        .I1(accelerate[3]),
        .O(\prediction[1]_i_44__0_n_0 ));
  LUT6 #(
    .INIT(64'h45555555FFFFFFFF)) 
    \prediction[1]_i_5__3 
       (.I0(\prediction_reg[1]_5 ),
        .I1(\prediction[1]_i_15__3_n_0 ),
        .I2(kde_prob_night_mean[9]),
        .I3(kde_prob_night_mean[10]),
        .I4(kde_prob_night_mean[8]),
        .I5(kde_prob_night_mean[14]),
        .O(\prediction[1]_i_5__3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFB8FF0000B800)) 
    \prediction[1]_i_6__2 
       (.I0(\prediction[1]_i_16__4_n_0 ),
        .I1(\prediction[1]_i_17__5_n_0 ),
        .I2(\prediction[1]_i_18_n_0 ),
        .I3(\prediction[1]_i_19__3_n_0 ),
        .I4(kde_prob_night_mean[15]),
        .I5(tree_out4_out),
        .O(\prediction[1]_i_6__2_n_0 ));
  LUT6 #(
    .INIT(64'h01115555FFFFFFFF)) 
    \prediction[1]_i_7__1 
       (.I0(\prediction[1]_i_2_0 ),
        .I1(kde_prob_mean[2]),
        .I2(kde_prob_mean[1]),
        .I3(kde_prob_mean[0]),
        .I4(\prediction[1]_i_2_1 ),
        .I5(\prediction[1]_i_2_2 ),
        .O(\prediction[1]_i_7__1_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000000002)) 
    \prediction[1]_i_8__0 
       (.I0(\prediction[1]_i_22__5_n_0 ),
        .I1(step_median[13]),
        .I2(step_median[12]),
        .I3(step_median[15]),
        .I4(step_median[14]),
        .I5(step_median[11]),
        .O(\prediction[1]_i_8__0_n_0 ));
  LUT6 #(
    .INIT(64'h00015555FFFFFFFF)) 
    \prediction[1]_i_9__0 
       (.I0(\prediction[1]_i_2_0 ),
        .I1(kde_prob_mean[0]),
        .I2(kde_prob_mean[1]),
        .I3(\prediction[1]_i_23_n_0 ),
        .I4(kde_prob_mean[4]),
        .I5(\prediction[1]_i_2_2 ),
        .O(\prediction[1]_i_9__0_n_0 ));
  FDRE \prediction_reg[0] 
       (.C(clk),
        .CE(\prediction_reg[1]_6 ),
        .D(\prediction[0]_i_1_n_0 ),
        .Q(\prediction_reg[0]_0 ),
        .R(\prediction_reg[1]_1 ));
  FDRE \prediction_reg[1] 
       (.C(clk),
        .CE(\prediction_reg[1]_6 ),
        .D(tree_out),
        .Q(\prediction_reg[1]_0 ),
        .R(\prediction_reg[1]_1 ));
endmodule

(* ORIG_REF_NAME = "decision_tree_6" *) 
module design_1_random_forest_elepha_0_0_decision_tree_6
   (accelerate_14_sp_1,
    is_night_0_sp_1,
    accelerate_8_sp_1,
    mean_speed_2_sp_1,
    accelerate_9_sp_1,
    kde_prob_mean_3_sp_1,
    \prediction_reg[0]_0 ,
    \prediction_reg[0]_1 ,
    \prediction_reg[1]_0 ,
    done_reg_0,
    done_reg_1,
    \prediction_reg[1]_1 ,
    clk,
    mean_speed,
    turning_angle_max,
    kde_prob_night_mean,
    \prediction_reg[0]_i_4_0 ,
    \prediction_reg[0]_i_4_1 ,
    dist_to_centroid_mean,
    kde_prob_mean,
    \prediction[0]_i_14_0 ,
    accelerate,
    \prediction_reg[1]_2 ,
    \prediction[0]_i_3__1_0 ,
    \prediction[0]_i_3__1_1 ,
    turning_angle_median,
    \prediction[0]_i_14_1 ,
    \prediction[0]_i_30__0_0 ,
    is_night,
    start,
    \result[1]_i_2 ,
    \result[1]_i_2_0 ,
    \result[1]_i_2_1 ,
    \result[1]_i_2_2 ,
    \prediction_reg[1]_3 );
  output accelerate_14_sp_1;
  output is_night_0_sp_1;
  output accelerate_8_sp_1;
  output mean_speed_2_sp_1;
  output accelerate_9_sp_1;
  output kde_prob_mean_3_sp_1;
  output \prediction_reg[0]_0 ;
  output \prediction_reg[0]_1 ;
  output \prediction_reg[1]_0 ;
  output done_reg_0;
  input [2:0]done_reg_1;
  input \prediction_reg[1]_1 ;
  input clk;
  input [15:0]mean_speed;
  input [12:0]turning_angle_max;
  input [15:0]kde_prob_night_mean;
  input \prediction_reg[0]_i_4_0 ;
  input \prediction_reg[0]_i_4_1 ;
  input [6:0]dist_to_centroid_mean;
  input [10:0]kde_prob_mean;
  input \prediction[0]_i_14_0 ;
  input [15:0]accelerate;
  input \prediction_reg[1]_2 ;
  input \prediction[0]_i_3__1_0 ;
  input \prediction[0]_i_3__1_1 ;
  input [15:0]turning_angle_median;
  input \prediction[0]_i_14_1 ;
  input \prediction[0]_i_30__0_0 ;
  input [15:0]is_night;
  input [0:0]start;
  input \result[1]_i_2 ;
  input \result[1]_i_2_0 ;
  input \result[1]_i_2_1 ;
  input \result[1]_i_2_2 ;
  input \prediction_reg[1]_3 ;

  wire [15:0]accelerate;
  wire accelerate_14_sn_1;
  wire accelerate_8_sn_1;
  wire accelerate_9_sn_1;
  wire clk;
  wire [6:0]dist_to_centroid_mean;
  wire done_i_1__5_n_0;
  wire done_reg_0;
  wire [2:0]done_reg_1;
  wire [15:0]is_night;
  wire is_night_0_sn_1;
  wire [10:0]kde_prob_mean;
  wire kde_prob_mean_3_sn_1;
  wire [15:0]kde_prob_night_mean;
  wire [15:0]mean_speed;
  wire mean_speed_2_sn_1;
  wire \prediction[0]_i_10__2_n_0 ;
  wire \prediction[0]_i_12__2_n_0 ;
  wire \prediction[0]_i_14_0 ;
  wire \prediction[0]_i_14_1 ;
  wire \prediction[0]_i_14_n_0 ;
  wire \prediction[0]_i_15__2_n_0 ;
  wire \prediction[0]_i_16__2_n_0 ;
  wire \prediction[0]_i_17__2_n_0 ;
  wire \prediction[0]_i_18__1_n_0 ;
  wire \prediction[0]_i_19__1_n_0 ;
  wire \prediction[0]_i_1__0_n_0 ;
  wire \prediction[0]_i_20__1_n_0 ;
  wire \prediction[0]_i_23__1_n_0 ;
  wire \prediction[0]_i_24__0_n_0 ;
  wire \prediction[0]_i_25__0_n_0 ;
  wire \prediction[0]_i_26__0_n_0 ;
  wire \prediction[0]_i_27_n_0 ;
  wire \prediction[0]_i_29__0_n_0 ;
  wire \prediction[0]_i_30__0_0 ;
  wire \prediction[0]_i_30__0_n_0 ;
  wire \prediction[0]_i_31__0_n_0 ;
  wire \prediction[0]_i_32__0_n_0 ;
  wire \prediction[0]_i_33_n_0 ;
  wire \prediction[0]_i_34__0_n_0 ;
  wire \prediction[0]_i_35_n_0 ;
  wire \prediction[0]_i_36__0_n_0 ;
  wire \prediction[0]_i_37__0_n_0 ;
  wire \prediction[0]_i_38__0_n_0 ;
  wire \prediction[0]_i_39__0_n_0 ;
  wire \prediction[0]_i_3__1_0 ;
  wire \prediction[0]_i_3__1_1 ;
  wire \prediction[0]_i_5__2_n_0 ;
  wire \prediction[0]_i_6__1_n_0 ;
  wire \prediction[0]_i_7__2_n_0 ;
  wire \prediction[0]_i_8__1_n_0 ;
  wire \prediction[0]_i_9__1_n_0 ;
  wire \prediction[1]_i_10__3_n_0 ;
  wire \prediction[1]_i_10__4_n_0 ;
  wire \prediction[1]_i_11__2_n_0 ;
  wire \prediction[1]_i_11__4_n_0 ;
  wire \prediction[1]_i_12__3_n_0 ;
  wire \prediction[1]_i_12__4_n_0 ;
  wire \prediction[1]_i_13__4_n_0 ;
  wire \prediction[1]_i_14__2_n_0 ;
  wire \prediction[1]_i_15__4_n_0 ;
  wire \prediction[1]_i_16__5_n_0 ;
  wire \prediction[1]_i_17__1_n_0 ;
  wire \prediction[1]_i_19__1_n_0 ;
  wire \prediction[1]_i_20__4_n_0 ;
  wire \prediction[1]_i_21__5_n_0 ;
  wire \prediction[1]_i_22__3_n_0 ;
  wire \prediction[1]_i_23__3_n_0 ;
  wire \prediction[1]_i_2__4_n_0 ;
  wire \prediction[1]_i_3_n_0 ;
  wire \prediction[1]_i_4__2_n_0 ;
  wire \prediction[1]_i_5__5_n_0 ;
  wire \prediction[1]_i_6__4_n_0 ;
  wire \prediction[1]_i_8__2_n_0 ;
  wire \prediction_reg[0]_0 ;
  wire \prediction_reg[0]_1 ;
  wire \prediction_reg[0]_i_4_0 ;
  wire \prediction_reg[0]_i_4_1 ;
  wire \prediction_reg[0]_i_4_n_0 ;
  wire \prediction_reg[1]_0 ;
  wire \prediction_reg[1]_1 ;
  wire \prediction_reg[1]_2 ;
  wire \prediction_reg[1]_3 ;
  wire \result[1]_i_2 ;
  wire \result[1]_i_2_0 ;
  wire \result[1]_i_2_1 ;
  wire \result[1]_i_2_2 ;
  wire [0:0]start;
  wire [5:5]t_done;
  wire tree_out3_out;
  wire tree_out4_in;
  wire tree_out6_out;
  wire tree_out__0;
  wire [12:0]turning_angle_max;
  wire [15:0]turning_angle_median;

  assign accelerate_14_sp_1 = accelerate_14_sn_1;
  assign accelerate_8_sp_1 = accelerate_8_sn_1;
  assign accelerate_9_sp_1 = accelerate_9_sn_1;
  assign is_night_0_sp_1 = is_night_0_sn_1;
  assign kde_prob_mean_3_sp_1 = kde_prob_mean_3_sn_1;
  assign mean_speed_2_sp_1 = mean_speed_2_sn_1;
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT2 #(
    .INIT(4'hD)) 
    done_i_1__5
       (.I0(start),
        .I1(t_done),
        .O(done_i_1__5_n_0));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT4 #(
    .INIT(16'h7FFF)) 
    done_i_2
       (.I0(t_done),
        .I1(done_reg_1[0]),
        .I2(done_reg_1[2]),
        .I3(done_reg_1[1]),
        .O(done_reg_0));
  FDRE done_reg
       (.C(clk),
        .CE(1'b1),
        .D(done_i_1__5_n_0),
        .Q(t_done),
        .R(\prediction_reg[1]_1 ));
  LUT6 #(
    .INIT(64'h10FFFFFFFFFFFFFF)) 
    \prediction[0]_i_10__2 
       (.I0(accelerate[2]),
        .I1(\prediction[0]_i_20__1_n_0 ),
        .I2(\prediction[0]_i_3__1_0 ),
        .I3(accelerate[5]),
        .I4(accelerate[6]),
        .I5(\prediction[0]_i_3__1_1 ),
        .O(\prediction[0]_i_10__2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT3 #(
    .INIT(8'hFE)) 
    \prediction[0]_i_11__2 
       (.I0(accelerate[9]),
        .I1(accelerate[11]),
        .I2(accelerate[10]),
        .O(accelerate_9_sn_1));
  LUT6 #(
    .INIT(64'h10115555FFFFFFFF)) 
    \prediction[0]_i_12__2 
       (.I0(\prediction[0]_i_23__1_n_0 ),
        .I1(mean_speed[8]),
        .I2(\prediction[0]_i_24__0_n_0 ),
        .I3(mean_speed[7]),
        .I4(mean_speed[9]),
        .I5(mean_speed[15]),
        .O(\prediction[0]_i_12__2_n_0 ));
  LUT6 #(
    .INIT(64'h22222222222222A2)) 
    \prediction[0]_i_13 
       (.I0(\prediction[0]_i_25__0_n_0 ),
        .I1(mean_speed[15]),
        .I2(\prediction[0]_i_26__0_n_0 ),
        .I3(mean_speed[13]),
        .I4(mean_speed[14]),
        .I5(mean_speed[12]),
        .O(tree_out3_out));
  LUT6 #(
    .INIT(64'hFFFF22F200002202)) 
    \prediction[0]_i_14 
       (.I0(\prediction[0]_i_27_n_0 ),
        .I1(\prediction_reg[0]_i_4_0 ),
        .I2(\prediction_reg[0]_i_4_1 ),
        .I3(\prediction[0]_i_29__0_n_0 ),
        .I4(dist_to_centroid_mean[6]),
        .I5(\prediction[0]_i_30__0_n_0 ),
        .O(\prediction[0]_i_14_n_0 ));
  LUT6 #(
    .INIT(64'h01005555FFFFFFFF)) 
    \prediction[0]_i_15__2 
       (.I0(kde_prob_night_mean[9]),
        .I1(kde_prob_night_mean[6]),
        .I2(kde_prob_night_mean[7]),
        .I3(\prediction[0]_i_31__0_n_0 ),
        .I4(kde_prob_night_mean[8]),
        .I5(kde_prob_night_mean[10]),
        .O(\prediction[0]_i_15__2_n_0 ));
  LUT6 #(
    .INIT(64'h15555555FFFFFFFF)) 
    \prediction[0]_i_16__2 
       (.I0(turning_angle_max[4]),
        .I1(turning_angle_max[2]),
        .I2(turning_angle_max[3]),
        .I3(turning_angle_max[1]),
        .I4(turning_angle_max[0]),
        .I5(turning_angle_max[5]),
        .O(\prediction[0]_i_16__2_n_0 ));
  LUT5 #(
    .INIT(32'h80000000)) 
    \prediction[0]_i_17__2 
       (.I0(kde_prob_night_mean[10]),
        .I1(kde_prob_night_mean[11]),
        .I2(kde_prob_night_mean[12]),
        .I3(kde_prob_night_mean[13]),
        .I4(kde_prob_night_mean[14]),
        .O(\prediction[0]_i_17__2_n_0 ));
  LUT6 #(
    .INIT(64'h1011FFFFFFFFFFFF)) 
    \prediction[0]_i_18__1 
       (.I0(kde_prob_night_mean[3]),
        .I1(kde_prob_night_mean[4]),
        .I2(\prediction[0]_i_32__0_n_0 ),
        .I3(kde_prob_night_mean[2]),
        .I4(kde_prob_night_mean[6]),
        .I5(kde_prob_night_mean[5]),
        .O(\prediction[0]_i_18__1_n_0 ));
  LUT6 #(
    .INIT(64'h1011FFFFFFFFFFFF)) 
    \prediction[0]_i_19__1 
       (.I0(mean_speed[7]),
        .I1(mean_speed[8]),
        .I2(\prediction[0]_i_33_n_0 ),
        .I3(mean_speed[6]),
        .I4(mean_speed[10]),
        .I5(mean_speed[9]),
        .O(\prediction[0]_i_19__1_n_0 ));
  LUT6 #(
    .INIT(64'h001D001D001DFF1D)) 
    \prediction[0]_i_1__0 
       (.I0(tree_out6_out),
        .I1(accelerate_14_sn_1),
        .I2(\prediction_reg[0]_i_4_n_0 ),
        .I3(\prediction[1]_i_2__4_n_0 ),
        .I4(\prediction[0]_i_5__2_n_0 ),
        .I5(\prediction[0]_i_6__1_n_0 ),
        .O(\prediction[0]_i_1__0_n_0 ));
  LUT6 #(
    .INIT(64'h000000000010FF10)) 
    \prediction[0]_i_2 
       (.I0(turning_angle_max[11]),
        .I1(turning_angle_max[12]),
        .I2(\prediction[0]_i_7__2_n_0 ),
        .I3(\prediction[0]_i_8__1_n_0 ),
        .I4(is_night_0_sn_1),
        .I5(\prediction[0]_i_9__1_n_0 ),
        .O(tree_out6_out));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[0]_i_20__1 
       (.I0(accelerate[3]),
        .I1(accelerate[4]),
        .O(\prediction[0]_i_20__1_n_0 ));
  LUT5 #(
    .INIT(32'hFFFFFFFE)) 
    \prediction[0]_i_23__1 
       (.I0(mean_speed[10]),
        .I1(mean_speed[13]),
        .I2(mean_speed[14]),
        .I3(mean_speed[11]),
        .I4(mean_speed[12]),
        .O(\prediction[0]_i_23__1_n_0 ));
  LUT6 #(
    .INIT(64'h0000000055555557)) 
    \prediction[0]_i_24__0 
       (.I0(mean_speed[4]),
        .I1(mean_speed[2]),
        .I2(mean_speed[3]),
        .I3(mean_speed[1]),
        .I4(mean_speed[0]),
        .I5(\prediction[0]_i_34__0_n_0 ),
        .O(\prediction[0]_i_24__0_n_0 ));
  LUT6 #(
    .INIT(64'h000000000000555D)) 
    \prediction[0]_i_25__0 
       (.I0(accelerate[13]),
        .I1(\prediction[0]_i_35_n_0 ),
        .I2(accelerate[12]),
        .I3(accelerate[11]),
        .I4(accelerate[15]),
        .I5(accelerate[14]),
        .O(\prediction[0]_i_25__0_n_0 ));
  LUT6 #(
    .INIT(64'h777F7777777F777F)) 
    \prediction[0]_i_26__0 
       (.I0(mean_speed[11]),
        .I1(mean_speed[10]),
        .I2(mean_speed[8]),
        .I3(mean_speed[9]),
        .I4(\prediction[0]_i_36__0_n_0 ),
        .I5(mean_speed[7]),
        .O(\prediction[0]_i_26__0_n_0 ));
  LUT6 #(
    .INIT(64'h01001111FFFFFFFF)) 
    \prediction[0]_i_27 
       (.I0(kde_prob_mean[9]),
        .I1(kde_prob_mean[10]),
        .I2(kde_prob_mean[7]),
        .I3(\prediction[0]_i_37__0_n_0 ),
        .I4(kde_prob_mean[8]),
        .I5(\prediction[0]_i_14_0 ),
        .O(\prediction[0]_i_27_n_0 ));
  LUT6 #(
    .INIT(64'h0000000055557FFF)) 
    \prediction[0]_i_29__0 
       (.I0(dist_to_centroid_mean[4]),
        .I1(dist_to_centroid_mean[0]),
        .I2(dist_to_centroid_mean[1]),
        .I3(dist_to_centroid_mean[2]),
        .I4(dist_to_centroid_mean[3]),
        .I5(dist_to_centroid_mean[5]),
        .O(\prediction[0]_i_29__0_n_0 ));
  LUT6 #(
    .INIT(64'h00000000555577F7)) 
    \prediction[0]_i_30__0 
       (.I0(turning_angle_median[12]),
        .I1(turning_angle_median[10]),
        .I2(\prediction[0]_i_38__0_n_0 ),
        .I3(turning_angle_median[9]),
        .I4(turning_angle_median[11]),
        .I5(\prediction[0]_i_14_1 ),
        .O(\prediction[0]_i_30__0_n_0 ));
  LUT6 #(
    .INIT(64'h1FFFFFFFFFFFFFFF)) 
    \prediction[0]_i_31__0 
       (.I0(kde_prob_night_mean[0]),
        .I1(kde_prob_night_mean[1]),
        .I2(kde_prob_night_mean[4]),
        .I3(kde_prob_night_mean[5]),
        .I4(kde_prob_night_mean[2]),
        .I5(kde_prob_night_mean[3]),
        .O(\prediction[0]_i_31__0_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT2 #(
    .INIT(4'h1)) 
    \prediction[0]_i_32__0 
       (.I0(kde_prob_night_mean[1]),
        .I1(kde_prob_night_mean[0]),
        .O(\prediction[0]_i_32__0_n_0 ));
  LUT6 #(
    .INIT(64'h000000000000007F)) 
    \prediction[0]_i_33 
       (.I0(mean_speed[0]),
        .I1(mean_speed[1]),
        .I2(mean_speed[2]),
        .I3(mean_speed[4]),
        .I4(mean_speed[5]),
        .I5(mean_speed[3]),
        .O(\prediction[0]_i_33_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[0]_i_34__0 
       (.I0(mean_speed[5]),
        .I1(mean_speed[6]),
        .O(\prediction[0]_i_34__0_n_0 ));
  LUT6 #(
    .INIT(64'h01005555FFFFFFFF)) 
    \prediction[0]_i_35 
       (.I0(accelerate_8_sn_1),
        .I1(accelerate[5]),
        .I2(accelerate[6]),
        .I3(\prediction[0]_i_39__0_n_0 ),
        .I4(accelerate[7]),
        .I5(accelerate[10]),
        .O(\prediction[0]_i_35_n_0 ));
  LUT6 #(
    .INIT(64'h0000000057777777)) 
    \prediction[0]_i_36__0 
       (.I0(mean_speed[5]),
        .I1(mean_speed[4]),
        .I2(mean_speed[2]),
        .I3(mean_speed[3]),
        .I4(mean_speed[1]),
        .I5(mean_speed[6]),
        .O(\prediction[0]_i_36__0_n_0 ));
  LUT6 #(
    .INIT(64'h15FFFFFFFFFFFFFF)) 
    \prediction[0]_i_37__0 
       (.I0(kde_prob_mean[2]),
        .I1(kde_prob_mean[1]),
        .I2(kde_prob_mean[0]),
        .I3(kde_prob_mean[5]),
        .I4(kde_prob_mean[6]),
        .I5(kde_prob_mean_3_sn_1),
        .O(\prediction[0]_i_37__0_n_0 ));
  LUT6 #(
    .INIT(64'h0100FFFFFFFFFFFF)) 
    \prediction[0]_i_38__0 
       (.I0(turning_angle_median[4]),
        .I1(turning_angle_median[6]),
        .I2(turning_angle_median[5]),
        .I3(\prediction[0]_i_30__0_0 ),
        .I4(turning_angle_median[8]),
        .I5(turning_angle_median[7]),
        .O(\prediction[0]_i_38__0_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT5 #(
    .INIT(32'h01FFFFFF)) 
    \prediction[0]_i_39__0 
       (.I0(accelerate[0]),
        .I1(accelerate[1]),
        .I2(accelerate[2]),
        .I3(accelerate[4]),
        .I4(accelerate[3]),
        .O(\prediction[0]_i_39__0_n_0 ));
  LUT6 #(
    .INIT(64'h00000000555577F7)) 
    \prediction[0]_i_3__1 
       (.I0(accelerate[14]),
        .I1(accelerate[12]),
        .I2(\prediction[0]_i_10__2_n_0 ),
        .I3(accelerate_9_sn_1),
        .I4(accelerate[13]),
        .I5(accelerate[15]),
        .O(accelerate_14_sn_1));
  LUT6 #(
    .INIT(64'h0000000045555555)) 
    \prediction[0]_i_5__2 
       (.I0(\prediction[1]_i_11__2_n_0 ),
        .I1(\prediction[1]_i_10__3_n_0 ),
        .I2(turning_angle_median[14]),
        .I3(turning_angle_median[15]),
        .I4(turning_angle_median[13]),
        .I5(is_night_0_sn_1),
        .O(\prediction[0]_i_5__2_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000002FFF)) 
    \prediction[0]_i_6__1 
       (.I0(\prediction[0]_i_15__2_n_0 ),
        .I1(kde_prob_night_mean[11]),
        .I2(kde_prob_night_mean[12]),
        .I3(kde_prob_night_mean[13]),
        .I4(kde_prob_night_mean[15]),
        .I5(kde_prob_night_mean[14]),
        .O(\prediction[0]_i_6__1_n_0 ));
  LUT6 #(
    .INIT(64'h1055FFFFFFFFFFFF)) 
    \prediction[0]_i_7__2 
       (.I0(turning_angle_max[8]),
        .I1(turning_angle_max[6]),
        .I2(\prediction[0]_i_16__2_n_0 ),
        .I3(turning_angle_max[7]),
        .I4(turning_angle_max[10]),
        .I5(turning_angle_max[9]),
        .O(\prediction[0]_i_7__2_n_0 ));
  LUT6 #(
    .INIT(64'h00000000555577F7)) 
    \prediction[0]_i_8__1 
       (.I0(\prediction[0]_i_17__2_n_0 ),
        .I1(kde_prob_night_mean[8]),
        .I2(\prediction[0]_i_18__1_n_0 ),
        .I3(kde_prob_night_mean[7]),
        .I4(kde_prob_night_mean[9]),
        .I5(kde_prob_night_mean[15]),
        .O(\prediction[0]_i_8__1_n_0 ));
  LUT6 #(
    .INIT(64'h000000000000555D)) 
    \prediction[0]_i_9__1 
       (.I0(mean_speed[13]),
        .I1(\prediction[0]_i_19__1_n_0 ),
        .I2(mean_speed[12]),
        .I3(mean_speed[11]),
        .I4(mean_speed[15]),
        .I5(mean_speed[14]),
        .O(\prediction[0]_i_9__1_n_0 ));
  LUT6 #(
    .INIT(64'h000000005555555D)) 
    \prediction[1]_i_10__3 
       (.I0(turning_angle_median[11]),
        .I1(\prediction[1]_i_15__4_n_0 ),
        .I2(turning_angle_median[9]),
        .I3(turning_angle_median[10]),
        .I4(turning_angle_median[8]),
        .I5(turning_angle_median[12]),
        .O(\prediction[1]_i_10__3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFFE)) 
    \prediction[1]_i_10__4 
       (.I0(is_night[2]),
        .I1(is_night[1]),
        .I2(is_night[5]),
        .I3(is_night[6]),
        .I4(is_night[3]),
        .I5(is_night[4]),
        .O(\prediction[1]_i_10__4_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000002FFF)) 
    \prediction[1]_i_11__2 
       (.I0(\prediction[1]_i_16__5_n_0 ),
        .I1(kde_prob_night_mean[11]),
        .I2(kde_prob_night_mean[12]),
        .I3(kde_prob_night_mean[13]),
        .I4(kde_prob_night_mean[15]),
        .I5(kde_prob_night_mean[14]),
        .O(\prediction[1]_i_11__2_n_0 ));
  LUT4 #(
    .INIT(16'hFFFE)) 
    \prediction[1]_i_11__4 
       (.I0(is_night[12]),
        .I1(is_night[11]),
        .I2(is_night[14]),
        .I3(is_night[13]),
        .O(\prediction[1]_i_11__4_n_0 ));
  LUT2 #(
    .INIT(4'h8)) 
    \prediction[1]_i_12__3 
       (.I0(kde_prob_night_mean[2]),
        .I1(kde_prob_night_mean[3]),
        .O(\prediction[1]_i_12__3_n_0 ));
  LUT4 #(
    .INIT(16'hFFFE)) 
    \prediction[1]_i_12__4 
       (.I0(is_night[8]),
        .I1(is_night[7]),
        .I2(is_night[10]),
        .I3(is_night[9]),
        .O(\prediction[1]_i_12__4_n_0 ));
  LUT6 #(
    .INIT(64'hBFFFFFFFFFFFFFFF)) 
    \prediction[1]_i_13__4 
       (.I0(\prediction[1]_i_17__1_n_0 ),
        .I1(kde_prob_night_mean[14]),
        .I2(kde_prob_night_mean[13]),
        .I3(kde_prob_night_mean[12]),
        .I4(kde_prob_night_mean[11]),
        .I5(kde_prob_night_mean[10]),
        .O(\prediction[1]_i_13__4_n_0 ));
  LUT6 #(
    .INIT(64'h000000000000555D)) 
    \prediction[1]_i_14__2 
       (.I0(mean_speed[6]),
        .I1(mean_speed_2_sn_1),
        .I2(\prediction[1]_i_19__1_n_0 ),
        .I3(mean_speed[3]),
        .I4(mean_speed[8]),
        .I5(mean_speed[7]),
        .O(\prediction[1]_i_14__2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFF0155FFFFFFFF)) 
    \prediction[1]_i_15__4 
       (.I0(turning_angle_median[3]),
        .I1(turning_angle_median[0]),
        .I2(turning_angle_median[1]),
        .I3(turning_angle_median[2]),
        .I4(\prediction[1]_i_20__4_n_0 ),
        .I5(\prediction[1]_i_21__5_n_0 ),
        .O(\prediction[1]_i_15__4_n_0 ));
  LUT6 #(
    .INIT(64'h45FFFFFFFFFFFFFF)) 
    \prediction[1]_i_16__5 
       (.I0(kde_prob_night_mean[7]),
        .I1(\prediction[1]_i_22__3_n_0 ),
        .I2(kde_prob_night_mean[6]),
        .I3(kde_prob_night_mean[9]),
        .I4(kde_prob_night_mean[10]),
        .I5(kde_prob_night_mean[8]),
        .O(\prediction[1]_i_16__5_n_0 ));
  LUT6 #(
    .INIT(64'h000000005555FF7F)) 
    \prediction[1]_i_17__1 
       (.I0(kde_prob_night_mean[8]),
        .I1(kde_prob_night_mean[5]),
        .I2(kde_prob_night_mean[6]),
        .I3(\prediction[1]_i_23__3_n_0 ),
        .I4(kde_prob_night_mean[7]),
        .I5(kde_prob_night_mean[9]),
        .O(\prediction[1]_i_17__1_n_0 ));
  LUT3 #(
    .INIT(8'h7F)) 
    \prediction[1]_i_18__0 
       (.I0(mean_speed[2]),
        .I1(mean_speed[1]),
        .I2(mean_speed[0]),
        .O(mean_speed_2_sn_1));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[1]_i_19__1 
       (.I0(mean_speed[4]),
        .I1(mean_speed[5]),
        .O(\prediction[1]_i_19__1_n_0 ));
  LUT2 #(
    .INIT(4'h7)) 
    \prediction[1]_i_20__4 
       (.I0(turning_angle_median[6]),
        .I1(turning_angle_median[7]),
        .O(\prediction[1]_i_20__4_n_0 ));
  LUT2 #(
    .INIT(4'h8)) 
    \prediction[1]_i_21__2 
       (.I0(kde_prob_mean[3]),
        .I1(kde_prob_mean[4]),
        .O(kde_prob_mean_3_sn_1));
  LUT2 #(
    .INIT(4'h8)) 
    \prediction[1]_i_21__5 
       (.I0(turning_angle_median[4]),
        .I1(turning_angle_median[5]),
        .O(\prediction[1]_i_21__5_n_0 ));
  LUT6 #(
    .INIT(64'h0000000055555557)) 
    \prediction[1]_i_22__3 
       (.I0(kde_prob_night_mean[4]),
        .I1(kde_prob_night_mean[2]),
        .I2(kde_prob_night_mean[3]),
        .I3(kde_prob_night_mean[1]),
        .I4(kde_prob_night_mean[0]),
        .I5(kde_prob_night_mean[5]),
        .O(\prediction[1]_i_22__3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT5 #(
    .INIT(32'h00000057)) 
    \prediction[1]_i_23__3 
       (.I0(kde_prob_night_mean[2]),
        .I1(kde_prob_night_mean[1]),
        .I2(kde_prob_night_mean[0]),
        .I3(kde_prob_night_mean[4]),
        .I4(kde_prob_night_mean[3]),
        .O(\prediction[1]_i_23__3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[1]_i_29__0 
       (.I0(accelerate[8]),
        .I1(accelerate[9]),
        .O(accelerate_8_sn_1));
  LUT6 #(
    .INIT(64'h000000005555FF7F)) 
    \prediction[1]_i_2__4 
       (.I0(kde_prob_night_mean[14]),
        .I1(kde_prob_night_mean[7]),
        .I2(kde_prob_night_mean[8]),
        .I3(\prediction[1]_i_5__5_n_0 ),
        .I4(\prediction[1]_i_6__4_n_0 ),
        .I5(kde_prob_night_mean[15]),
        .O(\prediction[1]_i_2__4_n_0 ));
  LUT6 #(
    .INIT(64'hB8B8B888B8B8B8B8)) 
    \prediction[1]_i_3 
       (.I0(\prediction_reg[0]_i_4_n_0 ),
        .I1(accelerate_14_sn_1),
        .I2(tree_out4_in),
        .I3(mean_speed[14]),
        .I4(mean_speed[15]),
        .I5(\prediction[1]_i_8__2_n_0 ),
        .O(\prediction[1]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'h0001FFFF)) 
    \prediction[1]_i_3__4 
       (.I0(is_night[0]),
        .I1(\prediction[1]_i_10__4_n_0 ),
        .I2(\prediction[1]_i_11__4_n_0 ),
        .I3(\prediction[1]_i_12__4_n_0 ),
        .I4(is_night[15]),
        .O(is_night_0_sn_1));
  LUT5 #(
    .INIT(32'hAAAABBAB)) 
    \prediction[1]_i_4__2 
       (.I0(\prediction[0]_i_6__1_n_0 ),
        .I1(is_night_0_sn_1),
        .I2(\prediction_reg[1]_2 ),
        .I3(\prediction[1]_i_10__3_n_0 ),
        .I4(\prediction[1]_i_11__2_n_0 ),
        .O(\prediction[1]_i_4__2_n_0 ));
  LUT6 #(
    .INIT(64'h0000000055557FFF)) 
    \prediction[1]_i_5__5 
       (.I0(kde_prob_night_mean[5]),
        .I1(kde_prob_night_mean[0]),
        .I2(\prediction[1]_i_12__3_n_0 ),
        .I3(kde_prob_night_mean[1]),
        .I4(kde_prob_night_mean[4]),
        .I5(kde_prob_night_mean[6]),
        .O(\prediction[1]_i_5__5_n_0 ));
  LUT5 #(
    .INIT(32'hFFFFFFFE)) 
    \prediction[1]_i_6__4 
       (.I0(kde_prob_night_mean[9]),
        .I1(kde_prob_night_mean[12]),
        .I2(kde_prob_night_mean[13]),
        .I3(kde_prob_night_mean[10]),
        .I4(kde_prob_night_mean[11]),
        .O(\prediction[1]_i_6__4_n_0 ));
  LUT6 #(
    .INIT(64'h040404040404F704)) 
    \prediction[1]_i_7 
       (.I0(is_night_0_sn_1),
        .I1(\prediction[1]_i_13__4_n_0 ),
        .I2(kde_prob_night_mean[15]),
        .I3(\prediction[0]_i_7__2_n_0 ),
        .I4(turning_angle_max[12]),
        .I5(turning_angle_max[11]),
        .O(tree_out4_in));
  LUT6 #(
    .INIT(64'h10111111FFFFFFFF)) 
    \prediction[1]_i_8__2 
       (.I0(mean_speed[11]),
        .I1(mean_speed[12]),
        .I2(\prediction[1]_i_14__2_n_0 ),
        .I3(mean_speed[10]),
        .I4(mean_speed[9]),
        .I5(mean_speed[13]),
        .O(\prediction[1]_i_8__2_n_0 ));
  FDRE \prediction_reg[0] 
       (.C(clk),
        .CE(\prediction_reg[1]_3 ),
        .D(\prediction[0]_i_1__0_n_0 ),
        .Q(\prediction_reg[0]_1 ),
        .R(\prediction_reg[1]_1 ));
  MUXF7 \prediction_reg[0]_i_4 
       (.I0(tree_out3_out),
        .I1(\prediction[0]_i_14_n_0 ),
        .O(\prediction_reg[0]_i_4_n_0 ),
        .S(\prediction[0]_i_12__2_n_0 ));
  FDRE \prediction_reg[1] 
       (.C(clk),
        .CE(\prediction_reg[1]_3 ),
        .D(tree_out__0),
        .Q(\prediction_reg[1]_0 ),
        .R(\prediction_reg[1]_1 ));
  MUXF7 \prediction_reg[1]_i_1 
       (.I0(\prediction[1]_i_3_n_0 ),
        .I1(\prediction[1]_i_4__2_n_0 ),
        .O(tree_out__0),
        .S(\prediction[1]_i_2__4_n_0 ));
  LUT6 #(
    .INIT(64'h44B444B4BB4B44B4)) 
    \result[1]_i_6 
       (.I0(\prediction_reg[0]_1 ),
        .I1(\prediction_reg[1]_0 ),
        .I2(\result[1]_i_2 ),
        .I3(\result[1]_i_2_0 ),
        .I4(\result[1]_i_2_1 ),
        .I5(\result[1]_i_2_2 ),
        .O(\prediction_reg[0]_0 ));
endmodule

(* ORIG_REF_NAME = "decision_tree_7" *) 
module design_1_random_forest_elepha_0_0_decision_tree_7
   (done_reg_0,
    turning_angle_median_13_sp_1,
    \step_median[14] ,
    accelerate_3_sp_1,
    accelerate_10_sp_1,
    dist_to_centroid_mean_10_sp_1,
    turning_angle_median_0_sp_1,
    kde_prob_night_mean_9_sp_1,
    kde_prob_night_mean_4_sp_1,
    p_3_in,
    \prediction_reg[0]_0 ,
    clk,
    \prediction_reg[0]_1 ,
    mean_speed,
    step_median,
    \prediction[0]_i_4__0_0 ,
    accelerate,
    \prediction[0]_i_7_0 ,
    \prediction[0]_i_7_1 ,
    kde_prob_mean,
    kde_prob_night_mean,
    \prediction[0]_i_6_0 ,
    dist_to_centroid_mean,
    turning_angle_median,
    \prediction[0]_i_19__2_0 ,
    \prediction[0]_i_3__0_0 ,
    \prediction[0]_i_3__0_1 ,
    \prediction[0]_i_6_1 ,
    \prediction[0]_i_16__0_0 ,
    \prediction[0]_i_3__0_2 ,
    start,
    \prediction_reg[0]_2 );
  output [0:0]done_reg_0;
  output turning_angle_median_13_sp_1;
  output \step_median[14] ;
  output accelerate_3_sp_1;
  output accelerate_10_sp_1;
  output dist_to_centroid_mean_10_sp_1;
  output turning_angle_median_0_sp_1;
  output kde_prob_night_mean_9_sp_1;
  output kde_prob_night_mean_4_sp_1;
  output p_3_in;
  input \prediction_reg[0]_0 ;
  input clk;
  input \prediction_reg[0]_1 ;
  input [15:0]mean_speed;
  input [13:0]step_median;
  input \prediction[0]_i_4__0_0 ;
  input [15:0]accelerate;
  input \prediction[0]_i_7_0 ;
  input \prediction[0]_i_7_1 ;
  input [15:0]kde_prob_mean;
  input [15:0]kde_prob_night_mean;
  input \prediction[0]_i_6_0 ;
  input [15:0]dist_to_centroid_mean;
  input [14:0]turning_angle_median;
  input \prediction[0]_i_19__2_0 ;
  input \prediction[0]_i_3__0_0 ;
  input \prediction[0]_i_3__0_1 ;
  input \prediction[0]_i_6_1 ;
  input \prediction[0]_i_16__0_0 ;
  input \prediction[0]_i_3__0_2 ;
  input [1:0]start;
  input \prediction_reg[0]_2 ;

  wire [15:0]accelerate;
  wire accelerate_10_sn_1;
  wire accelerate_3_sn_1;
  wire clk;
  wire [15:0]dist_to_centroid_mean;
  wire dist_to_centroid_mean_10_sn_1;
  wire done_i_1__6_n_0;
  wire [0:0]done_reg_0;
  wire [15:0]kde_prob_mean;
  wire [15:0]kde_prob_night_mean;
  wire kde_prob_night_mean_4_sn_1;
  wire kde_prob_night_mean_9_sn_1;
  wire [15:0]mean_speed;
  wire p_3_in;
  wire \prediction[0]_i_10__1_n_0 ;
  wire \prediction[0]_i_11__0_n_0 ;
  wire \prediction[0]_i_12_n_0 ;
  wire \prediction[0]_i_13__2_n_0 ;
  wire \prediction[0]_i_14__0_n_0 ;
  wire \prediction[0]_i_15__1_n_0 ;
  wire \prediction[0]_i_16__0_0 ;
  wire \prediction[0]_i_16__0_n_0 ;
  wire \prediction[0]_i_16__1_n_0 ;
  wire \prediction[0]_i_17__1_n_0 ;
  wire \prediction[0]_i_17_n_0 ;
  wire \prediction[0]_i_18__0_n_0 ;
  wire \prediction[0]_i_19__2_0 ;
  wire \prediction[0]_i_1__1_n_0 ;
  wire \prediction[0]_i_20__0_n_0 ;
  wire \prediction[0]_i_21_n_0 ;
  wire \prediction[0]_i_22__0_n_0 ;
  wire \prediction[0]_i_23__0_n_0 ;
  wire \prediction[0]_i_24_n_0 ;
  wire \prediction[0]_i_25_n_0 ;
  wire \prediction[0]_i_26_n_0 ;
  wire \prediction[0]_i_27__0_n_0 ;
  wire \prediction[0]_i_28_n_0 ;
  wire \prediction[0]_i_29_n_0 ;
  wire \prediction[0]_i_2__0_n_0 ;
  wire \prediction[0]_i_30_n_0 ;
  wire \prediction[0]_i_31_n_0 ;
  wire \prediction[0]_i_32_n_0 ;
  wire \prediction[0]_i_33__0_n_0 ;
  wire \prediction[0]_i_34_n_0 ;
  wire \prediction[0]_i_35__0_n_0 ;
  wire \prediction[0]_i_36_n_0 ;
  wire \prediction[0]_i_37_n_0 ;
  wire \prediction[0]_i_38_n_0 ;
  wire \prediction[0]_i_39_n_0 ;
  wire \prediction[0]_i_3__0_0 ;
  wire \prediction[0]_i_3__0_1 ;
  wire \prediction[0]_i_3__0_2 ;
  wire \prediction[0]_i_3__0_n_0 ;
  wire \prediction[0]_i_40__0_n_0 ;
  wire \prediction[0]_i_42_n_0 ;
  wire \prediction[0]_i_43_n_0 ;
  wire \prediction[0]_i_45_n_0 ;
  wire \prediction[0]_i_46_n_0 ;
  wire \prediction[0]_i_47_n_0 ;
  wire \prediction[0]_i_4__0_0 ;
  wire \prediction[0]_i_4__0_n_0 ;
  wire \prediction[0]_i_6_0 ;
  wire \prediction[0]_i_6_1 ;
  wire \prediction[0]_i_6_n_0 ;
  wire \prediction[0]_i_7_0 ;
  wire \prediction[0]_i_7_1 ;
  wire \prediction[0]_i_7_n_0 ;
  wire \prediction[0]_i_8__2_n_0 ;
  wire \prediction[0]_i_9__2_n_0 ;
  wire \prediction[1]_i_1_n_0 ;
  wire \prediction_reg[0]_0 ;
  wire \prediction_reg[0]_1 ;
  wire \prediction_reg[0]_2 ;
  wire \prediction_reg_n_0_[0] ;
  wire \prediction_reg_n_0_[1] ;
  wire [1:0]start;
  wire [13:0]step_median;
  wire \step_median[14] ;
  wire tree_out0_out;
  wire tree_out1;
  wire tree_out2_out;
  wire [14:0]turning_angle_median;
  wire turning_angle_median_0_sn_1;
  wire turning_angle_median_13_sn_1;

  assign accelerate_10_sp_1 = accelerate_10_sn_1;
  assign accelerate_3_sp_1 = accelerate_3_sn_1;
  assign dist_to_centroid_mean_10_sp_1 = dist_to_centroid_mean_10_sn_1;
  assign kde_prob_night_mean_4_sp_1 = kde_prob_night_mean_4_sn_1;
  assign kde_prob_night_mean_9_sp_1 = kde_prob_night_mean_9_sn_1;
  assign turning_angle_median_0_sp_1 = turning_angle_median_0_sn_1;
  assign turning_angle_median_13_sp_1 = turning_angle_median_13_sn_1;
  LUT2 #(
    .INIT(4'hD)) 
    done_i_1__6
       (.I0(start[1]),
        .I1(done_reg_0),
        .O(done_i_1__6_n_0));
  FDRE done_reg
       (.C(clk),
        .CE(1'b1),
        .D(done_i_1__6_n_0),
        .Q(done_reg_0),
        .R(\prediction_reg[0]_0 ));
  LUT5 #(
    .INIT(32'h000055F7)) 
    \prediction[0]_i_10__1 
       (.I0(accelerate[14]),
        .I1(accelerate[12]),
        .I2(\prediction[0]_i_25_n_0 ),
        .I3(accelerate[13]),
        .I4(accelerate[15]),
        .O(\prediction[0]_i_10__1_n_0 ));
  LUT6 #(
    .INIT(64'h000000005555555D)) 
    \prediction[0]_i_11__0 
       (.I0(kde_prob_mean[14]),
        .I1(\prediction[0]_i_26_n_0 ),
        .I2(kde_prob_mean[12]),
        .I3(kde_prob_mean[13]),
        .I4(kde_prob_mean[11]),
        .I5(kde_prob_mean[15]),
        .O(\prediction[0]_i_11__0_n_0 ));
  LUT6 #(
    .INIT(64'h000000000000557F)) 
    \prediction[0]_i_12 
       (.I0(\prediction[0]_i_4__0_0 ),
        .I1(step_median[0]),
        .I2(step_median[1]),
        .I3(step_median[2]),
        .I4(\prediction[0]_i_17_n_0 ),
        .I5(step_median[7]),
        .O(\prediction[0]_i_12_n_0 ));
  LUT6 #(
    .INIT(64'h0008000000000000)) 
    \prediction[0]_i_13__2 
       (.I0(kde_prob_night_mean[8]),
        .I1(kde_prob_night_mean[9]),
        .I2(\prediction[0]_i_27__0_n_0 ),
        .I3(\prediction[0]_i_28_n_0 ),
        .I4(kde_prob_night_mean[10]),
        .I5(kde_prob_night_mean[11]),
        .O(\prediction[0]_i_13__2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT5 #(
    .INIT(32'h0000007F)) 
    \prediction[0]_i_14__0 
       (.I0(kde_prob_night_mean[0]),
        .I1(kde_prob_night_mean[1]),
        .I2(kde_prob_night_mean[2]),
        .I3(kde_prob_night_mean[4]),
        .I4(kde_prob_night_mean[3]),
        .O(\prediction[0]_i_14__0_n_0 ));
  LUT6 #(
    .INIT(64'h1055000010551055)) 
    \prediction[0]_i_15 
       (.I0(kde_prob_night_mean[15]),
        .I1(\prediction[0]_i_6_0 ),
        .I2(\prediction[0]_i_29_n_0 ),
        .I3(kde_prob_night_mean[14]),
        .I4(dist_to_centroid_mean[15]),
        .I5(\prediction[0]_i_30_n_0 ),
        .O(tree_out2_out));
  LUT6 #(
    .INIT(64'h000000000000557F)) 
    \prediction[0]_i_15__1 
       (.I0(step_median[4]),
        .I1(step_median[1]),
        .I2(step_median[2]),
        .I3(step_median[3]),
        .I4(step_median[6]),
        .I5(step_median[5]),
        .O(\prediction[0]_i_15__1_n_0 ));
  LUT6 #(
    .INIT(64'h000000000000557F)) 
    \prediction[0]_i_16__0 
       (.I0(\prediction[0]_i_31_n_0 ),
        .I1(mean_speed[0]),
        .I2(mean_speed[1]),
        .I3(mean_speed[2]),
        .I4(mean_speed[13]),
        .I5(mean_speed[12]),
        .O(\prediction[0]_i_16__0_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[0]_i_16__1 
       (.I0(step_median[10]),
        .I1(step_median[11]),
        .O(\prediction[0]_i_16__1_n_0 ));
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[0]_i_17 
       (.I0(step_median[8]),
        .I1(step_median[9]),
        .O(\prediction[0]_i_17_n_0 ));
  LUT6 #(
    .INIT(64'h000000000001FFFF)) 
    \prediction[0]_i_17__1 
       (.I0(\prediction[0]_i_32_n_0 ),
        .I1(mean_speed[10]),
        .I2(mean_speed[11]),
        .I3(\prediction[0]_i_33__0_n_0 ),
        .I4(mean_speed[12]),
        .I5(mean_speed[13]),
        .O(\prediction[0]_i_17__1_n_0 ));
  LUT6 #(
    .INIT(64'h11101111FFFFFFFF)) 
    \prediction[0]_i_18__0 
       (.I0(turning_angle_median[11]),
        .I1(turning_angle_median[12]),
        .I2(\prediction[0]_i_34_n_0 ),
        .I3(\prediction[0]_i_6_1 ),
        .I4(turning_angle_median[9]),
        .I5(\prediction[0]_i_35__0_n_0 ),
        .O(\prediction[0]_i_18__0_n_0 ));
  LUT6 #(
    .INIT(64'h4440444440404040)) 
    \prediction[0]_i_19__2 
       (.I0(turning_angle_median_13_sn_1),
        .I1(\prediction[0]_i_36_n_0 ),
        .I2(mean_speed[15]),
        .I3(mean_speed[13]),
        .I4(\prediction[0]_i_37_n_0 ),
        .I5(mean_speed[14]),
        .O(tree_out0_out));
  LUT1 #(
    .INIT(2'h1)) 
    \prediction[0]_i_1__1 
       (.I0(\prediction[0]_i_2__0_n_0 ),
        .O(\prediction[0]_i_1__1_n_0 ));
  LUT6 #(
    .INIT(64'h01001111FFFFFFFF)) 
    \prediction[0]_i_20__0 
       (.I0(accelerate[9]),
        .I1(accelerate[10]),
        .I2(accelerate[6]),
        .I3(accelerate_3_sn_1),
        .I4(\prediction[0]_i_7_0 ),
        .I5(\prediction[0]_i_7_1 ),
        .O(\prediction[0]_i_20__0_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFFE)) 
    \prediction[0]_i_21 
       (.I0(accelerate[4]),
        .I1(accelerate_10_sn_1),
        .I2(accelerate[6]),
        .I3(accelerate[5]),
        .I4(accelerate[8]),
        .I5(accelerate[7]),
        .O(\prediction[0]_i_21_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[0]_i_22__0 
       (.I0(kde_prob_night_mean[11]),
        .I1(kde_prob_night_mean[12]),
        .O(\prediction[0]_i_22__0_n_0 ));
  LUT6 #(
    .INIT(64'h00000001FFFFFFFF)) 
    \prediction[0]_i_23__0 
       (.I0(kde_prob_night_mean[0]),
        .I1(kde_prob_night_mean[1]),
        .I2(kde_prob_night_mean_4_sn_1),
        .I3(kde_prob_night_mean[2]),
        .I4(kde_prob_night_mean[3]),
        .I5(kde_prob_night_mean[6]),
        .O(\prediction[0]_i_23__0_n_0 ));
  LUT6 #(
    .INIT(64'h000000007777777F)) 
    \prediction[0]_i_24 
       (.I0(turning_angle_median[3]),
        .I1(turning_angle_median[4]),
        .I2(turning_angle_median[2]),
        .I3(turning_angle_median[1]),
        .I4(turning_angle_median[0]),
        .I5(turning_angle_median[5]),
        .O(\prediction[0]_i_24_n_0 ));
  LUT6 #(
    .INIT(64'h00000000555555F7)) 
    \prediction[0]_i_25 
       (.I0(accelerate[10]),
        .I1(accelerate[7]),
        .I2(\prediction[0]_i_38_n_0 ),
        .I3(accelerate[9]),
        .I4(accelerate[8]),
        .I5(accelerate[11]),
        .O(\prediction[0]_i_25_n_0 ));
  LUT6 #(
    .INIT(64'h01005555FFFFFFFF)) 
    \prediction[0]_i_26 
       (.I0(kde_prob_mean[9]),
        .I1(kde_prob_mean[6]),
        .I2(kde_prob_mean[7]),
        .I3(\prediction[0]_i_39_n_0 ),
        .I4(kde_prob_mean[8]),
        .I5(kde_prob_mean[10]),
        .O(\prediction[0]_i_26_n_0 ));
  LUT2 #(
    .INIT(4'h7)) 
    \prediction[0]_i_27__0 
       (.I0(kde_prob_night_mean[6]),
        .I1(kde_prob_night_mean[7]),
        .O(\prediction[0]_i_27__0_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT2 #(
    .INIT(4'h7)) 
    \prediction[0]_i_28 
       (.I0(kde_prob_night_mean[12]),
        .I1(kde_prob_night_mean[13]),
        .O(\prediction[0]_i_28_n_0 ));
  LUT6 #(
    .INIT(64'h8000000000000000)) 
    \prediction[0]_i_28__0 
       (.I0(dist_to_centroid_mean[10]),
        .I1(dist_to_centroid_mean[9]),
        .I2(dist_to_centroid_mean[13]),
        .I3(dist_to_centroid_mean[14]),
        .I4(dist_to_centroid_mean[11]),
        .I5(dist_to_centroid_mean[12]),
        .O(dist_to_centroid_mean_10_sn_1));
  LUT6 #(
    .INIT(64'h01005555FFFFFFFF)) 
    \prediction[0]_i_29 
       (.I0(kde_prob_night_mean[8]),
        .I1(kde_prob_night_mean[5]),
        .I2(kde_prob_night_mean[6]),
        .I3(\prediction[0]_i_40__0_n_0 ),
        .I4(kde_prob_night_mean[7]),
        .I5(kde_prob_night_mean_9_sn_1),
        .O(\prediction[0]_i_29_n_0 ));
  LUT6 #(
    .INIT(64'hFEF2FEF2FEF20E02)) 
    \prediction[0]_i_2__0 
       (.I0(\prediction[0]_i_3__0_n_0 ),
        .I1(\prediction[0]_i_4__0_n_0 ),
        .I2(tree_out1),
        .I3(\prediction[0]_i_6_n_0 ),
        .I4(\prediction_reg[0]_1 ),
        .I5(\prediction[0]_i_7_n_0 ),
        .O(\prediction[0]_i_2__0_n_0 ));
  LUT6 #(
    .INIT(64'h45555555FFFFFFFF)) 
    \prediction[0]_i_30 
       (.I0(\prediction[0]_i_42_n_0 ),
        .I1(\prediction[0]_i_43_n_0 ),
        .I2(dist_to_centroid_mean[5]),
        .I3(dist_to_centroid_mean[6]),
        .I4(dist_to_centroid_mean[4]),
        .I5(dist_to_centroid_mean_10_sn_1),
        .O(\prediction[0]_i_30_n_0 ));
  LUT6 #(
    .INIT(64'h2000000000000000)) 
    \prediction[0]_i_31 
       (.I0(mean_speed[3]),
        .I1(\prediction[0]_i_16__0_0 ),
        .I2(mean_speed[5]),
        .I3(mean_speed[4]),
        .I4(mean_speed[7]),
        .I5(mean_speed[6]),
        .O(\prediction[0]_i_31_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT3 #(
    .INIT(8'hFE)) 
    \prediction[0]_i_32 
       (.I0(mean_speed[5]),
        .I1(mean_speed[7]),
        .I2(mean_speed[6]),
        .O(\prediction[0]_i_32_n_0 ));
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[0]_i_33__0 
       (.I0(mean_speed[8]),
        .I1(mean_speed[9]),
        .O(\prediction[0]_i_33__0_n_0 ));
  LUT6 #(
    .INIT(64'h00000000555577F7)) 
    \prediction[0]_i_34 
       (.I0(turning_angle_median[7]),
        .I1(turning_angle_median[5]),
        .I2(turning_angle_median_0_sn_1),
        .I3(turning_angle_median[4]),
        .I4(turning_angle_median[6]),
        .I5(turning_angle_median[8]),
        .O(\prediction[0]_i_34_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \prediction[0]_i_35__0 
       (.I0(turning_angle_median[13]),
        .I1(turning_angle_median[14]),
        .O(\prediction[0]_i_35__0_n_0 ));
  LUT6 #(
    .INIT(64'h15551515FFFFFFFF)) 
    \prediction[0]_i_36 
       (.I0(turning_angle_median[10]),
        .I1(turning_angle_median[9]),
        .I2(turning_angle_median[8]),
        .I3(\prediction[0]_i_19__2_0 ),
        .I4(\prediction[0]_i_45_n_0 ),
        .I5(\prediction[0]_i_3__0_0 ),
        .O(\prediction[0]_i_36_n_0 ));
  LUT6 #(
    .INIT(64'h10115555FFFFFFFF)) 
    \prediction[0]_i_37 
       (.I0(mean_speed[11]),
        .I1(mean_speed[9]),
        .I2(\prediction[0]_i_46_n_0 ),
        .I3(\prediction[0]_i_47_n_0 ),
        .I4(mean_speed[10]),
        .I5(mean_speed[12]),
        .O(\prediction[0]_i_37_n_0 ));
  LUT6 #(
    .INIT(64'h000000000001FFFF)) 
    \prediction[0]_i_38 
       (.I0(accelerate[2]),
        .I1(accelerate[1]),
        .I2(accelerate[4]),
        .I3(accelerate[3]),
        .I4(accelerate[5]),
        .I5(accelerate[6]),
        .O(\prediction[0]_i_38_n_0 ));
  LUT6 #(
    .INIT(64'h00000001FFFFFFFF)) 
    \prediction[0]_i_39 
       (.I0(kde_prob_mean[0]),
        .I1(kde_prob_mean[2]),
        .I2(kde_prob_mean[1]),
        .I3(kde_prob_mean[4]),
        .I4(kde_prob_mean[3]),
        .I5(kde_prob_mean[5]),
        .O(\prediction[0]_i_39_n_0 ));
  LUT6 #(
    .INIT(64'h00EF00EFFFEF00EF)) 
    \prediction[0]_i_3__0 
       (.I0(\prediction[0]_i_8__2_n_0 ),
        .I1(turning_angle_median_13_sn_1),
        .I2(\prediction[0]_i_9__2_n_0 ),
        .I3(\prediction[0]_i_10__1_n_0 ),
        .I4(\prediction[0]_i_11__0_n_0 ),
        .I5(\step_median[14] ),
        .O(\prediction[0]_i_3__0_n_0 ));
  LUT4 #(
    .INIT(16'h01FF)) 
    \prediction[0]_i_40 
       (.I0(turning_angle_median[0]),
        .I1(turning_angle_median[1]),
        .I2(turning_angle_median[2]),
        .I3(turning_angle_median[3]),
        .O(turning_angle_median_0_sn_1));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT5 #(
    .INIT(32'h15FFFFFF)) 
    \prediction[0]_i_40__0 
       (.I0(kde_prob_night_mean[2]),
        .I1(kde_prob_night_mean[1]),
        .I2(kde_prob_night_mean[0]),
        .I3(kde_prob_night_mean[4]),
        .I4(kde_prob_night_mean[3]),
        .O(\prediction[0]_i_40__0_n_0 ));
  LUT2 #(
    .INIT(4'h8)) 
    \prediction[0]_i_41 
       (.I0(kde_prob_night_mean[9]),
        .I1(kde_prob_night_mean[10]),
        .O(kde_prob_night_mean_9_sn_1));
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[0]_i_42 
       (.I0(dist_to_centroid_mean[7]),
        .I1(dist_to_centroid_mean[8]),
        .O(\prediction[0]_i_42_n_0 ));
  LUT4 #(
    .INIT(16'h0007)) 
    \prediction[0]_i_43 
       (.I0(dist_to_centroid_mean[0]),
        .I1(dist_to_centroid_mean[1]),
        .I2(dist_to_centroid_mean[3]),
        .I3(dist_to_centroid_mean[2]),
        .O(\prediction[0]_i_43_n_0 ));
  LUT6 #(
    .INIT(64'h0001FFFFFFFFFFFF)) 
    \prediction[0]_i_45 
       (.I0(turning_angle_median[0]),
        .I1(turning_angle_median[1]),
        .I2(turning_angle_median[3]),
        .I3(turning_angle_median[2]),
        .I4(turning_angle_median[5]),
        .I5(turning_angle_median[4]),
        .O(\prediction[0]_i_45_n_0 ));
  LUT4 #(
    .INIT(16'h001F)) 
    \prediction[0]_i_46 
       (.I0(mean_speed[1]),
        .I1(mean_speed[2]),
        .I2(mean_speed[3]),
        .I3(mean_speed[4]),
        .O(\prediction[0]_i_46_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \prediction[0]_i_47 
       (.I0(mean_speed[6]),
        .I1(mean_speed[5]),
        .I2(mean_speed[8]),
        .I3(mean_speed[7]),
        .O(\prediction[0]_i_47_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT5 #(
    .INIT(32'h000000F7)) 
    \prediction[0]_i_4__0 
       (.I0(step_median[10]),
        .I1(step_median[11]),
        .I2(\prediction[0]_i_12_n_0 ),
        .I3(step_median[13]),
        .I4(step_median[12]),
        .O(\prediction[0]_i_4__0_n_0 ));
  LUT5 #(
    .INIT(32'h000000F7)) 
    \prediction[0]_i_5__1 
       (.I0(kde_prob_night_mean[5]),
        .I1(\prediction[0]_i_13__2_n_0 ),
        .I2(\prediction[0]_i_14__0_n_0 ),
        .I3(kde_prob_night_mean[15]),
        .I4(kde_prob_night_mean[14]),
        .O(tree_out1));
  LUT6 #(
    .INIT(64'h00A200A200AE00A2)) 
    \prediction[0]_i_6 
       (.I0(tree_out2_out),
        .I1(mean_speed[14]),
        .I2(\prediction[0]_i_16__0_n_0 ),
        .I3(mean_speed[15]),
        .I4(\prediction[0]_i_17__1_n_0 ),
        .I5(\prediction[0]_i_18__0_n_0 ),
        .O(\prediction[0]_i_6_n_0 ));
  LUT6 #(
    .INIT(64'h000022AE0000EEAE)) 
    \prediction[0]_i_7 
       (.I0(tree_out0_out),
        .I1(accelerate[14]),
        .I2(\prediction[0]_i_20__0_n_0 ),
        .I3(accelerate[13]),
        .I4(accelerate[15]),
        .I5(\prediction[0]_i_21_n_0 ),
        .O(\prediction[0]_i_7_n_0 ));
  LUT6 #(
    .INIT(64'h00000000555555F7)) 
    \prediction[0]_i_8 
       (.I0(step_median[12]),
        .I1(step_median[7]),
        .I2(\prediction[0]_i_15__1_n_0 ),
        .I3(\prediction[0]_i_16__1_n_0 ),
        .I4(\prediction[0]_i_17_n_0 ),
        .I5(step_median[13]),
        .O(\step_median[14] ));
  LUT6 #(
    .INIT(64'h00010000FFFFFFFF)) 
    \prediction[0]_i_8__2 
       (.I0(\prediction[0]_i_3__0_2 ),
        .I1(kde_prob_night_mean[13]),
        .I2(kde_prob_night_mean[14]),
        .I3(\prediction[0]_i_22__0_n_0 ),
        .I4(\prediction[0]_i_23__0_n_0 ),
        .I5(kde_prob_night_mean[15]),
        .O(\prediction[0]_i_8__2_n_0 ));
  LUT6 #(
    .INIT(64'h54555555FFFFFFFF)) 
    \prediction[0]_i_9__2 
       (.I0(turning_angle_median[10]),
        .I1(\prediction[0]_i_24_n_0 ),
        .I2(\prediction[0]_i_3__0_1 ),
        .I3(turning_angle_median[6]),
        .I4(turning_angle_median[7]),
        .I5(\prediction[0]_i_3__0_0 ),
        .O(\prediction[0]_i_9__2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT4 #(
    .INIT(16'hE020)) 
    \prediction[1]_i_1 
       (.I0(\prediction[0]_i_2__0_n_0 ),
        .I1(start[1]),
        .I2(start[0]),
        .I3(\prediction_reg_n_0_[1] ),
        .O(\prediction[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT3 #(
    .INIT(8'hFE)) 
    \prediction[1]_i_18__3 
       (.I0(turning_angle_median[12]),
        .I1(turning_angle_median[14]),
        .I2(turning_angle_median[13]),
        .O(turning_angle_median_13_sn_1));
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[1]_i_21__4 
       (.I0(kde_prob_night_mean[4]),
        .I1(kde_prob_night_mean[5]),
        .O(kde_prob_night_mean_4_sn_1));
  LUT6 #(
    .INIT(64'h01111111FFFFFFFF)) 
    \prediction[1]_i_38__0 
       (.I0(accelerate[3]),
        .I1(accelerate[4]),
        .I2(accelerate[2]),
        .I3(accelerate[1]),
        .I4(accelerate[0]),
        .I5(accelerate[5]),
        .O(accelerate_3_sn_1));
  LUT4 #(
    .INIT(16'hFFFE)) 
    \prediction[1]_i_39__0 
       (.I0(accelerate[10]),
        .I1(accelerate[9]),
        .I2(accelerate[12]),
        .I3(accelerate[11]),
        .O(accelerate_10_sn_1));
  FDRE \prediction_reg[0] 
       (.C(clk),
        .CE(\prediction_reg[0]_2 ),
        .D(\prediction[0]_i_1__1_n_0 ),
        .Q(\prediction_reg_n_0_[0] ),
        .R(\prediction_reg[0]_0 ));
  FDRE \prediction_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(\prediction[1]_i_1_n_0 ),
        .Q(\prediction_reg_n_0_[1] ),
        .R(1'b0));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \result[1]_i_7 
       (.I0(\prediction_reg_n_0_[1] ),
        .I1(\prediction_reg_n_0_[0] ),
        .O(p_3_in));
endmodule

(* ORIG_REF_NAME = "decision_tree_8" *) 
module design_1_random_forest_elepha_0_0_decision_tree_8
   (done_reg_0,
    kde_prob_mean_11_sp_1,
    \kde_prob_mean[14] ,
    step_median_9_sp_1,
    turning_angle_max_13_sp_1,
    accelerate_5_sp_1,
    turning_angle_median_8_sp_1,
    \start[1] ,
    \prediction_reg[1]_0 ,
    \prediction_reg[1]_1 ,
    clk,
    \prediction_reg[0]_0 ,
    \prediction_reg[0]_1 ,
    accelerate,
    turning_angle_max,
    kde_prob_mean,
    \prediction[0]_i_2__1 ,
    \prediction[0]_i_9_0 ,
    \prediction_reg[1]_2 ,
    step_median,
    \prediction_reg[1]_i_4_0 ,
    \prediction[1]_i_7__2_0 ,
    \prediction_reg[1]_3 ,
    turning_angle_median,
    \prediction_reg[1]_4 ,
    mean_speed,
    \prediction[1]_i_17_0 ,
    \prediction[1]_i_17_1 ,
    dist_to_centroid_mean,
    \prediction[1]_i_2__2_0 ,
    start);
  output [0:0]done_reg_0;
  output kde_prob_mean_11_sp_1;
  output \kde_prob_mean[14] ;
  output step_median_9_sp_1;
  output turning_angle_max_13_sp_1;
  output accelerate_5_sp_1;
  output turning_angle_median_8_sp_1;
  output \start[1] ;
  output \prediction_reg[1]_0 ;
  input \prediction_reg[1]_1 ;
  input clk;
  input \prediction_reg[0]_0 ;
  input \prediction_reg[0]_1 ;
  input [15:0]accelerate;
  input [15:0]turning_angle_max;
  input [11:0]kde_prob_mean;
  input \prediction[0]_i_2__1 ;
  input \prediction[0]_i_9_0 ;
  input \prediction_reg[1]_2 ;
  input [11:0]step_median;
  input \prediction_reg[1]_i_4_0 ;
  input \prediction[1]_i_7__2_0 ;
  input \prediction_reg[1]_3 ;
  input [10:0]turning_angle_median;
  input \prediction_reg[1]_4 ;
  input [8:0]mean_speed;
  input \prediction[1]_i_17_0 ;
  input \prediction[1]_i_17_1 ;
  input [14:0]dist_to_centroid_mean;
  input \prediction[1]_i_2__2_0 ;
  input [0:0]start;

  wire [15:0]accelerate;
  wire accelerate_5_sn_1;
  wire clk;
  wire [14:0]dist_to_centroid_mean;
  wire done_i_1__7_n_0;
  wire [0:0]done_reg_0;
  wire [11:0]kde_prob_mean;
  wire \kde_prob_mean[14] ;
  wire kde_prob_mean_11_sn_1;
  wire [8:0]mean_speed;
  wire \prediction[0]_i_19_n_0 ;
  wire \prediction[0]_i_1__6_n_0 ;
  wire \prediction[0]_i_2__1 ;
  wire \prediction[0]_i_9_0 ;
  wire \prediction[1]_i_10__5_n_0 ;
  wire \prediction[1]_i_12__2_n_0 ;
  wire \prediction[1]_i_13__2_n_0 ;
  wire \prediction[1]_i_15__0_n_0 ;
  wire \prediction[1]_i_16__1_n_0 ;
  wire \prediction[1]_i_17_0 ;
  wire \prediction[1]_i_17_1 ;
  wire \prediction[1]_i_17_n_0 ;
  wire \prediction[1]_i_18__1_n_0 ;
  wire \prediction[1]_i_19__0_n_0 ;
  wire \prediction[1]_i_20__0_n_0 ;
  wire \prediction[1]_i_21__0_n_0 ;
  wire \prediction[1]_i_22__0_n_0 ;
  wire \prediction[1]_i_23__2_n_0 ;
  wire \prediction[1]_i_24__0_n_0 ;
  wire \prediction[1]_i_25__1_n_0 ;
  wire \prediction[1]_i_26__1_n_0 ;
  wire \prediction[1]_i_27__0_n_0 ;
  wire \prediction[1]_i_28__1_n_0 ;
  wire \prediction[1]_i_2__2_0 ;
  wire \prediction[1]_i_2__2_n_0 ;
  wire \prediction[1]_i_3__3_n_0 ;
  wire \prediction[1]_i_5__1_n_0 ;
  wire \prediction[1]_i_6__1_n_0 ;
  wire \prediction[1]_i_7__2_0 ;
  wire \prediction[1]_i_7__2_n_0 ;
  wire \prediction[1]_i_8_n_0 ;
  wire \prediction_reg[0]_0 ;
  wire \prediction_reg[0]_1 ;
  wire \prediction_reg[1]_0 ;
  wire \prediction_reg[1]_1 ;
  wire \prediction_reg[1]_2 ;
  wire \prediction_reg[1]_3 ;
  wire \prediction_reg[1]_4 ;
  wire \prediction_reg[1]_i_4_0 ;
  wire \prediction_reg[1]_i_4_n_0 ;
  wire \prediction_reg_n_0_[0] ;
  wire \prediction_reg_n_0_[1] ;
  wire [0:0]start;
  wire \start[1] ;
  wire [11:0]step_median;
  wire step_median_9_sn_1;
  wire tree_out2_out;
  wire tree_out__1;
  wire [15:0]turning_angle_max;
  wire turning_angle_max_13_sn_1;
  wire [10:0]turning_angle_median;
  wire turning_angle_median_8_sn_1;

  assign accelerate_5_sp_1 = accelerate_5_sn_1;
  assign kde_prob_mean_11_sp_1 = kde_prob_mean_11_sn_1;
  assign step_median_9_sp_1 = step_median_9_sn_1;
  assign turning_angle_max_13_sp_1 = turning_angle_max_13_sn_1;
  assign turning_angle_median_8_sp_1 = turning_angle_median_8_sn_1;
  LUT2 #(
    .INIT(4'hD)) 
    done_i_1__7
       (.I0(start),
        .I1(done_reg_0),
        .O(done_i_1__7_n_0));
  FDRE done_reg
       (.C(clk),
        .CE(1'b1),
        .D(done_i_1__7_n_0),
        .Q(done_reg_0),
        .R(\prediction_reg[1]_1 ));
  LUT3 #(
    .INIT(8'h80)) 
    \prediction[0]_i_10__0 
       (.I0(turning_angle_max[13]),
        .I1(turning_angle_max[15]),
        .I2(turning_angle_max[14]),
        .O(turning_angle_max_13_sn_1));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFFE)) 
    \prediction[0]_i_19 
       (.I0(\prediction[0]_i_9_0 ),
        .I1(kde_prob_mean[1]),
        .I2(kde_prob_mean[2]),
        .I3(\prediction_reg[0]_1 ),
        .I4(kde_prob_mean[5]),
        .I5(kde_prob_mean[6]),
        .O(\prediction[0]_i_19_n_0 ));
  LUT6 #(
    .INIT(64'h5455444457557777)) 
    \prediction[0]_i_1__6 
       (.I0(\prediction_reg[1]_i_4_n_0 ),
        .I1(\prediction_reg[0]_0 ),
        .I2(\prediction_reg[0]_1 ),
        .I3(\prediction[1]_i_3__3_n_0 ),
        .I4(kde_prob_mean_11_sn_1),
        .I5(\prediction[1]_i_2__2_n_0 ),
        .O(\prediction[0]_i_1__6_n_0 ));
  LUT6 #(
    .INIT(64'h00000000555577F7)) 
    \prediction[0]_i_9 
       (.I0(kde_prob_mean[10]),
        .I1(kde_prob_mean_11_sn_1),
        .I2(\prediction[0]_i_2__1 ),
        .I3(\prediction[0]_i_19_n_0 ),
        .I4(kde_prob_mean[9]),
        .I5(kde_prob_mean[11]),
        .O(\kde_prob_mean[14] ));
  LUT5 #(
    .INIT(32'h0000007F)) 
    \prediction[1]_i_10__5 
       (.I0(turning_angle_median[0]),
        .I1(turning_angle_median[1]),
        .I2(turning_angle_median[2]),
        .I3(turning_angle_median[4]),
        .I4(turning_angle_median[3]),
        .O(\prediction[1]_i_10__5_n_0 ));
  LUT2 #(
    .INIT(4'h7)) 
    \prediction[1]_i_11__1 
       (.I0(turning_angle_median[8]),
        .I1(turning_angle_median[9]),
        .O(turning_angle_median_8_sn_1));
  LUT5 #(
    .INIT(32'hBFFFFFFF)) 
    \prediction[1]_i_12__2 
       (.I0(\prediction[1]_i_20__0_n_0 ),
        .I1(turning_angle_max[6]),
        .I2(turning_angle_max[7]),
        .I3(turning_angle_max[4]),
        .I4(turning_angle_max[5]),
        .O(\prediction[1]_i_12__2_n_0 ));
  LUT6 #(
    .INIT(64'h000000000000555D)) 
    \prediction[1]_i_13__2 
       (.I0(step_median[2]),
        .I1(\prediction[1]_i_7__2_0 ),
        .I2(step_median[1]),
        .I3(step_median[0]),
        .I4(step_median[4]),
        .I5(step_median[3]),
        .O(\prediction[1]_i_13__2_n_0 ));
  LUT4 #(
    .INIT(16'hFFFE)) 
    \prediction[1]_i_14__1 
       (.I0(step_median[7]),
        .I1(step_median[6]),
        .I2(step_median[9]),
        .I3(step_median[8]),
        .O(step_median_9_sn_1));
  LUT6 #(
    .INIT(64'h10115555FFFFFFFF)) 
    \prediction[1]_i_15__0 
       (.I0(accelerate[13]),
        .I1(accelerate[11]),
        .I2(\prediction[1]_i_21__0_n_0 ),
        .I3(accelerate[10]),
        .I4(accelerate[12]),
        .I5(accelerate[14]),
        .O(\prediction[1]_i_15__0_n_0 ));
  LUT6 #(
    .INIT(64'h000000007777FF7F)) 
    \prediction[1]_i_16__1 
       (.I0(turning_angle_max[11]),
        .I1(turning_angle_max[12]),
        .I2(\prediction[1]_i_22__0_n_0 ),
        .I3(\prediction[1]_i_23__2_n_0 ),
        .I4(turning_angle_max[10]),
        .I5(turning_angle_max[13]),
        .O(\prediction[1]_i_16__1_n_0 ));
  LUT6 #(
    .INIT(64'h000000000000005D)) 
    \prediction[1]_i_17 
       (.I0(mean_speed[5]),
        .I1(\prediction[1]_i_24__0_n_0 ),
        .I2(mean_speed[4]),
        .I3(mean_speed[7]),
        .I4(mean_speed[8]),
        .I5(mean_speed[6]),
        .O(\prediction[1]_i_17_n_0 ));
  LUT6 #(
    .INIT(64'h00000000555577F7)) 
    \prediction[1]_i_18__1 
       (.I0(dist_to_centroid_mean[13]),
        .I1(dist_to_centroid_mean[11]),
        .I2(\prediction[1]_i_25__1_n_0 ),
        .I3(dist_to_centroid_mean[10]),
        .I4(dist_to_centroid_mean[12]),
        .I5(dist_to_centroid_mean[14]),
        .O(\prediction[1]_i_18__1_n_0 ));
  LUT6 #(
    .INIT(64'h10555555FFFFFFFF)) 
    \prediction[1]_i_19__0 
       (.I0(accelerate[11]),
        .I1(accelerate[8]),
        .I2(\prediction[1]_i_26__1_n_0 ),
        .I3(accelerate[10]),
        .I4(accelerate[9]),
        .I5(accelerate[12]),
        .O(\prediction[1]_i_19__0_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFEEAE000022A2)) 
    \prediction[1]_i_1__2 
       (.I0(\prediction[1]_i_2__2_n_0 ),
        .I1(kde_prob_mean_11_sn_1),
        .I2(\prediction[1]_i_3__3_n_0 ),
        .I3(\prediction_reg[0]_1 ),
        .I4(\prediction_reg[0]_0 ),
        .I5(\prediction_reg[1]_i_4_n_0 ),
        .O(tree_out__1));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT4 #(
    .INIT(16'h0001)) 
    \prediction[1]_i_20__0 
       (.I0(turning_angle_max[2]),
        .I1(turning_angle_max[3]),
        .I2(turning_angle_max[1]),
        .I3(turning_angle_max[0]),
        .O(\prediction[1]_i_20__0_n_0 ));
  LUT6 #(
    .INIT(64'h0000000055557FFF)) 
    \prediction[1]_i_21__0 
       (.I0(accelerate[4]),
        .I1(accelerate[0]),
        .I2(accelerate[1]),
        .I3(accelerate[2]),
        .I4(accelerate[3]),
        .I5(\prediction[1]_i_27__0_n_0 ),
        .O(\prediction[1]_i_21__0_n_0 ));
  LUT5 #(
    .INIT(32'h80000000)) 
    \prediction[1]_i_22__0 
       (.I0(turning_angle_max[5]),
        .I1(turning_angle_max[8]),
        .I2(turning_angle_max[9]),
        .I3(turning_angle_max[6]),
        .I4(turning_angle_max[7]),
        .O(\prediction[1]_i_22__0_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT5 #(
    .INIT(32'h0000007F)) 
    \prediction[1]_i_23__2 
       (.I0(turning_angle_max[0]),
        .I1(turning_angle_max[1]),
        .I2(turning_angle_max[2]),
        .I3(turning_angle_max[4]),
        .I4(turning_angle_max[3]),
        .O(\prediction[1]_i_23__2_n_0 ));
  LUT6 #(
    .INIT(64'h10FFFFFFFFFFFFFF)) 
    \prediction[1]_i_24__0 
       (.I0(mean_speed[0]),
        .I1(\prediction[1]_i_17_0 ),
        .I2(\prediction[1]_i_17_1 ),
        .I3(mean_speed[2]),
        .I4(mean_speed[3]),
        .I5(mean_speed[1]),
        .O(\prediction[1]_i_24__0_n_0 ));
  LUT6 #(
    .INIT(64'h1115FFFFFFFFFFFF)) 
    \prediction[1]_i_25__1 
       (.I0(\prediction[1]_i_28__1_n_0 ),
        .I1(dist_to_centroid_mean[2]),
        .I2(dist_to_centroid_mean[1]),
        .I3(dist_to_centroid_mean[0]),
        .I4(dist_to_centroid_mean[9]),
        .I5(dist_to_centroid_mean[8]),
        .O(\prediction[1]_i_25__1_n_0 ));
  LUT6 #(
    .INIT(64'h15555555FFFFFFFF)) 
    \prediction[1]_i_26__1 
       (.I0(accelerate_5_sn_1),
        .I1(accelerate[3]),
        .I2(accelerate[4]),
        .I3(accelerate[2]),
        .I4(accelerate[1]),
        .I5(accelerate[7]),
        .O(\prediction[1]_i_26__1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT5 #(
    .INIT(32'hFFFFFFFE)) 
    \prediction[1]_i_27__0 
       (.I0(accelerate[5]),
        .I1(accelerate[8]),
        .I2(accelerate[9]),
        .I3(accelerate[6]),
        .I4(accelerate[7]),
        .O(\prediction[1]_i_27__0_n_0 ));
  LUT5 #(
    .INIT(32'hFFFFFFFE)) 
    \prediction[1]_i_28__1 
       (.I0(dist_to_centroid_mean[3]),
        .I1(dist_to_centroid_mean[6]),
        .I2(dist_to_centroid_mean[7]),
        .I3(dist_to_centroid_mean[4]),
        .I4(dist_to_centroid_mean[5]),
        .O(\prediction[1]_i_28__1_n_0 ));
  LUT6 #(
    .INIT(64'hFF8AFFFFFFFFFFFF)) 
    \prediction[1]_i_2__2 
       (.I0(\prediction_reg[1]_3 ),
        .I1(turning_angle_median[10]),
        .I2(\prediction[1]_i_5__1_n_0 ),
        .I3(\prediction[1]_i_6__1_n_0 ),
        .I4(turning_angle_max_13_sn_1),
        .I5(\prediction_reg[1]_4 ),
        .O(\prediction[1]_i_2__2_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \prediction[1]_i_2__5 
       (.I0(start),
        .O(\start[1] ));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT2 #(
    .INIT(4'hE)) 
    \prediction[1]_i_35__0 
       (.I0(accelerate[5]),
        .I1(accelerate[6]),
        .O(accelerate_5_sn_1));
  LUT6 #(
    .INIT(64'h1115FFFFFFFFFFFF)) 
    \prediction[1]_i_3__3 
       (.I0(kde_prob_mean[3]),
        .I1(kde_prob_mean[2]),
        .I2(kde_prob_mean[1]),
        .I3(kde_prob_mean[0]),
        .I4(\prediction_reg[1]_2 ),
        .I5(kde_prob_mean[4]),
        .O(\prediction[1]_i_3__3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFF4555)) 
    \prediction[1]_i_5__1 
       (.I0(turning_angle_median[7]),
        .I1(\prediction[1]_i_10__5_n_0 ),
        .I2(turning_angle_median[6]),
        .I3(turning_angle_median[5]),
        .I4(\prediction[1]_i_2__2_0 ),
        .I5(turning_angle_median_8_sn_1),
        .O(\prediction[1]_i_5__1_n_0 ));
  LUT2 #(
    .INIT(4'h8)) 
    \prediction[1]_i_6__0 
       (.I0(kde_prob_mean[7]),
        .I1(kde_prob_mean[8]),
        .O(kde_prob_mean_11_sn_1));
  LUT6 #(
    .INIT(64'h00000000777777F7)) 
    \prediction[1]_i_6__1 
       (.I0(turning_angle_max[10]),
        .I1(turning_angle_max[11]),
        .I2(\prediction[1]_i_12__2_n_0 ),
        .I3(turning_angle_max[9]),
        .I4(turning_angle_max[8]),
        .I5(turning_angle_max[12]),
        .O(\prediction[1]_i_6__1_n_0 ));
  LUT6 #(
    .INIT(64'h000000007777FF7F)) 
    \prediction[1]_i_7__2 
       (.I0(step_median[10]),
        .I1(step_median[11]),
        .I2(step_median[5]),
        .I3(\prediction[1]_i_13__2_n_0 ),
        .I4(step_median_9_sn_1),
        .I5(\prediction_reg[1]_i_4_0 ),
        .O(\prediction[1]_i_7__2_n_0 ));
  LUT6 #(
    .INIT(64'hD0D0D0D0FFD0D0D0)) 
    \prediction[1]_i_8 
       (.I0(\prediction[1]_i_15__0_n_0 ),
        .I1(accelerate[15]),
        .I2(\kde_prob_mean[14] ),
        .I3(turning_angle_max[14]),
        .I4(turning_angle_max[15]),
        .I5(\prediction[1]_i_16__1_n_0 ),
        .O(\prediction[1]_i_8_n_0 ));
  LUT6 #(
    .INIT(64'h1110111110101010)) 
    \prediction[1]_i_9__2 
       (.I0(\prediction[1]_i_17_n_0 ),
        .I1(\prediction[1]_i_18__1_n_0 ),
        .I2(accelerate[15]),
        .I3(accelerate[13]),
        .I4(\prediction[1]_i_19__0_n_0 ),
        .I5(accelerate[14]),
        .O(tree_out2_out));
  FDRE \prediction_reg[0] 
       (.C(clk),
        .CE(\start[1] ),
        .D(\prediction[0]_i_1__6_n_0 ),
        .Q(\prediction_reg_n_0_[0] ),
        .R(\prediction_reg[1]_1 ));
  FDRE \prediction_reg[1] 
       (.C(clk),
        .CE(\start[1] ),
        .D(tree_out__1),
        .Q(\prediction_reg_n_0_[1] ),
        .R(\prediction_reg[1]_1 ));
  MUXF7 \prediction_reg[1]_i_4 
       (.I0(\prediction[1]_i_8_n_0 ),
        .I1(tree_out2_out),
        .O(\prediction_reg[1]_i_4_n_0 ),
        .S(\prediction[1]_i_7__2_n_0 ));
  LUT2 #(
    .INIT(4'h2)) 
    \result[1]_i_5 
       (.I0(\prediction_reg_n_0_[1] ),
        .I1(\prediction_reg_n_0_[0] ),
        .O(\prediction_reg[1]_0 ));
endmodule

(* ORIG_REF_NAME = "random_forest_elephant" *) 
module design_1_random_forest_elepha_0_0_random_forest_elephant
   (done,
    result,
    clk,
    kde_prob_night_mean,
    dist_to_centroid_mean,
    accelerate,
    mean_speed,
    turning_angle_max,
    kde_prob_mean,
    step_median,
    turning_angle_median,
    is_night,
    start);
  output done;
  output [1:0]result;
  input clk;
  input [15:0]kde_prob_night_mean;
  input [15:0]dist_to_centroid_mean;
  input [15:0]accelerate;
  input [15:0]mean_speed;
  input [15:0]turning_angle_max;
  input [15:0]kde_prob_mean;
  input [15:0]step_median;
  input [15:0]turning_angle_median;
  input [15:0]is_night;
  input [1:0]start;

  wire [15:0]accelerate;
  wire clk;
  wire [15:0]dist_to_centroid_mean;
  wire done;
  wire [15:0]is_night;
  wire [15:0]kde_prob_mean;
  wire [15:0]kde_prob_night_mean;
  wire [15:0]mean_speed;
  wire p_3_in;
  wire [1:0]result;
  wire [1:0]start;
  wire [15:0]step_median;
  wire t1_n_1;
  wire t1_n_2;
  wire t1_n_3;
  wire t1_n_4;
  wire t1_n_5;
  wire t1_n_6;
  wire t1_n_7;
  wire t2_n_1;
  wire t2_n_2;
  wire t2_n_3;
  wire t2_n_4;
  wire t2_n_5;
  wire t2_n_6;
  wire t2_n_7;
  wire t2_n_8;
  wire t2_n_9;
  wire t3_n_0;
  wire t3_n_1;
  wire t3_n_2;
  wire t3_n_3;
  wire t3_n_4;
  wire t3_n_5;
  wire t3_n_6;
  wire t3_n_7;
  wire t3_n_8;
  wire t3_n_9;
  wire t4_n_1;
  wire t4_n_10;
  wire t4_n_11;
  wire t4_n_12;
  wire t4_n_13;
  wire t4_n_14;
  wire t4_n_15;
  wire t4_n_2;
  wire t4_n_3;
  wire t4_n_4;
  wire t4_n_5;
  wire t4_n_6;
  wire t4_n_7;
  wire t4_n_8;
  wire t4_n_9;
  wire t5_n_1;
  wire t5_n_2;
  wire t5_n_3;
  wire t5_n_4;
  wire t5_n_5;
  wire t5_n_6;
  wire t6_n_0;
  wire t6_n_1;
  wire t6_n_2;
  wire t6_n_3;
  wire t6_n_4;
  wire t6_n_5;
  wire t6_n_6;
  wire t6_n_7;
  wire t6_n_8;
  wire t6_n_9;
  wire t7_n_1;
  wire t7_n_2;
  wire t7_n_3;
  wire t7_n_4;
  wire t7_n_5;
  wire t7_n_6;
  wire t7_n_7;
  wire t7_n_8;
  wire t8_n_1;
  wire t8_n_2;
  wire t8_n_3;
  wire t8_n_4;
  wire t8_n_5;
  wire t8_n_6;
  wire t8_n_7;
  wire t8_n_8;
  wire [7:0]t_done;
  wire [15:0]turning_angle_max;
  wire [15:0]turning_angle_median;

  FDRE done_reg
       (.C(clk),
        .CE(1'b1),
        .D(t3_n_9),
        .Q(done),
        .R(1'b0));
  FDRE \result_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(t3_n_8),
        .Q(result[0]),
        .R(1'b0));
  FDRE \result_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(t3_n_7),
        .Q(result[1]),
        .R(1'b0));
  design_1_random_forest_elepha_0_0_decision_tree_1 t1
       (.accelerate(accelerate),
        .accelerate_13_sp_1(t1_n_2),
        .accelerate_14_sp_1(t1_n_1),
        .clk(clk),
        .kde_prob_mean(kde_prob_mean),
        .kde_prob_night_mean(kde_prob_night_mean[15:2]),
        .mean_speed(mean_speed),
        .mean_speed_10_sp_1(t1_n_3),
        .mean_speed_5_sp_1(t1_n_4),
        .mean_speed_9_sp_1(t1_n_5),
        .\prediction[1]_i_10__1_0 (t4_n_3),
        .\prediction[1]_i_10__1_1 (t6_n_2),
        .\prediction[1]_i_12__1_0 (t4_n_7),
        .\prediction[1]_i_13__3_0 (t8_n_5),
        .\prediction[1]_i_13__3_1 (t5_n_4),
        .\prediction[1]_i_13__3_2 (t2_n_4),
        .\prediction[1]_i_17__0_0 (t4_n_10),
        .\prediction_reg[0]_0 (t1_n_7),
        .\prediction_reg[0]_1 (t8_n_1),
        .\prediction_reg[0]_2 (t2_n_2),
        .\prediction_reg[0]_3 (t3_n_2),
        .\prediction_reg[0]_4 (t3_n_0),
        .\prediction_reg[1]_0 (t1_n_6),
        .\prediction_reg[1]_1 (t4_n_1),
        .\prediction_reg[1]_2 (t3_n_3),
        .\prediction_reg[1]_3 (t8_n_7),
        .\prediction_reg[1]_i_6 (t6_n_4),
        .\prediction_reg[1]_i_7_0 (t6_n_1),
        .start(start[1]),
        .step_median(step_median[13:2]),
        .t_done(t_done[0]),
        .turning_angle_max({turning_angle_max[15:12],turning_angle_max[9:0]}),
        .turning_angle_median(turning_angle_median[15:1]));
  design_1_random_forest_elepha_0_0_decision_tree_2 t2
       (.accelerate(accelerate),
        .accelerate_7_sp_1(t2_n_4),
        .clk(clk),
        .dist_to_centroid_mean(dist_to_centroid_mean),
        .dist_to_centroid_mean_12_sp_1(t2_n_6),
        .kde_prob_mean(kde_prob_mean),
        .\kde_prob_mean[7]_0 (t2_n_2),
        .kde_prob_mean_3_sp_1(t2_n_3),
        .kde_prob_mean_7_sp_1(t2_n_1),
        .kde_prob_night_mean(kde_prob_night_mean),
        .kde_prob_night_mean_2_sp_1(t2_n_7),
        .kde_prob_night_mean_8_sp_1(t2_n_5),
        .mean_speed(mean_speed),
        .\prediction[1]_i_14__0_0 (t5_n_2),
        .\prediction[1]_i_23__0_0 (t8_n_3),
        .\prediction[1]_i_23__0_1 (t4_n_3),
        .\prediction[1]_i_24__4_0 (t4_n_12),
        .\prediction[1]_i_24__4_1 (t7_n_7),
        .\prediction[1]_i_25__0_0 (t4_n_10),
        .\prediction[1]_i_5__4_0 (t7_n_8),
        .\prediction[1]_i_9__3_0 (t1_n_3),
        .\prediction_reg[0]_0 (t2_n_9),
        .\prediction_reg[0]_1 (t4_n_2),
        .\prediction_reg[1]_0 (t2_n_8),
        .\prediction_reg[1]_1 (t4_n_1),
        .\prediction_reg[1]_2 (t3_n_0),
        .\prediction_reg[1]_3 (t8_n_1),
        .\prediction_reg[1]_4 (t3_n_2),
        .\prediction_reg[1]_5 (t1_n_1),
        .\prediction_reg[1]_6 (t4_n_11),
        .\prediction_reg[1]_7 (t8_n_7),
        .\prediction_reg[1]_i_4__0_0 (t4_n_8),
        .start(start[1]),
        .step_median({step_median[13:12],step_median[7:0]}),
        .t_done(t_done[1]));
  design_1_random_forest_elepha_0_0_decision_tree_3 t3
       (.D({t3_n_7,t3_n_8}),
        .accelerate(accelerate[11:0]),
        .clk(clk),
        .dist_to_centroid_mean(dist_to_centroid_mean),
        .done_reg_0(t3_n_9),
        .done_reg_1({t_done[3],t_done[1:0]}),
        .done_reg_2(t6_n_9),
        .kde_prob_mean(kde_prob_mean),
        .kde_prob_mean_0_sp_1(t3_n_4),
        .kde_prob_mean_13_sp_1(t3_n_0),
        .kde_prob_mean_5_sp_1(t3_n_3),
        .kde_prob_mean_9_sp_1(t3_n_2),
        .kde_prob_night_mean({kde_prob_night_mean[15:14],kde_prob_night_mean[10:4]}),
        .mean_speed(mean_speed),
        .p_3_in(p_3_in),
        .\prediction[0]_i_2__1_0 (t8_n_4),
        .\prediction[1]_i_12_0 (t4_n_4),
        .\prediction[1]_i_4__0_0 (t4_n_11),
        .\prediction[1]_i_4__0_1 (t2_n_2),
        .\prediction[1]_i_7__0_0 (t2_n_6),
        .\prediction_reg[0]_0 (t4_n_2),
        .\prediction_reg[0]_1 (t1_n_2),
        .\prediction_reg[0]_2 (t7_n_2),
        .\prediction_reg[0]_3 (t6_n_0),
        .\prediction_reg[0]_4 (t8_n_2),
        .\prediction_reg[0]_5 (t4_n_10),
        .\prediction_reg[1]_0 (t4_n_1),
        .\prediction_reg[1]_1 (t8_n_1),
        .\prediction_reg[1]_2 (t8_n_7),
        .\result[1]_i_2_0 (t1_n_7),
        .\result[1]_i_2_1 (t1_n_6),
        .\result[1]_i_2_2 (t2_n_9),
        .\result[1]_i_2_3 (t2_n_8),
        .\result_reg[0] (t4_n_13),
        .\result_reg[0]_0 (t8_n_8),
        .\result_reg[0]_1 (t6_n_6),
        .start(start[1]),
        .turning_angle_max(turning_angle_max[12:2]),
        .turning_angle_median(turning_angle_median[15:1]),
        .turning_angle_median_10_sp_1(t3_n_5),
        .turning_angle_median_13_sp_1(t3_n_1),
        .turning_angle_median_6_sp_1(t3_n_6));
  design_1_random_forest_elepha_0_0_decision_tree_4 t4
       (.accelerate(accelerate[15:6]),
        .\accelerate[11] (t4_n_8),
        .clk(clk),
        .done_reg_0(t_done[3]),
        .kde_prob_mean(kde_prob_mean),
        .kde_prob_mean_0_sp_1(t4_n_4),
        .kde_prob_mean_6_sp_1(t4_n_2),
        .kde_prob_night_mean(kde_prob_night_mean),
        .kde_prob_night_mean_11_sp_1(t4_n_11),
        .kde_prob_night_mean_2_sp_1(t4_n_12),
        .mean_speed({mean_speed[15:14],mean_speed[11:0]}),
        .mean_speed_0_sp_1(t4_n_10),
        .\prediction[1]_i_14_0 (t5_n_2),
        .\prediction[1]_i_15__1_0 (t7_n_3),
        .\prediction[1]_i_15__1_1 (t7_n_4),
        .\prediction[1]_i_15__1_2 (t5_n_1),
        .\prediction[1]_i_24_0 (t3_n_4),
        .\prediction[1]_i_4__3_0 (t2_n_7),
        .\prediction_reg[0]_0 (t4_n_13),
        .\prediction_reg[0]_1 (t4_n_14),
        .\prediction_reg[0]_2 (t8_n_1),
        .\prediction_reg[0]_3 (t2_n_2),
        .\prediction_reg[0]_4 (t2_n_3),
        .\prediction_reg[0]_5 (t3_n_2),
        .\prediction_reg[0]_6 (t3_n_0),
        .\prediction_reg[0]_7 (t8_n_4),
        .\prediction_reg[1]_0 (t4_n_15),
        .\prediction_reg[1]_1 (t7_n_1),
        .\prediction_reg[1]_2 (t2_n_5),
        .\prediction_reg[1]_3 (t8_n_7),
        .\result[1]_i_2 (t5_n_6),
        .\result[1]_i_2_0 (t5_n_5),
        .\result[1]_i_2_1 (t6_n_7),
        .\result[1]_i_2_2 (t6_n_8),
        .start(start),
        .start_0_sp_1(t4_n_1),
        .step_median(step_median),
        .step_median_14_sp_1(t4_n_3),
        .step_median_1_sp_1(t4_n_5),
        .step_median_6_sp_1(t4_n_6),
        .turning_angle_max(turning_angle_max),
        .turning_angle_max_10_sp_1(t4_n_7),
        .turning_angle_median(turning_angle_median),
        .turning_angle_median_11_sp_1(t4_n_9));
  design_1_random_forest_elepha_0_0_decision_tree_5 t5
       (.accelerate(accelerate),
        .accelerate_1_sp_1(t5_n_4),
        .accelerate_5_sp_1(t5_n_1),
        .clk(clk),
        .dist_to_centroid_mean(dist_to_centroid_mean),
        .kde_prob_mean(kde_prob_mean[4:0]),
        .kde_prob_night_mean(kde_prob_night_mean),
        .kde_prob_night_mean_8_sp_1(t5_n_3),
        .mean_speed(mean_speed),
        .mean_speed_12_sp_1(t5_n_2),
        .\prediction[1]_i_20__1_0 (t4_n_8),
        .\prediction[1]_i_2_0 (t3_n_3),
        .\prediction[1]_i_2_1 (t6_n_5),
        .\prediction[1]_i_2_2 (t2_n_2),
        .\prediction[1]_i_4_0 (t2_n_1),
        .\prediction_reg[0]_0 (t5_n_6),
        .\prediction_reg[0]_1 (t6_n_1),
        .\prediction_reg[1]_0 (t5_n_5),
        .\prediction_reg[1]_1 (t4_n_1),
        .\prediction_reg[1]_2 (t8_n_1),
        .\prediction_reg[1]_3 (t3_n_2),
        .\prediction_reg[1]_4 (t3_n_0),
        .\prediction_reg[1]_5 (t4_n_11),
        .\prediction_reg[1]_6 (t8_n_7),
        .start(start[1]),
        .step_median(step_median),
        .t_done(t_done[4]));
  design_1_random_forest_elepha_0_0_decision_tree_6 t6
       (.accelerate(accelerate),
        .accelerate_14_sp_1(t6_n_0),
        .accelerate_8_sp_1(t6_n_2),
        .accelerate_9_sp_1(t6_n_4),
        .clk(clk),
        .dist_to_centroid_mean({dist_to_centroid_mean[15],dist_to_centroid_mean[8:3]}),
        .done_reg_0(t6_n_9),
        .done_reg_1({t_done[7:6],t_done[4]}),
        .is_night(is_night),
        .is_night_0_sp_1(t6_n_1),
        .kde_prob_mean(kde_prob_mean[10:0]),
        .kde_prob_mean_3_sp_1(t6_n_5),
        .kde_prob_night_mean(kde_prob_night_mean),
        .mean_speed(mean_speed),
        .mean_speed_2_sp_1(t6_n_3),
        .\prediction[0]_i_14_0 (t8_n_1),
        .\prediction[0]_i_14_1 (t7_n_1),
        .\prediction[0]_i_30__0_0 (t7_n_6),
        .\prediction[0]_i_3__1_0 (t5_n_4),
        .\prediction[0]_i_3__1_1 (t2_n_4),
        .\prediction_reg[0]_0 (t6_n_6),
        .\prediction_reg[0]_1 (t6_n_7),
        .\prediction_reg[0]_i_4_0 (t3_n_0),
        .\prediction_reg[0]_i_4_1 (t7_n_5),
        .\prediction_reg[1]_0 (t6_n_8),
        .\prediction_reg[1]_1 (t4_n_1),
        .\prediction_reg[1]_2 (t3_n_1),
        .\prediction_reg[1]_3 (t8_n_7),
        .\result[1]_i_2 (t4_n_15),
        .\result[1]_i_2_0 (t4_n_14),
        .\result[1]_i_2_1 (t5_n_5),
        .\result[1]_i_2_2 (t5_n_6),
        .start(start[1]),
        .turning_angle_max(turning_angle_max[15:3]),
        .turning_angle_median(turning_angle_median));
  design_1_random_forest_elepha_0_0_decision_tree_7 t7
       (.accelerate(accelerate),
        .accelerate_10_sp_1(t7_n_4),
        .accelerate_3_sp_1(t7_n_3),
        .clk(clk),
        .dist_to_centroid_mean(dist_to_centroid_mean),
        .dist_to_centroid_mean_10_sp_1(t7_n_5),
        .done_reg_0(t_done[6]),
        .kde_prob_mean(kde_prob_mean),
        .kde_prob_night_mean(kde_prob_night_mean),
        .kde_prob_night_mean_4_sp_1(t7_n_8),
        .kde_prob_night_mean_9_sp_1(t7_n_7),
        .mean_speed(mean_speed),
        .p_3_in(p_3_in),
        .\prediction[0]_i_16__0_0 (t1_n_5),
        .\prediction[0]_i_19__2_0 (t3_n_6),
        .\prediction[0]_i_3__0_0 (t4_n_9),
        .\prediction[0]_i_3__0_1 (t8_n_6),
        .\prediction[0]_i_3__0_2 (t5_n_3),
        .\prediction[0]_i_4__0_0 (t4_n_6),
        .\prediction[0]_i_6_0 (t4_n_11),
        .\prediction[0]_i_6_1 (t3_n_5),
        .\prediction[0]_i_7_0 (t2_n_4),
        .\prediction[0]_i_7_1 (t4_n_8),
        .\prediction_reg[0]_0 (t4_n_1),
        .\prediction_reg[0]_1 (t2_n_1),
        .\prediction_reg[0]_2 (t8_n_7),
        .start(start),
        .step_median(step_median[15:2]),
        .\step_median[14] (t7_n_2),
        .turning_angle_median({turning_angle_median[15:12],turning_angle_median[10:0]}),
        .turning_angle_median_0_sp_1(t7_n_6),
        .turning_angle_median_13_sp_1(t7_n_1));
  design_1_random_forest_elepha_0_0_decision_tree_8 t8
       (.accelerate(accelerate),
        .accelerate_5_sp_1(t8_n_5),
        .clk(clk),
        .dist_to_centroid_mean(dist_to_centroid_mean[15:1]),
        .done_reg_0(t_done[7]),
        .kde_prob_mean({kde_prob_mean[15:11],kde_prob_mean[8:2]}),
        .\kde_prob_mean[14] (t8_n_2),
        .kde_prob_mean_11_sp_1(t8_n_1),
        .mean_speed({mean_speed[15:8],mean_speed[3]}),
        .\prediction[0]_i_2__1 (t3_n_4),
        .\prediction[0]_i_9_0 (t3_n_3),
        .\prediction[1]_i_17_0 (t1_n_4),
        .\prediction[1]_i_17_1 (t6_n_3),
        .\prediction[1]_i_2__2_0 (t3_n_5),
        .\prediction[1]_i_7__2_0 (t4_n_5),
        .\prediction_reg[0]_0 (t3_n_0),
        .\prediction_reg[0]_1 (t3_n_2),
        .\prediction_reg[1]_0 (t8_n_8),
        .\prediction_reg[1]_1 (t4_n_1),
        .\prediction_reg[1]_2 (t2_n_2),
        .\prediction_reg[1]_3 (t3_n_1),
        .\prediction_reg[1]_4 (t6_n_1),
        .\prediction_reg[1]_i_4_0 (t4_n_3),
        .start(start[1]),
        .\start[1] (t8_n_7),
        .step_median(step_median[13:2]),
        .step_median_9_sp_1(t8_n_3),
        .turning_angle_max(turning_angle_max),
        .turning_angle_max_13_sp_1(t8_n_4),
        .turning_angle_median({turning_angle_median[12],turning_angle_median[9:0]}),
        .turning_angle_median_8_sp_1(t8_n_6));
endmodule
`ifndef GLBL
`define GLBL
`timescale  1 ps / 1 ps

module glbl ();

    parameter ROC_WIDTH = 100000;
    parameter TOC_WIDTH = 0;
    parameter GRES_WIDTH = 10000;
    parameter GRES_START = 10000;

//--------   STARTUP Globals --------------
    wire GSR;
    wire GTS;
    wire GWE;
    wire PRLD;
    wire GRESTORE;
    tri1 p_up_tmp;
    tri (weak1, strong0) PLL_LOCKG = p_up_tmp;

    wire PROGB_GLBL;
    wire CCLKO_GLBL;
    wire FCSBO_GLBL;
    wire [3:0] DO_GLBL;
    wire [3:0] DI_GLBL;
   
    reg GSR_int;
    reg GTS_int;
    reg PRLD_int;
    reg GRESTORE_int;

//--------   JTAG Globals --------------
    wire JTAG_TDO_GLBL;
    wire JTAG_TCK_GLBL;
    wire JTAG_TDI_GLBL;
    wire JTAG_TMS_GLBL;
    wire JTAG_TRST_GLBL;

    reg JTAG_CAPTURE_GLBL;
    reg JTAG_RESET_GLBL;
    reg JTAG_SHIFT_GLBL;
    reg JTAG_UPDATE_GLBL;
    reg JTAG_RUNTEST_GLBL;

    reg JTAG_SEL1_GLBL = 0;
    reg JTAG_SEL2_GLBL = 0 ;
    reg JTAG_SEL3_GLBL = 0;
    reg JTAG_SEL4_GLBL = 0;

    reg JTAG_USER_TDO1_GLBL = 1'bz;
    reg JTAG_USER_TDO2_GLBL = 1'bz;
    reg JTAG_USER_TDO3_GLBL = 1'bz;
    reg JTAG_USER_TDO4_GLBL = 1'bz;

    assign (strong1, weak0) GSR = GSR_int;
    assign (strong1, weak0) GTS = GTS_int;
    assign (weak1, weak0) PRLD = PRLD_int;
    assign (strong1, weak0) GRESTORE = GRESTORE_int;

    initial begin
	GSR_int = 1'b1;
	PRLD_int = 1'b1;
	#(ROC_WIDTH)
	GSR_int = 1'b0;
	PRLD_int = 1'b0;
    end

    initial begin
	GTS_int = 1'b1;
	#(TOC_WIDTH)
	GTS_int = 1'b0;
    end

    initial begin 
	GRESTORE_int = 1'b0;
	#(GRES_START);
	GRESTORE_int = 1'b1;
	#(GRES_WIDTH);
	GRESTORE_int = 1'b0;
    end

endmodule
`endif

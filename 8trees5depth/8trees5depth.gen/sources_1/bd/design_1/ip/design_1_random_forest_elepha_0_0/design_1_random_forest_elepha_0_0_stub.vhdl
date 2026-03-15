-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2024.1 (win64) Build 5076996 Wed May 22 18:37:14 MDT 2024
-- Date        : Fri Mar 13 12:33:03 2026
-- Host        : DESKTOP-AUH71TB running 64-bit major release  (build 9200)
-- Command     : write_vhdl -force -mode synth_stub
--               d:/8trees5depth/8trees5depth.gen/sources_1/bd/design_1/ip/design_1_random_forest_elepha_0_0/design_1_random_forest_elepha_0_0_stub.vhdl
-- Design      : design_1_random_forest_elepha_0_0
-- Purpose     : Stub declaration of top-level module interface
-- Device      : xc7z010clg400-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity design_1_random_forest_elepha_0_0 is
  Port ( 
    clk : in STD_LOGIC;
    start : in STD_LOGIC_VECTOR ( 1 downto 0 );
    kde_prob_mean : in STD_LOGIC_VECTOR ( 15 downto 0 );
    kde_prob_night_mean : in STD_LOGIC_VECTOR ( 15 downto 0 );
    dist_to_centroid_mean : in STD_LOGIC_VECTOR ( 15 downto 0 );
    step_median : in STD_LOGIC_VECTOR ( 15 downto 0 );
    mean_speed : in STD_LOGIC_VECTOR ( 15 downto 0 );
    accelerate : in STD_LOGIC_VECTOR ( 15 downto 0 );
    turning_angle_max : in STD_LOGIC_VECTOR ( 15 downto 0 );
    turning_angle_median : in STD_LOGIC_VECTOR ( 15 downto 0 );
    is_night : in STD_LOGIC_VECTOR ( 15 downto 0 );
    done : out STD_LOGIC;
    result : out STD_LOGIC_VECTOR ( 1 downto 0 )
  );

end design_1_random_forest_elepha_0_0;

architecture stub of design_1_random_forest_elepha_0_0 is
attribute syn_black_box : boolean;
attribute black_box_pad_pin : string;
attribute syn_black_box of stub : architecture is true;
attribute black_box_pad_pin of stub : architecture is "clk,start[1:0],kde_prob_mean[15:0],kde_prob_night_mean[15:0],dist_to_centroid_mean[15:0],step_median[15:0],mean_speed[15:0],accelerate[15:0],turning_angle_max[15:0],turning_angle_median[15:0],is_night[15:0],done,result[1:0]";
attribute X_CORE_INFO : string;
attribute X_CORE_INFO of stub : architecture is "random_forest_elephant,Vivado 2024.1";
begin
end;

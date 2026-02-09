module decision_tree_8 (
    input wire [31:0] kde_very_low_prob_count, kde_low_prob_ratio, dist_to_centroid_mean, step_max, mean_speed, turning_angle_mean, turning_angle_median, hour,
    output reg tree_out
);

always @(*) begin
    if (step_max <= 32'h316E6020) begin
        if (kde_low_prob_ratio <= 32'h80000000) begin
            if (step_max <= 32'h05BAC394) begin
                if (step_max <= 32'h05A45594) begin
                    if (mean_speed <= 32'h09751058) begin
                        if (turning_angle_mean <= 32'hEB739400) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (dist_to_centroid_mean <= 32'h31E82B20) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (dist_to_centroid_mean <= 32'h12C740D0) begin
                        tree_out = 1'b1;
                    end else begin
                        if (turning_angle_mean <= 32'hECDDF900) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end else begin
                if (turning_angle_mean <= 32'hEB81CE80) begin
                    if (mean_speed <= 32'h4573FDC0) begin
                        if (dist_to_centroid_mean <= 32'h7C47FA00) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (dist_to_centroid_mean <= 32'h5D80DF80) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    tree_out = 1'b1;
                end
            end
        end else begin
            if (turning_angle_mean <= 32'hE999B980) begin
                if (dist_to_centroid_mean <= 32'h84E89080) begin
                    if (step_max <= 32'h0C7A3B98) begin
                        if (mean_speed <= 32'h13128680) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (kde_very_low_prob_count <= 32'h80000000) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (dist_to_centroid_mean <= 32'h8BB81480) begin
                        if (step_max <= 32'h0FFB65B8) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (turning_angle_mean <= 32'h2D0ECE60) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end else begin
                if (turning_angle_median <= 32'hEBAB9280) begin
                    if (turning_angle_median <= 32'hEA7A7780) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    tree_out = 1'b1;
                end
            end
        end
    end else begin
        if (dist_to_centroid_mean <= 32'h05326858) begin
            if (mean_speed <= 32'h460C10C0) begin
                tree_out = 1'b0;
            end else begin
                tree_out = 1'b1;
            end
        end else begin
            if (mean_speed <= 32'h584D8400) begin
                tree_out = 1'b0;
            end else begin
                if (dist_to_centroid_mean <= 32'h2A4B2B40) begin
                    if (turning_angle_median <= 32'h22206A40) begin
                        if (turning_angle_mean <= 32'h1FF4C510) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (step_max <= 32'h339CB280) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (turning_angle_median <= 32'h01694E2C) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end
        end
    end
end
endmodule

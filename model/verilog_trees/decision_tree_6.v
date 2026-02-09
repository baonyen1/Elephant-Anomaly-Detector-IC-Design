module decision_tree_6 (
    input wire [31:0] kde_very_low_prob_count, kde_low_prob_ratio, dist_to_centroid_mean, step_max, mean_speed, turning_angle_mean, turning_angle_median, hour,
    output reg tree_out
);

always @(*) begin
    if (kde_low_prob_ratio <= 32'h80000000) begin
        if (dist_to_centroid_mean <= 32'h47C68EC0) begin
            if (turning_angle_mean <= 32'hEB84D700) begin
                if (mean_speed <= 32'h58BEC780) begin
                    if (step_max <= 32'h271CEC40) begin
                        if (mean_speed <= 32'h24497500) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (turning_angle_median <= 32'h3B4C07E0) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    tree_out = 1'b1;
                end
            end else begin
                tree_out = 1'b1;
            end
        end else begin
            if (mean_speed <= 32'h59034DC0) begin
                if (turning_angle_median <= 32'hEB76FA80) begin
                    if (turning_angle_mean <= 32'h7D136AC0) begin
                        if (mean_speed <= 32'h000A1130) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    if (step_max <= 32'h000D431B) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end else begin
                tree_out = 1'b1;
            end
        end
    end else begin
        if (turning_angle_median <= 32'hE9D94480) begin
            if (hour <= 32'hDD174600) begin
                if (dist_to_centroid_mean <= 32'h86D22480) begin
                    if (kde_very_low_prob_count <= 32'h80000000) begin
                        if (turning_angle_median <= 32'h4D09C400) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (hour <= 32'hAE8BA300) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (turning_angle_median <= 32'h4B6117C0) begin
                        if (turning_angle_median <= 32'h4818DEC0) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end else begin
                if (turning_angle_mean <= 32'h74FAC540) begin
                    if (dist_to_centroid_mean <= 32'h3EC56200) begin
                        tree_out = 1'b0;
                    end else begin
                        if (dist_to_centroid_mean <= 32'h8C075B80) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (kde_very_low_prob_count <= 32'h80000000) begin
                        tree_out = 1'b0;
                    end else begin
                        if (step_max <= 32'h18F3D250) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end
        end else begin
            tree_out = 1'b1;
        end
    end
end
endmodule

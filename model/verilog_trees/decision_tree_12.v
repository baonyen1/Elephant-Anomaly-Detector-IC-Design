module decision_tree_12 (
    input wire [31:0] kde_very_low_prob_count, kde_low_prob_ratio, dist_to_centroid_mean, step_max, mean_speed, turning_angle_mean, turning_angle_median, hour,
    output reg tree_out
);

always @(*) begin
    if (turning_angle_mean <= 32'hEB4AAD80) begin
        if (kde_very_low_prob_count <= 32'h80000000) begin
            if (step_max <= 32'h3186B240) begin
                if (step_max <= 32'h0006A312) begin
                    if (turning_angle_mean <= 32'h5EABD3C0) begin
                        tree_out = 1'b0;
                    end else begin
                        if (mean_speed <= 32'h00099DC3) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (kde_low_prob_ratio <= 32'h80000000) begin
                        if (mean_speed <= 32'h47BC8040) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (dist_to_centroid_mean <= 32'h546D0E80) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end else begin
                if (turning_angle_mean <= 32'hD86EE900) begin
                    if (step_max <= 32'h8BD5A280) begin
                        if (step_max <= 32'h31B3FA40) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    if (mean_speed <= 32'h4F628A40) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end
        end else begin
            if (mean_speed <= 32'h00764E3C) begin
                if (turning_angle_median <= 32'h8AB0EF80) begin
                    tree_out = 1'b0;
                end else begin
                    tree_out = 1'b1;
                end
            end else begin
                if (mean_speed <= 32'h195F4A90) begin
                    if (mean_speed <= 32'h04F85190) begin
                        if (turning_angle_median <= 32'hB3E84580) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (mean_speed <= 32'h0E5CEC60) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (hour <= 32'hF45D1780) begin
                        if (mean_speed <= 32'h1A3ED3D0) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (turning_angle_median <= 32'h7DE16080) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end
        end
    end else begin
        if (turning_angle_mean <= 32'hEB84D700) begin
            if (kde_very_low_prob_count <= 32'h80000000) begin
                tree_out = 1'b0;
            end else begin
                tree_out = 1'b1;
            end
        end else begin
            tree_out = 1'b1;
        end
    end
end
endmodule

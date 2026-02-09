module decision_tree_1 (
    input wire [31:0] kde_very_low_prob_count, kde_low_prob_ratio, dist_to_centroid_mean, step_max, mean_speed, turning_angle_mean, turning_angle_median, hour,
    output reg tree_out
);

always @(*) begin
    if (turning_angle_mean <= 32'hEB7F5700) begin
        if (mean_speed <= 32'h584D8400) begin
            if (kde_low_prob_ratio <= 32'h80000000) begin
                if (hour <= 32'h80000040) begin
                    if (dist_to_centroid_mean <= 32'h7C336F40) begin
                        if (dist_to_centroid_mean <= 32'h69E03DC0) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (turning_angle_median <= 32'h0FEE8FB8) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (step_max <= 32'h0006A30A) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end else begin
                if (kde_very_low_prob_count <= 32'h80000000) begin
                    if (mean_speed <= 32'h3057A8C0) begin
                        if (dist_to_centroid_mean <= 32'h547E6E40) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (hour <= 32'h45D17480) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (turning_angle_mean <= 32'h188F3EE0) begin
                        if (turning_angle_median <= 32'h13276830) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (mean_speed <= 32'h2D5E8A40) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end
        end else begin
            if (step_max <= 32'h315D1140) begin
                tree_out = 1'b0;
            end else begin
                if (step_max <= 32'h31DD75A0) begin
                    if (turning_angle_mean <= 32'hBC95FF00) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b1;
                    end
                end else begin
                    tree_out = 1'b1;
                end
            end
        end
    end else begin
        if (mean_speed <= 32'h000F3BE7) begin
            tree_out = 1'b1;
        end else begin
            tree_out = 1'b1;
        end
    end
end
endmodule

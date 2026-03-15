module decision_tree_10 (
    input wire [15:0] kde_prob_mean, kde_prob_night_mean, dist_to_centroid_mean, step_median, mean_speed, accelerate, turning_angle_max, turning_angle_median, is_night,
    output reg tree_out
);

always @(*) begin
    if (kde_prob_mean <= 16'h1988) begin
        tree_out = 1'b1;
    end else begin
        if (accelerate <= 16'h530A) begin
            if (turning_angle_median <= 16'hE0A4) begin
                if (step_median <= 16'h49EC) begin
                    if (turning_angle_max <= 16'h0003) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    if (kde_prob_mean <= 16'h3FF0) begin
                        tree_out = 1'b1;
                    end else begin
                        if (turning_angle_median <= 16'h1558) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end else begin
                if (step_median <= 16'h3982) begin
                    tree_out = 1'b0;
                end else begin
                    tree_out = 1'b1;
                end
            end
        end else begin
            if (step_median <= 16'h12C4) begin
                if (mean_speed <= 16'h17BE) begin
                    tree_out = 1'b0;
                end else begin
                    if (kde_prob_night_mean <= 16'h3CFA) begin
                        tree_out = 1'b1;
                    end else begin
                        if (is_night <= 16'h8000) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end else begin
                if (step_median <= 16'h3298) begin
                    if (dist_to_centroid_mean <= 16'h5105) begin
                        if (accelerate <= 16'h54C2) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (accelerate <= 16'h6185) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (kde_prob_night_mean <= 16'h7BDE) begin
                        if (mean_speed <= 16'h93F0) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (dist_to_centroid_mean <= 16'h3418) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end
        end
    end
end
endmodule

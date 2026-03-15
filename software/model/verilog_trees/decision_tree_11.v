module decision_tree_11 (
    input wire [15:0] kde_prob_mean, kde_prob_night_mean, dist_to_centroid_mean, step_median, mean_speed, accelerate, turning_angle_max, turning_angle_median, is_night,
    output reg tree_out
);

always @(*) begin
    if (kde_prob_mean <= 16'h198A) begin
        tree_out = 1'b1;
    end else begin
        if (kde_prob_mean <= 16'h49C5) begin
            if (accelerate <= 16'h4FB2) begin
                if (mean_speed <= 16'h57BA) begin
                    tree_out = 1'b0;
                end else begin
                    if (dist_to_centroid_mean <= 16'h2A2A) begin
                        if (turning_angle_max <= 16'h188B) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (dist_to_centroid_mean <= 16'h3672) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end else begin
                if (mean_speed <= 16'h3A8D) begin
                    if (kde_prob_night_mean <= 16'h2AE8) begin
                        tree_out = 1'b1;
                    end else begin
                        if (dist_to_centroid_mean <= 16'h07A5) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (kde_prob_night_mean <= 16'h34EB) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end
        end else begin
            if (step_median <= 16'h2E4C) begin
                tree_out = 1'b0;
            end else begin
                if (step_median <= 16'h75AE) begin
                    if (kde_prob_night_mean <= 16'hA464) begin
                        if (dist_to_centroid_mean <= 16'h10B0) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (kde_prob_mean <= 16'hA16A) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    tree_out = 1'b1;
                end
            end
        end
    end
end
endmodule

module decision_tree_12 (
    input wire [15:0] kde_prob_mean, kde_prob_night_mean, dist_to_centroid_mean, step_median, mean_speed, accelerate, turning_angle_max, turning_angle_median, is_night,
    output reg tree_out
);

always @(*) begin
    if (dist_to_centroid_mean <= 16'h74E8) begin
        if (kde_prob_mean <= 16'h19C8) begin
            if (accelerate <= 16'h0D22) begin
                if (turning_angle_median <= 16'h20F0) begin
                    if (kde_prob_mean <= 16'h1924) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    tree_out = 1'b1;
                end
            end else begin
                tree_out = 1'b1;
            end
        end else begin
            if (accelerate <= 16'h519A) begin
                if (mean_speed <= 16'h4FDD) begin
                    if (kde_prob_night_mean <= 16'h08B6) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    if (turning_angle_median <= 16'hDB16) begin
                        if (kde_prob_night_mean <= 16'h3312) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (kde_prob_mean <= 16'h4B9A) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end else begin
                if (mean_speed <= 16'h17BE) begin
                    tree_out = 1'b0;
                end else begin
                    if (turning_angle_median <= 16'h0D1F) begin
                        tree_out = 1'b0;
                    end else begin
                        if (mean_speed <= 16'h9322) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end
        end
    end else begin
        if (step_median <= 16'h004B) begin
            tree_out = 1'b0;
        end else begin
            if (kde_prob_mean <= 16'h198C) begin
                if (kde_prob_night_mean <= 16'h0006) begin
                    tree_out = 1'b1;
                end else begin
                    tree_out = 1'b1;
                end
            end else begin
                if (turning_angle_max <= 16'h1F10) begin
                    if (step_median <= 16'h37BE) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b1;
                    end
                end else begin
                    if (step_median <= 16'h010C) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end
        end
    end
end
endmodule

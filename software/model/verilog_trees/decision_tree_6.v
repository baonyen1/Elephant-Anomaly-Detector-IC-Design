module decision_tree_6 (
    input wire [15:0] kde_prob_mean, kde_prob_night_mean, dist_to_centroid_mean, step_median, mean_speed, accelerate, turning_angle_max, turning_angle_median, is_night,
    output reg tree_out
);

always @(*) begin
    if (kde_prob_night_mean <= 16'h41AE) begin
        if (kde_prob_night_mean <= 16'h353C) begin
            if (dist_to_centroid_mean <= 16'h82FD) begin
                if (step_median <= 16'h0748) begin
                    if (dist_to_centroid_mean <= 16'h5A7A) begin
                        if (dist_to_centroid_mean <= 16'h2B2C) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (kde_prob_mean <= 16'h19A1) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (is_night <= 16'h8000) begin
                        if (turning_angle_median <= 16'h8158) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (kde_prob_night_mean <= 16'h1796) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end else begin
                if (accelerate <= 16'h0003) begin
                    tree_out = 1'b1;
                end else begin
                    tree_out = 1'b1;
                end
            end
        end else begin
            if (mean_speed <= 16'h1B38) begin
                if (dist_to_centroid_mean <= 16'h59B8) begin
                    if (kde_prob_mean <= 16'h19A1) begin
                        tree_out = 1'b1;
                    end else begin
                        if (kde_prob_night_mean <= 16'h3648) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (kde_prob_mean <= 16'h19E8) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end else begin
                if (is_night <= 16'h8000) begin
                    tree_out = 1'b0;
                end else begin
                    if (turning_angle_median <= 16'hC602) begin
                        if (turning_angle_max <= 16'h00E8) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (turning_angle_max <= 16'hD62A) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end
        end
    end else begin
        if (mean_speed <= 16'h4F75) begin
            if (kde_prob_night_mean <= 16'h4773) begin
                if (turning_angle_median <= 16'h2BF7) begin
                    if (kde_prob_night_mean <= 16'h4262) begin
                        tree_out = 1'b0;
                    end else begin
                        if (kde_prob_mean <= 16'h19BE) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (accelerate <= 16'h4572) begin
                        if (mean_speed <= 16'h31E8) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end else begin
                if (kde_prob_night_mean <= 16'h492A) begin
                    if (turning_angle_median <= 16'h0F9A) begin
                        if (step_median <= 16'h13FD) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (kde_prob_mean <= 16'h1B98) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (mean_speed <= 16'h4DA0) begin
                        if (step_median <= 16'h0003) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (step_median <= 16'h2B1E) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end
        end else begin
            if (kde_prob_mean <= 16'h49C5) begin
                if (accelerate <= 16'h4F2E) begin
                    if (turning_angle_median <= 16'hC986) begin
                        if (accelerate <= 16'h0CAE) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        tree_out = 1'b1;
                    end
                end else begin
                    if (dist_to_centroid_mean <= 16'h048A) begin
                        if (dist_to_centroid_mean <= 16'h0193) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end else begin
                if (kde_prob_night_mean <= 16'hF3A0) begin
                    if (accelerate <= 16'h6456) begin
                        if (step_median <= 16'h2D02) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (accelerate <= 16'h66DC) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (kde_prob_night_mean <= 16'hF49A) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end
        end
    end
end
endmodule

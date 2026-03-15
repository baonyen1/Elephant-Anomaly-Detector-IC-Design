module decision_tree_9 (
    input wire [15:0] kde_prob_mean, kde_prob_night_mean, dist_to_centroid_mean, step_median, mean_speed, accelerate, turning_angle_max, turning_angle_median, is_night,
    output reg tree_out
);

always @(*) begin
    if (kde_prob_night_mean <= 16'h4498) begin
        if (kde_prob_mean <= 16'h19C8) begin
            if (step_median <= 16'h029D) begin
                if (step_median <= 16'h0294) begin
                    tree_out = 1'b1;
                end else begin
                    tree_out = 1'b0;
                end
            end else begin
                if (kde_prob_mean <= 16'h19A0) begin
                    tree_out = 1'b1;
                end else begin
                    if (dist_to_centroid_mean <= 16'h65DB) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end
        end else begin
            if (step_median <= 16'h3034) begin
                if (kde_prob_night_mean <= 16'h0FF6) begin
                    if (dist_to_centroid_mean <= 16'h5B40) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    if (step_median <= 16'h0D22) begin
                        if (is_night <= 16'h8000) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (step_median <= 16'h0D58) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end else begin
                if (mean_speed <= 16'h66E1) begin
                    if (accelerate <= 16'h09E4) begin
                        tree_out = 1'b0;
                    end else begin
                        if (dist_to_centroid_mean <= 16'h225E) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (turning_angle_median <= 16'h2BCB) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end
        end
    end else begin
        if (mean_speed <= 16'h59AC) begin
            if (kde_prob_mean <= 16'h193C) begin
                tree_out = 1'b1;
            end else begin
                if (accelerate <= 16'h51E2) begin
                    if (accelerate <= 16'h0000) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    if (turning_angle_median <= 16'hEB5B) begin
                        if (accelerate <= 16'h51F9) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (turning_angle_median <= 16'hF4A3) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end
        end else begin
            if (kde_prob_mean <= 16'h5804) begin
                if (is_night <= 16'h8000) begin
                    tree_out = 1'b0;
                end else begin
                    if (kde_prob_night_mean <= 16'h5162) begin
                        tree_out = 1'b0;
                    end else begin
                        if (step_median <= 16'h60F4) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end else begin
                if (kde_prob_mean <= 16'hCC16) begin
                    tree_out = 1'b0;
                end else begin
                    if (kde_prob_mean <= 16'hCF34) begin
                        tree_out = 1'b1;
                    end else begin
                        if (turning_angle_max <= 16'h0AB0) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end
        end
    end
end
endmodule

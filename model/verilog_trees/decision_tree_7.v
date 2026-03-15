module decision_tree_7 (
    input wire [15:0] kde_prob_mean, kde_prob_night_mean, dist_to_centroid_mean, step_median, mean_speed, accelerate, turning_angle_max, turning_angle_median, is_night,
    output reg tree_out
);

always @(*) begin
    if (kde_prob_night_mean <= 16'h3FE6) begin
        if (kde_prob_mean <= 16'h1999) begin
            tree_out = 1'b1;
        end else begin
            if (accelerate <= 16'h59A6) begin
                if (turning_angle_median <= 16'h1B30) begin
                    if (mean_speed <= 16'h55E9) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b1;
                    end
                end else begin
                    tree_out = 1'b0;
                end
            end else begin
                if (accelerate <= 16'h600F) begin
                    tree_out = 1'b1;
                end else begin
                    if (turning_angle_max <= 16'h071A) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end
        end
    end else begin
        if (step_median <= 16'h31EB) begin
            if (mean_speed <= 16'h4FFA) begin
                if (dist_to_centroid_mean <= 16'h7E72) begin
                    if (step_median <= 16'h10A8) begin
                        if (accelerate <= 16'h1530) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (kde_prob_night_mean <= 16'h4936) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (accelerate <= 16'h0EB6) begin
                        if (kde_prob_night_mean <= 16'h469A) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (accelerate <= 16'h1974) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end else begin
                if (turning_angle_median <= 16'hCEA8) begin
                    tree_out = 1'b0;
                end else begin
                    if (turning_angle_max <= 16'hF04E) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end
        end else begin
            if (kde_prob_night_mean <= 16'h7392) begin
                if (kde_prob_mean <= 16'h463A) begin
                    if (kde_prob_night_mean <= 16'h4148) begin
                        tree_out = 1'b0;
                    end else begin
                        if (step_median <= 16'h32C2) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (turning_angle_median <= 16'h182B) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end else begin
                if (turning_angle_max <= 16'hE743) begin
                    if (dist_to_centroid_mean <= 16'h60B1) begin
                        if (kde_prob_mean <= 16'hCDA2) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (kde_prob_night_mean <= 16'hA6F6) begin
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

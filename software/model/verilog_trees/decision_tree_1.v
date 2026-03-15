module decision_tree_1 (
    input wire [15:0] kde_prob_mean, kde_prob_night_mean, dist_to_centroid_mean, step_median, mean_speed, accelerate, turning_angle_max, turning_angle_median, is_night,
    output reg tree_out
);

always @(*) begin
    if (kde_prob_mean <= 16'h1998) begin
        tree_out = 1'b1;
    end else begin
        if (accelerate <= 16'h519A) begin
            if (mean_speed <= 16'h57BA) begin
                if (mean_speed <= 16'h4FB8) begin
                    if (kde_prob_night_mean <= 16'h084A) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    if (step_median <= 16'h2C8B) begin
                        tree_out = 1'b1;
                    end else begin
                        if (accelerate <= 16'h06E1) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end else begin
                if (accelerate <= 16'h0CAE) begin
                    if (dist_to_centroid_mean <= 16'h3458) begin
                        if (kde_prob_night_mean <= 16'h7326) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (dist_to_centroid_mean <= 16'h38F1) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    tree_out = 1'b0;
                end
            end
        end else begin
            if (mean_speed <= 16'h4D1A) begin
                if (kde_prob_mean <= 16'h1EA0) begin
                    if (accelerate <= 16'h6126) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    tree_out = 1'b0;
                end
            end else begin
                if (is_night <= 16'h8000) begin
                    tree_out = 1'b0;
                end else begin
                    if (turning_angle_max <= 16'hAFE8) begin
                        if (kde_prob_mean <= 16'h4B0E) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (kde_prob_night_mean <= 16'h884A) begin
                            tree_out = 1'b1;
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

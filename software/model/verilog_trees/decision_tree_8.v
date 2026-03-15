module decision_tree_8 (
    input wire [15:0] kde_prob_mean, kde_prob_night_mean, dist_to_centroid_mean, step_median, mean_speed, accelerate, turning_angle_max, turning_angle_median, is_night,
    output reg tree_out
);

always @(*) begin
    if (kde_prob_mean <= 16'h19D3) begin
        if (is_night <= 16'h8000) begin
            if (turning_angle_max <= 16'hECF0) begin
                tree_out = 1'b1;
            end else begin
                if (turning_angle_median <= 16'hEF66) begin
                    tree_out = 1'b0;
                end else begin
                    tree_out = 1'b1;
                end
            end
        end else begin
            tree_out = 1'b1;
        end
    end else begin
        if (step_median <= 16'h3092) begin
            if (accelerate <= 16'h569D) begin
                if (turning_angle_max <= 16'hECF4) begin
                    if (is_night <= 16'h8000) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    if (accelerate <= 16'h242A) begin
                        if (accelerate <= 16'h000A) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (turning_angle_median <= 16'hED4E) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end else begin
                if (turning_angle_max <= 16'h6137) begin
                    if (dist_to_centroid_mean <= 16'h5609) begin
                        if (mean_speed <= 16'h014A) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (step_median <= 16'h0C58) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (kde_prob_mean <= 16'h1CDE) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end
        end else begin
            if (kde_prob_mean <= 16'h5804) begin
                if (step_median <= 16'h4D8E) begin
                    if (dist_to_centroid_mean <= 16'h1EE2) begin
                        if (mean_speed <= 16'h7526) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (accelerate <= 16'h476A) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (accelerate <= 16'h42CC) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end else begin
                if (accelerate <= 16'hC468) begin
                    tree_out = 1'b0;
                end else begin
                    tree_out = 1'b1;
                end
            end
        end
    end
end
endmodule

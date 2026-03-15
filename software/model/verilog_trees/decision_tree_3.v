module decision_tree_3 (
    input wire [15:0] kde_prob_mean, kde_prob_night_mean, dist_to_centroid_mean, step_median, mean_speed, accelerate, turning_angle_max, turning_angle_median, is_night,
    output reg tree_out
);

always @(*) begin
    if (mean_speed <= 16'h140A) begin
        if (dist_to_centroid_mean <= 16'h7B10) begin
            if (kde_prob_mean <= 16'h199B) begin
                tree_out = 1'b1;
            end else begin
                if (is_night <= 16'h8000) begin
                    tree_out = 1'b0;
                end else begin
                    tree_out = 1'b0;
                end
            end
        end else begin
            if (kde_prob_night_mean <= 16'h456F) begin
                if (kde_prob_mean <= 16'h19BC) begin
                    tree_out = 1'b1;
                end else begin
                    tree_out = 1'b0;
                end
            end else begin
                tree_out = 1'b0;
            end
        end
    end else begin
        if (kde_prob_mean <= 16'h19C8) begin
            if (turning_angle_median <= 16'hED2D) begin
                if (kde_prob_mean <= 16'h1988) begin
                    tree_out = 1'b1;
                end else begin
                    if (dist_to_centroid_mean <= 16'h4CE8) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end else begin
                if (kde_prob_mean <= 16'h1604) begin
                    tree_out = 1'b1;
                end else begin
                    tree_out = 1'b0;
                end
            end
        end else begin
            if (accelerate <= 16'h51E2) begin
                if (step_median <= 16'h4257) begin
                    if (mean_speed <= 16'h57BA) begin
                        tree_out = 1'b0;
                    end else begin
                        if (step_median <= 16'h311A) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
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
                if (kde_prob_mean <= 16'h5804) begin
                    if (kde_prob_mean <= 16'h4CC6) begin
                        if (turning_angle_median <= 16'hE48F) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (step_median <= 16'h4695) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (turning_angle_median <= 16'hE27B) begin
                        if (mean_speed <= 16'hD2FC) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end
        end
    end
end
endmodule

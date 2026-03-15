module decision_tree_5 (
    input wire [15:0] kde_prob_mean, kde_prob_night_mean, dist_to_centroid_mean, step_median, mean_speed, accelerate, turning_angle_max, turning_angle_median, is_night,
    output reg tree_out
);

always @(*) begin
    if (is_night <= 16'h8000) begin
        if (step_median <= 16'h0524) begin
            if (kde_prob_mean <= 16'h199A) begin
                tree_out = 1'b1;
            end else begin
                tree_out = 1'b0;
            end
        end else begin
            if (kde_prob_mean <= 16'h1990) begin
                tree_out = 1'b1;
            end else begin
                tree_out = 1'b0;
            end
        end
    end else begin
        if (kde_prob_night_mean <= 16'h4773) begin
            if (dist_to_centroid_mean <= 16'h0E5E) begin
                tree_out = 1'b0;
            end else begin
                if (kde_prob_mean <= 16'h1999) begin
                    tree_out = 1'b1;
                end else begin
                    if (accelerate <= 16'h51BC) begin
                        if (dist_to_centroid_mean <= 16'h2C0E) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (kde_prob_mean <= 16'h1F94) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end
        end else begin
            if (mean_speed <= 16'h4FFA) begin
                if (kde_prob_mean <= 16'h19BD) begin
                    tree_out = 1'b1;
                end else begin
                    tree_out = 1'b0;
                end
            end else begin
                if (accelerate <= 16'h50B2) begin
                    if (turning_angle_max <= 16'hDC08) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b1;
                    end
                end else begin
                    if (kde_prob_mean <= 16'h4B82) begin
                        if (step_median <= 16'h3298) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (kde_prob_night_mean <= 16'hDC8E) begin
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

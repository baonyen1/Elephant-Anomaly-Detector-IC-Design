module decision_tree_2 (
    input wire [15:0] kde_prob_mean, kde_prob_night_mean, dist_to_centroid_mean, step_median, mean_speed, accelerate, turning_angle_max, turning_angle_median, is_night,
    output reg tree_out
);

always @(*) begin
    if (kde_prob_night_mean <= 16'h44CD) begin
        if (kde_prob_mean <= 16'h19C8) begin
            if (kde_prob_mean <= 16'h1999) begin
                tree_out = 1'b1;
            end else begin
                if (kde_prob_mean <= 16'h19BE) begin
                    if (kde_prob_mean <= 16'h19A8) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    tree_out = 1'b1;
                end
            end
        end else begin
            if (mean_speed <= 16'h57A8) begin
                if (kde_prob_night_mean <= 16'h44C8) begin
                    if (accelerate <= 16'h55AC) begin
                        tree_out = 1'b0;
                    end else begin
                        if (accelerate <= 16'h6126) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    tree_out = 1'b1;
                end
            end else begin
                if (accelerate <= 16'h1DC7) begin
                    tree_out = 1'b1;
                end else begin
                    tree_out = 1'b0;
                end
            end
        end
    end else begin
        if (kde_prob_night_mean <= 16'h6615) begin
            if (step_median <= 16'h32C2) begin
                if (kde_prob_mean <= 16'h199F) begin
                    tree_out = 1'b1;
                end else begin
                    tree_out = 1'b0;
                end
            end else begin
                if (mean_speed <= 16'h9356) begin
                    if (accelerate <= 16'h4F80) begin
                        if (mean_speed <= 16'h7230) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (accelerate <= 16'h58CA) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (step_median <= 16'h5504) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end
        end else begin
            if (step_median <= 16'h2AAA) begin
                tree_out = 1'b0;
            end else begin
                if (kde_prob_mean <= 16'h7AA6) begin
                    if (accelerate <= 16'h4F2E) begin
                        if (kde_prob_night_mean <= 16'h772B) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (turning_angle_median <= 16'hD02E) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (mean_speed <= 16'h5408) begin
                        if (mean_speed <= 16'h5343) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (kde_prob_mean <= 16'h7C3A) begin
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

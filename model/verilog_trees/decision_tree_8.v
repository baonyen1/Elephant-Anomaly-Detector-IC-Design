module decision_tree_8 (
    input wire [31:0] kde_low_prob_ratio, kde_prob_min, dist_to_centroid_mean, turning_angle_max, mean_speed, turning_entropy,
    output reg tree_out
);

always @(*) begin
    if (kde_low_prob_ratio <= 32'h80000000) begin
        tree_out = 1'b0;
    end else begin
        if (dist_to_centroid_mean <= 32'h33B5B040) begin
            tree_out = 1'b0;
        end else begin
            if (kde_prob_min <= 32'h2394F100) begin
                if (kde_prob_min <= 32'h1F9FB170) begin
                    if (dist_to_centroid_mean <= 32'h80BA3180) begin
                        if (turning_angle_max <= 32'hF513EA80) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (dist_to_centroid_mean <= 32'h825A4600) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (mean_speed <= 32'h34E29A60) begin
                        if (turning_entropy <= 32'h8801A7C0) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end else begin
                tree_out = 1'b0;
            end
        end
    end
end
endmodule

module decision_tree_11 (
    input wire [31:0] kde_low_prob_ratio, kde_prob_min, dist_to_centroid_mean, turning_angle_max, mean_speed, turning_entropy,
    output reg tree_out
);

always @(*) begin
    if (kde_low_prob_ratio <= 32'h80000000) begin
        tree_out = 1'b0;
    end else begin
        if (mean_speed <= 32'h002CEDE6) begin
            tree_out = 1'b0;
        end else begin
            if (kde_prob_min <= 32'h244B4320) begin
                if (turning_angle_max <= 32'hFEFA1800) begin
                    if (kde_prob_min <= 32'h22A9D6E0) begin
                        if (dist_to_centroid_mean <= 32'h33144940) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (kde_prob_min <= 32'h23799960) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (kde_prob_min <= 32'h1B6A0470) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end else begin
                tree_out = 1'b0;
            end
        end
    end
end
endmodule

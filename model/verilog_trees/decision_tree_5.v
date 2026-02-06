module decision_tree_5 (
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
            if (turning_angle_max <= 32'h0060FAA0) begin
                tree_out = 1'b0;
            end else begin
                if (kde_prob_min <= 32'h24217440) begin
                    if (kde_prob_min <= 32'h1DFBD540) begin
                        if (turning_angle_max <= 32'hFE0A7080) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (kde_prob_min <= 32'h1E5FADE0) begin
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
end
endmodule

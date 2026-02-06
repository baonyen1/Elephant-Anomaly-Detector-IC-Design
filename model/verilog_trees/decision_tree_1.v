module decision_tree_1 (
    input wire [31:0] kde_low_prob_ratio, kde_prob_min, dist_to_centroid_mean, turning_angle_max, mean_speed, turning_entropy,
    output reg tree_out
);

always @(*) begin
    if (kde_prob_min <= 32'h244B4320) begin
        if (dist_to_centroid_mean <= 32'h344FBF20) begin
            tree_out = 1'b0;
        end else begin
            if (kde_prob_min <= 32'h215A00E0) begin
                if (mean_speed <= 32'h0047F772) begin
                    tree_out = 1'b0;
                end else begin
                    if (kde_prob_min <= 32'h19028FA0) begin
                        if (kde_prob_min <= 32'h17AFF3B0) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (mean_speed <= 32'h3B2CD740) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end else begin
                if (turning_angle_max <= 32'h80175980) begin
                    if (dist_to_centroid_mean <= 32'h4B97DC00) begin
                        tree_out = 1'b0;
                    end else begin
                        if (turning_entropy <= 32'hB3574C80) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (turning_angle_max <= 32'h97BBE680) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end
        end
    end else begin
        tree_out = 1'b0;
    end
end
endmodule

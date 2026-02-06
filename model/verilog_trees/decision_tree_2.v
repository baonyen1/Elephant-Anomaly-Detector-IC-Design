module decision_tree_2 (
    input wire [31:0] kde_low_prob_ratio, kde_prob_min, dist_to_centroid_mean, turning_angle_max, mean_speed, turning_entropy,
    output reg tree_out
);

always @(*) begin
    if (dist_to_centroid_mean <= 32'h6FE3F400) begin
        if (dist_to_centroid_mean <= 32'h334D03A0) begin
            tree_out = 1'b0;
        end else begin
            if (kde_low_prob_ratio <= 32'h80000000) begin
                tree_out = 1'b0;
            end else begin
                if (mean_speed <= 32'h619E5240) begin
                    if (mean_speed <= 32'h0029FA14) begin
                        tree_out = 1'b0;
                    end else begin
                        if (mean_speed <= 32'h1694BAC0) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (mean_speed <= 32'h8C8FC980) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end
        end
    end else begin
        if (kde_prob_min <= 32'h244B4320) begin
            if (turning_entropy <= 32'h5FCD7540) begin
                tree_out = 1'b0;
            end else begin
                if (dist_to_centroid_mean <= 32'h83C20A80) begin
                    if (turning_entropy <= 32'hDA853680) begin
                        if (turning_entropy <= 32'hD1804400) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (mean_speed <= 32'h00815224) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (kde_prob_min <= 32'h20B1EAA0) begin
                        tree_out = 1'b1;
                    end else begin
                        if (mean_speed <= 32'h38289740) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end
        end else begin
            if (dist_to_centroid_mean <= 32'h6FE9F140) begin
                tree_out = 1'b0;
            end else begin
                tree_out = 1'b0;
            end
        end
    end
end
endmodule

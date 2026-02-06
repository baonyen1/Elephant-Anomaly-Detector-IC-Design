module decision_tree_10 (
    input wire [31:0] kde_low_prob_ratio, kde_prob_min, dist_to_centroid_mean, turning_angle_max, mean_speed, turning_entropy,
    output reg tree_out
);

always @(*) begin
    if (mean_speed <= 32'h23F3C880) begin
        if (mean_speed <= 32'h00CAB65C) begin
            if (kde_prob_min <= 32'h188B2DB0) begin
                tree_out = 1'b1;
            end else begin
                tree_out = 1'b0;
            end
        end else begin
            if (kde_low_prob_ratio <= 32'h80000000) begin
                tree_out = 1'b0;
            end else begin
                if (turning_entropy <= 32'h5FCD7540) begin
                    tree_out = 1'b0;
                end else begin
                    if (dist_to_centroid_mean <= 32'h33D40320) begin
                        tree_out = 1'b0;
                    end else begin
                        if (turning_angle_max <= 32'hFE761200) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end
        end
    end else begin
        if (turning_entropy <= 32'hC6CF8200) begin
            if (dist_to_centroid_mean <= 32'h52238FC0) begin
                if (mean_speed <= 32'h8B147980) begin
                    if (kde_prob_min <= 32'h1C32AC10) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    if (turning_angle_max <= 32'h51405380) begin
                        if (turning_angle_max <= 32'h3E310840) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end else begin
                if (dist_to_centroid_mean <= 32'h68A960C0) begin
                    if (dist_to_centroid_mean <= 32'h602B8B00) begin
                        if (kde_prob_min <= 32'h1ECCA690) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    if (kde_low_prob_ratio <= 32'h80000000) begin
                        tree_out = 1'b0;
                    end else begin
                        if (turning_entropy <= 32'hB3574C80) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end
        end else begin
            if (dist_to_centroid_mean <= 32'h3463B140) begin
                tree_out = 1'b0;
            end else begin
                if (kde_prob_min <= 32'h228207C0) begin
                    if (kde_prob_min <= 32'h1E6A22C0) begin
                        tree_out = 1'b1;
                    end else begin
                        if (kde_prob_min <= 32'h207A2EA0) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (kde_prob_min <= 32'h2304FDE0) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end
        end
    end
end
endmodule

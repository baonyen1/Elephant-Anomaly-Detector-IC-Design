module decision_tree_4 (
    input wire [31:0] kde_low_prob_ratio, kde_prob_min, dist_to_centroid_mean, turning_angle_max, mean_speed, turning_entropy,
    output reg tree_out
);

always @(*) begin
    if (dist_to_centroid_mean <= 32'h74761F00) begin
        if (kde_low_prob_ratio <= 32'h80000000) begin
            tree_out = 1'b0;
        end else begin
            if (dist_to_centroid_mean <= 32'h33144940) begin
                tree_out = 1'b0;
            end else begin
                if (dist_to_centroid_mean <= 32'h722B1840) begin
                    if (turning_angle_max <= 32'h01818C0E) begin
                        tree_out = 1'b0;
                    end else begin
                        if (dist_to_centroid_mean <= 32'h702952C0) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    tree_out = 1'b0;
                end
            end
        end
    end else begin
        if (kde_prob_min <= 32'h24201040) begin
            if (dist_to_centroid_mean <= 32'h8410F580) begin
                if (dist_to_centroid_mean <= 32'h83BFAE00) begin
                    if (mean_speed <= 32'h15DBA630) begin
                        if (dist_to_centroid_mean <= 32'h7696CE40) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (dist_to_centroid_mean <= 32'h7B2C89C0) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    tree_out = 1'b0;
                end
            end else begin
                if (dist_to_centroid_mean <= 32'h86370B00) begin
                    if (kde_prob_min <= 32'h1F7ECDD0) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    tree_out = 1'b1;
                end
            end
        end else begin
            tree_out = 1'b0;
        end
    end
end
endmodule

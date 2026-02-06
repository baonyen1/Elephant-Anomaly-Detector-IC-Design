module decision_tree_6 (
    input wire [31:0] kde_low_prob_ratio, kde_prob_min, dist_to_centroid_mean, turning_angle_max, mean_speed, turning_entropy,
    output reg tree_out
);

always @(*) begin
    if (kde_prob_min <= 32'h2457E0C0) begin
        if (dist_to_centroid_mean <= 32'h83459F00) begin
            if (kde_prob_min <= 32'h239D1FA0) begin
                if (kde_prob_min <= 32'h1F996540) begin
                    if (turning_angle_max <= 32'hFEFA1800) begin
                        if (kde_prob_min <= 32'h1A3C3300) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    if (kde_prob_min <= 32'h201707C0) begin
                        tree_out = 1'b0;
                    end else begin
                        if (turning_entropy <= 32'hE89A4680) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end else begin
                tree_out = 1'b0;
            end
        end else begin
            if (kde_prob_min <= 32'h2245F800) begin
                if (turning_entropy <= 32'h447F9510) begin
                    tree_out = 1'b1;
                end else begin
                    tree_out = 1'b1;
                end
            end else begin
                if (turning_entropy <= 32'hE8E5FD80) begin
                    tree_out = 1'b0;
                end else begin
                    if (dist_to_centroid_mean <= 32'h84431F00) begin
                        tree_out = 1'b1;
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

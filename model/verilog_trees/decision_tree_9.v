module decision_tree_9 (
    input wire [31:0] kde_low_prob_ratio, kde_prob_min, dist_to_centroid_mean, turning_angle_max, mean_speed, turning_entropy,
    output reg tree_out
);

always @(*) begin
    if (dist_to_centroid_mean <= 32'h74761F00) begin
        if (kde_low_prob_ratio <= 32'h80000000) begin
            tree_out = 1'b0;
        end else begin
            if (kde_prob_min <= 32'h22C11B20) begin
                if (turning_entropy <= 32'hCB93E800) begin
                    if (dist_to_centroid_mean <= 32'h35AADF40) begin
                        tree_out = 1'b0;
                    end else begin
                        if (mean_speed <= 32'h4EA552C0) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (turning_angle_max <= 32'hFDE98C80) begin
                        if (turning_angle_max <= 32'hC2E6C200) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (turning_angle_max <= 32'hFEA34880) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end else begin
                if (dist_to_centroid_mean <= 32'h28C74580) begin
                    tree_out = 1'b0;
                end else begin
                    tree_out = 1'b0;
                end
            end
        end
    end else begin
        if (kde_prob_min <= 32'h2457E0C0) begin
            if (dist_to_centroid_mean <= 32'h82FD8600) begin
                if (dist_to_centroid_mean <= 32'h82E7D300) begin
                    if (turning_entropy <= 32'h810D0EC0) begin
                        tree_out = 1'b0;
                    end else begin
                        if (mean_speed <= 32'h10153E60) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    tree_out = 1'b0;
                end
            end else begin
                if (dist_to_centroid_mean <= 32'h84E89080) begin
                    if (mean_speed <= 32'h0A9DA510) begin
                        if (kde_prob_min <= 32'h1B715270) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (dist_to_centroid_mean <= 32'h84918500) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
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

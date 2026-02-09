module decision_tree_2 (
    input wire [31:0] kde_very_low_prob_count, kde_low_prob_ratio, dist_to_centroid_mean, step_max, mean_speed, turning_angle_mean, turning_angle_median, hour,
    output reg tree_out
);

always @(*) begin
    if (dist_to_centroid_mean <= 32'h8235B280) begin
        if (mean_speed <= 32'h58BDA240) begin
            if (kde_very_low_prob_count <= 32'h80000000) begin
                if (hour <= 32'hDD174600) begin
                    if (dist_to_centroid_mean <= 32'h1401C840) begin
                        if (turning_angle_median <= 32'hEC48C480) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (turning_angle_mean <= 32'hEB7B9B00) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (dist_to_centroid_mean <= 32'h68B24980) begin
                        if (mean_speed <= 32'h1E0D4810) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (mean_speed <= 32'h1AC7A0C0) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end else begin
                if (mean_speed <= 32'h2DB35560) begin
                    if (turning_angle_mean <= 32'h57619140) begin
                        if (mean_speed <= 32'h01151062) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (dist_to_centroid_mean <= 32'h49D5A9C0) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (dist_to_centroid_mean <= 32'h61CA1AC0) begin
                        if (step_max <= 32'h1EC56C70) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end
        end else begin
            if (turning_angle_median <= 32'h010C0938) begin
                tree_out = 1'b1;
            end else begin
                tree_out = 1'b1;
            end
        end
    end else begin
        if (kde_low_prob_ratio <= 32'h80000000) begin
            tree_out = 1'b0;
        end else begin
            if (turning_angle_median <= 32'h00585F88) begin
                tree_out = 1'b0;
            end else begin
                if (mean_speed <= 32'h3F651800) begin
                    if (dist_to_centroid_mean <= 32'h8BFA4800) begin
                        if (mean_speed <= 32'h1D3BAB10) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (dist_to_centroid_mean <= 32'h8DFC3F80) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (mean_speed <= 32'h4229A4C0) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end
        end
    end
end
endmodule

module decision_tree_4 (
    input wire [31:0] kde_very_low_prob_count, kde_low_prob_ratio, dist_to_centroid_mean, step_max, mean_speed, turning_angle_mean, turning_angle_median, hour,
    output reg tree_out
);

always @(*) begin
    if (turning_angle_mean <= 32'hEB808C00) begin
        if (mean_speed <= 32'h56F30340) begin
            if (kde_very_low_prob_count <= 32'h80000000) begin
                if (turning_angle_mean <= 32'h4F9B0EC0) begin
                    if (mean_speed <= 32'h4573FDC0) begin
                        if (turning_angle_median <= 32'h4F8B7140) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (hour <= 32'h22E8BA40) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (hour <= 32'h0BA2E8C0) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end else begin
                if (turning_angle_median <= 32'h071C7ECC) begin
                    tree_out = 1'b1;
                end else begin
                    if (turning_angle_median <= 32'h07CF33E8) begin
                        tree_out = 1'b0;
                    end else begin
                        if (step_max <= 32'h0D1FC410) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end
        end else begin
            if (hour <= 32'h22E8BA40) begin
                if (step_max <= 32'h31B709E0) begin
                    if (kde_very_low_prob_count <= 32'h80000000) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b1;
                    end
                end else begin
                    tree_out = 1'b1;
                end
            end else begin
                if (step_max <= 32'h3186B240) begin
                    tree_out = 1'b0;
                end else begin
                    if (turning_angle_median <= 32'h98F5BB80) begin
                        tree_out = 1'b1;
                    end else begin
                        if (turning_angle_median <= 32'h9C6F9C00) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end
        end
    end else begin
        if (kde_very_low_prob_count <= 32'h80000000) begin
            tree_out = 1'b1;
        end else begin
            tree_out = 1'b1;
        end
    end
end
endmodule

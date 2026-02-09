module decision_tree_5 (
    input wire [31:0] kde_very_low_prob_count, kde_low_prob_ratio, dist_to_centroid_mean, step_max, mean_speed, turning_angle_mean, turning_angle_median, hour,
    output reg tree_out
);

always @(*) begin
    if (turning_angle_median <= 32'hEB808C00) begin
        if (step_max <= 32'h31657A40) begin
            if (dist_to_centroid_mean <= 32'h7C489E00) begin
                if (dist_to_centroid_mean <= 32'h528956C0) begin
                    if (mean_speed <= 32'h21C65E20) begin
                        if (turning_angle_median <= 32'h0682B718) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (turning_angle_mean <= 32'hCFB24900) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (kde_very_low_prob_count <= 32'h80000000) begin
                        if (kde_low_prob_ratio <= 32'h80000000) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (turning_angle_median <= 32'h0A5DB8A0) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end else begin
                if (turning_angle_mean <= 32'hB3A82C80) begin
                    if (turning_angle_mean <= 32'hAF400680) begin
                        if (turning_angle_median <= 32'h768865C0) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (kde_very_low_prob_count <= 32'h80000000) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (kde_very_low_prob_count <= 32'h80000000) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end
        end else begin
            if (turning_angle_median <= 32'h75E7F740) begin
                if (hour <= 32'h3A2E8BC0) begin
                    if (step_max <= 32'hA12FFA00) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    if (mean_speed <= 32'h588A62C0) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end else begin
                if (turning_angle_mean <= 32'h76414100) begin
                    tree_out = 1'b0;
                end else begin
                    if (mean_speed <= 32'h4E3A6B80) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end
        end
    end else begin
        if (turning_angle_mean <= 32'hEB947200) begin
            tree_out = 1'b1;
        end else begin
            tree_out = 1'b1;
        end
    end
end
endmodule

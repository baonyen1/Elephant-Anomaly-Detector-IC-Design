module decision_tree_9 (
    input wire [31:0] kde_very_low_prob_count, kde_low_prob_ratio, dist_to_centroid_mean, step_max, mean_speed, turning_angle_mean, turning_angle_median, hour,
    output reg tree_out
);

always @(*) begin
    if (turning_angle_median <= 32'hEB47AA00) begin
        if (kde_low_prob_ratio <= 32'h80000000) begin
            if (dist_to_centroid_mean <= 32'h3058B180) begin
                if (hour <= 32'h51745D40) begin
                    if (step_max <= 32'h32004720) begin
                        if (turning_angle_mean <= 32'h28A15FA0) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (step_max <= 32'h654C7D80) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (step_max <= 32'h31B3FA40) begin
                        if (turning_angle_median <= 32'h001BB9FE) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (dist_to_centroid_mean <= 32'h0D57B5F0) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end else begin
                if (turning_angle_mean <= 32'h89691780) begin
                    if (hour <= 32'h51745D40) begin
                        if (hour <= 32'h22E8BA40) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (step_max <= 32'h317A6140) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (step_max <= 32'h31962C20) begin
                        tree_out = 1'b0;
                    end else begin
                        if (turning_angle_mean <= 32'h9A3D6000) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end
        end else begin
            if (turning_angle_median <= 32'h26982BA0) begin
                if (turning_angle_mean <= 32'h1C455BE0) begin
                    if (kde_very_low_prob_count <= 32'h80000000) begin
                        if (dist_to_centroid_mean <= 32'h773712C0) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (turning_angle_median <= 32'h178D6D60) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (turning_angle_median <= 32'h1F067140) begin
                        if (mean_speed <= 32'h24E77FC0) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (turning_angle_median <= 32'h2070B2E0) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end else begin
                if (turning_angle_median <= 32'h296CA9E0) begin
                    tree_out = 1'b0;
                end else begin
                    if (dist_to_centroid_mean <= 32'h830C1780) begin
                        if (dist_to_centroid_mean <= 32'h3A56E860) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (kde_very_low_prob_count <= 32'h80000000) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end
        end
    end else begin
        if (mean_speed <= 32'h01B842EF) begin
            if (turning_angle_median <= 32'hEB70C700) begin
                tree_out = 1'b0;
            end else begin
                tree_out = 1'b1;
            end
        end else begin
            if (turning_angle_median <= 32'hEB99B400) begin
                if (kde_very_low_prob_count <= 32'h80000000) begin
                    tree_out = 1'b0;
                end else begin
                    tree_out = 1'b1;
                end
            end else begin
                tree_out = 1'b1;
            end
        end
    end
end
endmodule

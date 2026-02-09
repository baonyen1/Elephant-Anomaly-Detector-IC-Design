module decision_tree_3 (
    input wire [31:0] kde_very_low_prob_count, kde_low_prob_ratio, dist_to_centroid_mean, step_max, mean_speed, turning_angle_mean, turning_angle_median, hour,
    output reg tree_out
);

always @(*) begin
    if (turning_angle_median <= 32'hEB4AAD80) begin
        if (dist_to_centroid_mean <= 32'h7C214F40) begin
            if (turning_angle_median <= 32'h2F9E1F20) begin
                if (step_max <= 32'h31D22780) begin
                    if (kde_low_prob_ratio <= 32'h80000000) begin
                        if (hour <= 32'h51745D40) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (kde_very_low_prob_count <= 32'h80000000) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (dist_to_centroid_mean <= 32'h0CBADFE0) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end else begin
                if (kde_low_prob_ratio <= 32'h80000000) begin
                    if (step_max <= 32'h31ED90A0) begin
                        if (dist_to_centroid_mean <= 32'h74E92B80) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (mean_speed <= 32'h46175380) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (mean_speed <= 32'h38AE8FC0) begin
                        if (kde_very_low_prob_count <= 32'h80000000) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (dist_to_centroid_mean <= 32'h370ED420) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end
        end else begin
            if (kde_very_low_prob_count <= 32'h80000000) begin
                if (step_max <= 32'h1AE5E3C0) begin
                    if (dist_to_centroid_mean <= 32'h7E449680) begin
                        if (dist_to_centroid_mean <= 32'h7E3FAB00) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (mean_speed <= 32'h0AAF7160) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (turning_angle_mean <= 32'h1262D918) begin
                        tree_out = 1'b0;
                    end else begin
                        if (turning_angle_mean <= 32'h4B569600) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end else begin
                if (hour <= 32'hAE8BA300) begin
                    tree_out = 1'b1;
                end else begin
                    if (dist_to_centroid_mean <= 32'h858D0500) begin
                        if (turning_angle_median <= 32'h9BEDC180) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (step_max <= 32'h104C6A80) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end
        end
    end else begin
        if (step_max <= 32'h00F68D70) begin
            if (dist_to_centroid_mean <= 32'h54687140) begin
                if (turning_angle_mean <= 32'hEC94B780) begin
                    tree_out = 1'b0;
                end else begin
                    tree_out = 1'b1;
                end
            end else begin
                tree_out = 1'b1;
            end
        end else begin
            if (turning_angle_median <= 32'hEB83A200) begin
                if (hour <= 32'h3A2E8BC0) begin
                    tree_out = 1'b1;
                end else begin
                    tree_out = 1'b0;
                end
            end else begin
                tree_out = 1'b1;
            end
        end
    end
end
endmodule

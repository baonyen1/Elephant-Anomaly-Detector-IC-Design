module decision_tree_10 (
    input wire [31:0] kde_very_low_prob_count, kde_low_prob_ratio, dist_to_centroid_mean, step_max, mean_speed, turning_angle_mean, turning_angle_median, hour,
    output reg tree_out
);

always @(*) begin
    if (turning_angle_mean <= 32'hEB4AAD80) begin
        if (mean_speed <= 32'h5415E3C0) begin
            if (hour <= 32'hDD174600) begin
                if (hour <= 32'h0BA2E8C0) begin
                    if (step_max <= 32'h0518E644) begin
                        if (turning_angle_median <= 32'hAD15D480) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (kde_low_prob_ratio <= 32'h80000000) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (turning_angle_mean <= 32'hEA2F3480) begin
                        if (mean_speed <= 32'h000A1130) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (step_max <= 32'h034D8E8E) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end else begin
                if (dist_to_centroid_mean <= 32'h84F93600) begin
                    if (turning_angle_median <= 32'h4BE35A80) begin
                        if (kde_low_prob_ratio <= 32'h80000000) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (turning_angle_mean <= 32'h72851140) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (dist_to_centroid_mean <= 32'h8C075B80) begin
                        if (dist_to_centroid_mean <= 32'h8977F780) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end
        end else begin
            if (step_max <= 32'h31B3FA40) begin
                if (mean_speed <= 32'h552707C0) begin
                    if (turning_angle_median <= 32'h3531C6A0) begin
                        if (step_max <= 32'h2FCC0E60) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (step_max <= 32'h2EE64860) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (turning_angle_median <= 32'h1D7C0550) begin
                        if (mean_speed <= 32'h5847DCC0) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end else begin
                tree_out = 1'b1;
            end
        end
    end else begin
        if (turning_angle_median <= 32'hEB808C00) begin
            if (kde_low_prob_ratio <= 32'h80000000) begin
                tree_out = 1'b0;
            end else begin
                tree_out = 1'b1;
            end
        end else begin
            tree_out = 1'b1;
        end
    end
end
endmodule

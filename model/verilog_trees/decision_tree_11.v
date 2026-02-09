module decision_tree_11 (
    input wire [31:0] kde_very_low_prob_count, kde_low_prob_ratio, dist_to_centroid_mean, step_max, mean_speed, turning_angle_mean, turning_angle_median, hour,
    output reg tree_out
);

always @(*) begin
    if (kde_low_prob_ratio <= 32'h80000000) begin
        if (step_max <= 32'h318688A0) begin
            if (dist_to_centroid_mean <= 32'h1473F3A0) begin
                if (hour <= 32'hC5D17480) begin
                    tree_out = 1'b0;
                end else begin
                    if (step_max <= 32'h109EF3F0) begin
                        tree_out = 1'b0;
                    end else begin
                        if (turning_angle_mean <= 32'hE4BE7F00) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end else begin
                if (turning_angle_mean <= 32'hEB7B9B00) begin
                    if (step_max <= 32'h2A74D700) begin
                        if (mean_speed <= 32'h000A1130) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (hour <= 32'hF45D1780) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    tree_out = 1'b1;
                end
            end
        end else begin
            if (step_max <= 32'h31EBB360) begin
                if (hour <= 32'hA2E8BA80) begin
                    tree_out = 1'b0;
                end else begin
                    if (step_max <= 32'h319E9F60) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end else begin
                if (turning_angle_mean <= 32'h06B96C1C) begin
                    if (mean_speed <= 32'h4454B060) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b1;
                    end
                end else begin
                    if (mean_speed <= 32'h4E85FEC0) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end
        end
    end else begin
        if (turning_angle_mean <= 32'hDC7A9300) begin
            if (mean_speed <= 32'h4F34B680) begin
                if (dist_to_centroid_mean <= 32'h86CCE780) begin
                    if (turning_angle_mean <= 32'h12F79ED0) begin
                        if (kde_very_low_prob_count <= 32'h80000000) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (step_max <= 32'h05ED75E8) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (turning_angle_mean <= 32'h384CE680) begin
                        if (kde_very_low_prob_count <= 32'h80000000) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (dist_to_centroid_mean <= 32'h8B871100) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end else begin
                if (step_max <= 32'h2CD4D7E0) begin
                    if (mean_speed <= 32'h4FB2F400) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    tree_out = 1'b1;
                end
            end
        end else begin
            if (hour <= 32'hC5D17480) begin
                if (step_max <= 32'h0037FD9E) begin
                    tree_out = 1'b0;
                end else begin
                    if (dist_to_centroid_mean <= 32'h31DF3360) begin
                        tree_out = 1'b0;
                    end else begin
                        if (turning_angle_median <= 32'hE6C9A280) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end else begin
                if (dist_to_centroid_mean <= 32'h4D856680) begin
                    tree_out = 1'b1;
                end else begin
                    if (turning_angle_median <= 32'hE999B980) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end
        end
    end
end
endmodule

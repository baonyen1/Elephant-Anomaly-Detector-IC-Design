module decision_tree_7 (
    input wire [31:0] kde_very_low_prob_count, kde_low_prob_ratio, dist_to_centroid_mean, step_max, mean_speed, turning_angle_mean, turning_angle_median, hour,
    output reg tree_out
);

always @(*) begin
    if (kde_very_low_prob_count <= 32'h80000000) begin
        if (turning_angle_mean <= 32'hEB7F5700) begin
            if (hour <= 32'h51745D40) begin
                if (step_max <= 32'h31EEB9C0) begin
                    if (kde_low_prob_ratio <= 32'h80000000) begin
                        if (turning_angle_mean <= 32'h4E976BC0) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (step_max <= 32'h1A1ED8B0) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (mean_speed <= 32'h48F39840) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end else begin
                if (dist_to_centroid_mean <= 32'h06E775E0) begin
                    if (turning_angle_mean <= 32'h6DCBBE40) begin
                        tree_out = 1'b0;
                    end else begin
                        if (turning_angle_mean <= 32'hBA325200) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (hour <= 32'hC5D17480) begin
                        if (turning_angle_median <= 32'hD0688580) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (dist_to_centroid_mean <= 32'h4F53C400) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end
        end else begin
            tree_out = 1'b1;
        end
    end else begin
        if (mean_speed <= 32'h00764E3C) begin
            tree_out = 1'b0;
        end else begin
            if (hour <= 32'h9745D180) begin
                if (mean_speed <= 32'h0D2F4638) begin
                    if (dist_to_centroid_mean <= 32'h55C167C0) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b1;
                    end
                end else begin
                    if (mean_speed <= 32'h0DC86758) begin
                        tree_out = 1'b0;
                    end else begin
                        if (turning_angle_mean <= 32'h07EC5340) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end else begin
                if (dist_to_centroid_mean <= 32'h3DD15120) begin
                    tree_out = 1'b0;
                end else begin
                    if (dist_to_centroid_mean <= 32'h8C075B80) begin
                        if (turning_angle_median <= 32'h01D6E41B) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (turning_angle_mean <= 32'h2F995E80) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end
        end
    end
end
endmodule

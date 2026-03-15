module decision_tree_4 (
    input wire [15:0] kde_prob_mean, kde_prob_night_mean, dist_to_centroid_mean, step_median, mean_speed, accelerate, turning_angle_max, turning_angle_median, is_night,
    output reg tree_out
);

always @(*) begin
    if (kde_prob_night_mean <= 16'h4498) begin
        if (kde_prob_mean <= 16'h19C8) begin
            if (turning_angle_max <= 16'hED84) begin
                if (turning_angle_median <= 16'h1FDA) begin
                    if (turning_angle_max <= 16'h1FB0) begin
                        if (mean_speed <= 16'h04B6) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    if (kde_prob_mean <= 16'h19AA) begin
                        tree_out = 1'b1;
                    end else begin
                        if (dist_to_centroid_mean <= 16'h65DB) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end else begin
                if (mean_speed <= 16'h1D29) begin
                    tree_out = 1'b1;
                end else begin
                    if (step_median <= 16'h123D) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end
        end else begin
            if (step_median <= 16'h2EC2) begin
                if (mean_speed <= 16'h17E5) begin
                    tree_out = 1'b0;
                end else begin
                    if (kde_prob_mean <= 16'h1D4D) begin
                        if (kde_prob_mean <= 16'h1D1C) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (turning_angle_max <= 16'h0454) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end else begin
                if (step_median <= 16'h3630) begin
                    if (kde_prob_mean <= 16'h1B86) begin
                        tree_out = 1'b1;
                    end else begin
                        if (dist_to_centroid_mean <= 16'h2C16) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    tree_out = 1'b0;
                end
            end
        end
    end else begin
        if (kde_prob_mean <= 16'h3568) begin
            if (mean_speed <= 16'h5A1D) begin
                if (turning_angle_median <= 16'h3180) begin
                    if (turning_angle_max <= 16'h3162) begin
                        if (step_median <= 16'h0D38) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        tree_out = 1'b1;
                    end
                end else begin
                    tree_out = 1'b0;
                end
            end else begin
                if (kde_prob_mean <= 16'h23BC) begin
                    if (dist_to_centroid_mean <= 16'h648C) begin
                        tree_out = 1'b1;
                    end else begin
                        tree_out = 1'b0;
                    end
                end else begin
                    tree_out = 1'b1;
                end
            end
        end else begin
            if (turning_angle_max <= 16'hF1DA) begin
                if (dist_to_centroid_mean <= 16'h10A8) begin
                    if (kde_prob_night_mean <= 16'h7176) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b1;
                    end
                end else begin
                    if (dist_to_centroid_mean <= 16'h2299) begin
                        if (dist_to_centroid_mean <= 16'h228E) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (step_median <= 16'h2E34) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end else begin
                if (turning_angle_median <= 16'hF212) begin
                    tree_out = 1'b1;
                end else begin
                    if (mean_speed <= 16'h5948) begin
                        if (kde_prob_mean <= 16'h3DC4) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end
        end
    end
end
endmodule

module decision_tree_12 (
    input wire [31:0] kde_low_prob_ratio, kde_prob_min, dist_to_centroid_mean, turning_angle_max, mean_speed, turning_entropy,
    output reg tree_out
);

always @(*) begin
    if (kde_prob_min <= 32'h241EACA0) begin
        if (dist_to_centroid_mean <= 32'h33144940) begin
            tree_out = 1'b0;
        end else begin
            if (mean_speed <= 32'h00422E65) begin
                tree_out = 1'b0;
            end else begin
                if (mean_speed <= 32'h1C745F60) begin
                    if (turning_angle_max <= 32'hFE5B3D80) begin
                        if (kde_prob_min <= 32'h20ACA2E0) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (kde_prob_min <= 32'h1D1234D0) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (mean_speed <= 32'h3730AAC0) begin
                        if (turning_angle_max <= 32'h45974880) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (mean_speed <= 32'h3791E3C0) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end
        end
    end else begin
        tree_out = 1'b0;
    end
end
endmodule

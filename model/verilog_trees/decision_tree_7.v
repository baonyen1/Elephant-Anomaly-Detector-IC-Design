module decision_tree_7 (
    input wire [31:0] kde_low_prob_ratio, kde_prob_min, dist_to_centroid_mean, turning_angle_max, mean_speed, turning_entropy,
    output reg tree_out
);

always @(*) begin
    if (dist_to_centroid_mean <= 32'h6F7E0240) begin
        if (kde_low_prob_ratio <= 32'h80000000) begin
            tree_out = 1'b0;
        end else begin
            if (turning_entropy <= 32'hFA13A400) begin
                if (turning_entropy <= 32'hE89A4680) begin
                    if (dist_to_centroid_mean <= 32'h34725A80) begin
                        tree_out = 1'b0;
                    end else begin
                        if (turning_angle_max <= 32'hDBB93900) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (turning_entropy <= 32'hEE86A300) begin
                        if (dist_to_centroid_mean <= 32'h389D12E0) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (mean_speed <= 32'h06F41FB0) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end else begin
                if (turning_angle_max <= 32'h1EED1BA0) begin
                    if (kde_prob_min <= 32'h1DEC2580) begin
                        tree_out = 1'b1;
                    end else begin
                        if (kde_prob_min <= 32'h238A1F00) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (dist_to_centroid_mean <= 32'h32FF7A40) begin
                        tree_out = 1'b0;
                    end else begin
                        if (kde_prob_min <= 32'h1EE827F0) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end
            end
        end
    end else begin
        if (kde_low_prob_ratio <= 32'h80000000) begin
            tree_out = 1'b0;
        end else begin
            if (kde_prob_min <= 32'h24201040) begin
                if (mean_speed <= 32'h003D3464) begin
                    tree_out = 1'b0;
                end else begin
                    if (dist_to_centroid_mean <= 32'h811C2600) begin
                        if (turning_entropy <= 32'h810D0EC0) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (turning_entropy <= 32'hE89A4680) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end else begin
                tree_out = 1'b0;
            end
        end
    end
end
endmodule

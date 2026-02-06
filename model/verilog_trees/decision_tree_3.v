module decision_tree_3 (
    input wire [31:0] kde_low_prob_ratio, kde_prob_min, dist_to_centroid_mean, turning_angle_max, mean_speed, turning_entropy,
    output reg tree_out
);

always @(*) begin
    if (mean_speed <= 32'h23F3C880) begin
        if (turning_entropy <= 32'h8801A800) begin
            if (turning_angle_max <= 32'hB9044780) begin
                if (dist_to_centroid_mean <= 32'h6B26DBC0) begin
                    if (kde_prob_min <= 32'h1C43AD70) begin
                        tree_out = 1'b1;
                    end else begin
                        if (kde_low_prob_ratio <= 32'h80000000) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    if (dist_to_centroid_mean <= 32'h7BCBF000) begin
                        if (kde_low_prob_ratio <= 32'h80000000) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        tree_out = 1'b1;
                    end
                end
            end else begin
                tree_out = 1'b0;
            end
        end else begin
            if (dist_to_centroid_mean <= 32'h71B94680) begin
                if (kde_prob_min <= 32'h227C8360) begin
                    if (mean_speed <= 32'h11184F00) begin
                        if (turning_angle_max <= 32'h06351030) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (dist_to_centroid_mean <= 32'h67B12780) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end
                end else begin
                    tree_out = 1'b0;
                end
            end else begin
                if (kde_prob_min <= 32'h209031E0) begin
                    if (dist_to_centroid_mean <= 32'h746AB740) begin
                        if (mean_speed <= 32'h08AF3104) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b0;
                        end
                    end else begin
                        if (dist_to_centroid_mean <= 32'h8250F380) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end else begin
                    if (dist_to_centroid_mean <= 32'h71EF6700) begin
                        tree_out = 1'b0;
                    end else begin
                        tree_out = 1'b0;
                    end
                end
            end
        end
    end else begin
        if (kde_low_prob_ratio <= 32'h80000000) begin
            if (mean_speed <= 32'h2403F4E0) begin
                tree_out = 1'b0;
            end else begin
                tree_out = 1'b0;
            end
        end else begin
            if (kde_prob_min <= 32'h24655AC0) begin
                if (dist_to_centroid_mean <= 32'h33A7C900) begin
                    tree_out = 1'b0;
                end else begin
                    if (kde_prob_min <= 32'h1F8DE5A0) begin
                        if (kde_prob_min <= 32'h19898370) begin
                            tree_out = 1'b1;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end else begin
                        if (turning_angle_max <= 32'h209216A0) begin
                            tree_out = 1'b0;
                        end else begin
                            tree_out = 1'b1;
                        end
                    end
                end
            end else begin
                if (mean_speed <= 32'h24F62820) begin
                    tree_out = 1'b0;
                end else begin
                    tree_out = 1'b0;
                end
            end
        end
    end
end
endmodule

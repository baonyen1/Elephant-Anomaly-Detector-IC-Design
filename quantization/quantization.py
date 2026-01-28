import pandas as pd
import numpy as np

# 🔹 Đường dẫn file
input_path = "elephant_6features_cleaned.csv"
output_quantized_path1 = "Quantized_Combined_Features.csv"
output_quantized_path2 = "C:\\Users\\nguye\\Documents\\TKVM\\training\\model\\Quantized_Combined_Features.csv"
output_scale_table_path = "Quantization_Scales.csv"

# 🔹 Đọc dữ liệu
df = pd.read_csv(input_path)

print("📊 Kiểu dữ liệu các cột:")
print(df.dtypes)

# 🔹 Tách label ra riêng (không quantize)
label_col = "is_outside"
labels = df[label_col]
features_df = df.drop(columns=[label_col])

quantized_data = pd.DataFrame()
scale_table = []

# 🔹 Hàm lượng tử hóa sang uint32
def quantize_column_to_uint32(series):
    series = series.replace([np.inf, -np.inf], np.nan)

    if series.isnull().all():
        return pd.Series([0]*len(series), dtype='uint32'), 1.0, 0.0, 0.0

    min_val = series.min()
    max_val = series.max()

    scale = (max_val - min_val) / (2**32 - 1) if max_val != min_val else 1.0
    series_filled = series.fillna(min_val)

    quantized = ((series_filled - min_val) / scale).round().astype('uint32')
    return quantized, scale, min_val, max_val

# 🔹 Quantize từng feature
for col in features_df.columns:
    q_col, scale, min_val, max_val = quantize_column_to_uint32(features_df[col])
    quantized_data[col] = q_col

    scale_table.append({
        "feature": col,
        "scale": scale,
        "min": min_val,
        "max": max_val
    })

# 🔹 Gắn label lại
quantized_data[label_col] = labels

# 🔹 Lưu file
quantized_data.to_csv(output_quantized_path1, index=False)
quantized_data.to_csv(output_quantized_path2, index=False)
pd.DataFrame(scale_table).to_csv(output_scale_table_path, index=False)

print("\n✅ Quantization hoàn tất!")
print("• Data:", output_quantized_path1, output_quantized_path2)
print("• Scale table:", output_scale_table_path)

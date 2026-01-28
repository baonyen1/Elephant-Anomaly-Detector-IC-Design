import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import csv
import json

# 🔹 Đường dẫn file gốc và file đầu ra
input_path = "elephant_6features_cleaned.csv"
output_quantized_path = "Quantized_Combined_Features.csv"
output_scale_table_path = "Quantization_Scales.csv"
output_label_mapping_csv = "label_encoding_mapping.csv"
output_label_mapping_json = "label_encoding_mapping.json"

# 🔹 Đọc dữ liệu
df = pd.read_csv(input_path)
# 🔹 Mã hóa các cột dạng chuỗi (object)
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# === Xuất mapping ra CSV (mỗi dòng: cột,giá trị gốc,giá trị mã hóa) ===
with open(output_label_mapping_csv, "w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["column", "original_value", "encoded_value"])
    for col, le in label_encoders.items():
        for idx, value in enumerate(le.classes_):
            writer.writerow([col, value, idx])

# === Xuất mapping ra JSON (dễ load lại để giải mã ngược) ===
label_mapping_dict = {
    col: {value: int(idx) for idx, value in enumerate(le.classes_)}
    for col, le in label_encoders.items()
}
with open(output_label_mapping_json, "w", encoding="utf-8") as f:
    json.dump(label_mapping_dict, f, ensure_ascii=False, indent=2)

# 🔹 Tách các cột số
numeric_df = df.select_dtypes(include=['int64', 'float64']).copy()
quantized_data = pd.DataFrame()
scale_table = []

# 🔹 Hàm lượng tử hóa từng cột sang uint32
def quantize_column_to_uint32(series):
    # Loại bỏ giá trị không hợp lệ
    series = series.replace([np.inf, -np.inf], np.nan)
    if series.isnull().all():
        quantized = pd.Series([0] * len(series), index=series.index, dtype='uint32')
        return quantized, 1.0, 0.0, 0.0
    min_val = series.min()
    max_val = series.max()
    scale = (max_val - min_val) / (2**32 - 1) if max_val != min_val else 1.0
    series_filled = series.fillna(min_val)
    quantized = ((series_filled - min_val) / scale).round().astype('uint32')
    return quantized, scale, min_val, max_val

# 🔹 Lượng tử hóa toàn bộ cột số
for col in numeric_df.columns:
    try:
        quantized_data[col], scale, min_val, max_val = quantize_column_to_uint32(numeric_df[col])
        scale_table.append({
            'feature': col,
            'scale': scale,
            'min': min_val,
            'max': max_val
        })
    except Exception as e:
        print(f"⚠️ Lỗi khi lượng tử hóa cột {col}: {e}")

# 🔹 Gộp lại với phần dữ liệu không phải số (nếu có)
non_numeric_df = df.select_dtypes(exclude=['int64', 'float64'])
quantized_data = pd.concat([quantized_data, non_numeric_df], axis=1)

# 🔹 Lưu dữ liệu
quantized_data.to_csv(output_quantized_path, index=False)
pd.DataFrame(scale_table).to_csv(output_scale_table_path, index=False)

print("✅ Đã lượng tử hóa, lưu dữ liệu và xuất mapping mã hóa thành công!")
print(f"• File mapping dạng CSV: {output_label_mapping_csv}")
print(f"• File mapping dạng JSON: {output_label_mapping_json}")

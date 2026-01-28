import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =========================
# 1. ĐỌC VÀ CHUẨN BỊ DỮ LIỆU
# =========================
print("⏳ Đang tải dữ liệu...")
df = pd.read_csv('elephant_features_kde_enhanced.csv')

# Loại bỏ timestamp và thay thế vô cực
if 'timestamp' in df.columns:
    df = df.drop(columns=['timestamp'])
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

# =========================
# 2. FILTER SELECTED FEATURES (FPGA)
# =========================
SELECTED_FEATURES = [
    "kde_low_prob_ratio",
    "kde_prob_min",
    "dist_to_centroid_mean",
    "turning_angle_max",
    "mean_speed",
    "turning_entropy"
]

print("\n🔍 Filtering selected 6 features for FPGA...")

# Kiểm tra xem đủ cột chưa
missing_features = [f for f in SELECTED_FEATURES if f not in df.columns]
if missing_features:
    raise ValueError(f"❌ Missing features in dataset: {missing_features}")

# Chỉ giữ lại 6 feature + label
df = df[SELECTED_FEATURES + ["is_outside"]]

print("✅ Selected features:")
for f in SELECTED_FEATURES:
    print("-", f)

# =========================
# 3. LỌC DỮ LIỆU RÁC (DATA CLEANING) - QUAN TRỌNG
# =========================
print("\n🧹 Đang thực hiện lọc dữ liệu rác (Artifacts)...")
original_count = len(df)

# LOGIC LỌC:
# Rác thường xuất hiện khi resample bị thiếu dữ liệu và điền số 0 vào tất cả các cột.
# Chúng ta lọc bỏ những dòng mà:
# 1. Khoảng cách tới tâm đàn gần như bằng 0 (dist_to_centroid_mean < 0.1 mét)
# 2. VÀ Tốc độ bằng 0 (mean_speed == 0)
# (Điều này vô lý về mặt vật lý -> chắc chắn là rác)

mask_garbage = (df['dist_to_centroid_mean'] < 0.1) & \
               (df['mean_speed'] == 0)

# Nếu bạn muốn chắc chắn hơn nữa, có thể thêm điều kiện các cột khác cũng = 0
# mask_garbage = mask_garbage & (df['turning_angle_max'] == 0)

# Thực hiện lọc (giữ lại những dòng KHÔNG phải rác)
df_clean = df[~mask_garbage].copy()

# Báo cáo kết quả lọc
removed_count = original_count - len(df_clean)
print(f"   - Tổng số dòng ban đầu: {original_count}")
print(f"   - Số dòng rác (toàn số 0) bị loại bỏ: {removed_count}")
print(f"   - Số dòng sạch còn lại: {len(df_clean)}")

# Kiểm tra an toàn: Xem có lỡ tay xóa mất nhãn 'Outside' nào không?
deleted_outsiders = df[mask_garbage]['is_outside'].sum()
if deleted_outsiders > 0:
    print(f"⚠️ CẢNH BÁO: Có {deleted_outsiders} dòng nhãn 'Outside' bị xóa. Cần kiểm tra lại nếu số này lớn!")
else:
    print("✅ AN TOÀN: Không có dữ liệu bất thường (Outside) nào bị xóa nhầm.")

# Cập nhật lại dataframe chính
df = df_clean

# =========================
# 4. LƯU DATASET SẠCH
# =========================
print("\nDataset shape final:", df.shape)

output_filename = "elephant_6features_cleaned.csv"
df.to_csv(output_filename, index=False)
print(f"🎉 Đã lưu file sạch: {output_filename}")

# Hiển thị 5 dòng đầu để kiểm tra
print("\n5 dòng dữ liệu đầu tiên:")
print(df.head())
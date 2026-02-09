import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier

# =========================
# 1. ĐỌC VÀ CHUẨN BỊ DỮ LIỆU
# =========================
print("⏳ Đang tải dữ liệu...")
df = pd.read_csv('elephant_features_kde.csv')

# Loại bỏ timestamp và thay thế vô cực
if 'timestamp' in df.columns:
    df = df.drop(columns=['timestamp'])

df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

LABEL_COL = "is_outside"


def plot_gini_by_group(gini_df, feature_groups, top_k=None):
    """
    Vẽ Gini Importance cho từng nhóm đặc trưng
    
    Parameters
    ----------
    gini_df : DataFrame
        Bảng gồm ['feature', 'gini']
    feature_groups : dict
        {group_name: [feature1, feature2, ...]}
    top_k : int or None
        Nếu set, chỉ vẽ top_k feature có gini cao nhất trong mỗi nhóm
    """

    for group_name, feats in feature_groups.items():
        sub = gini_df[gini_df['feature'].isin(feats)].copy()

        if sub.empty:
            print(f"[{group_name}] không có feature hợp lệ → bỏ qua")
            continue

        sub = sub.sort_values('gini', ascending=False)

        if top_k is not None:
            sub = sub.head(top_k)

        plt.figure(figsize=(8, 4))
        plt.barh(sub['feature'], sub['gini'])
        plt.gca().invert_yaxis()
        plt.title(f"Gini Importance – Group: {group_name}")
        plt.xlabel("Gini Importance")
        plt.tight_layout()
        plt.show()


# =========================
# 2. KHAI BÁO NHÓM ĐẶC TRƯNG
# =========================
feature_groups = {
    'kde': [
        'kde_prob_mean', 'kde_prob_std',
        'kde_prob_day_mean', 'kde_prob_night_mean',
        'kde_prob_adaptive_mean', 'kde_low_prob_ratio',
        'kde_very_low_prob_count'
    ],
    'centroid': [
        'dist_to_centroid_mean'
    ],
    'step': [
        'step_mean', 'step_std', 'step_max', 'step_median'
    ],
    'speed': [
        'mean_speed', 'accelerate',
        'speed_roll_var_4h_mean', 'speed_roll_var_8h_mean',
        'accel_roll_var_4h_mean', 'accel_roll_var_8h_mean'
    ],
    'turning': [
        'turning_angle_mean', 'turning_angle_std',
        'turning_angle_max', 'turning_angle_median',
        'sharp_turns_ratio', 'moderate_turns_ratio',
        'turning_entropy'
    ],
    'time': [
        'hour', 'is_night'
    ]
}

# =========================
# 3. TRAIN RF ĐỂ LẤY GINI
# =========================
print("\n🌲 Training Random Forest để tính Gini Importance...")

X_all = df.drop(columns=[LABEL_COL])
y = df[LABEL_COL]

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=2,
    criterion='gini',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

rf.fit(X_all, y)

gini_df = pd.DataFrame({
    'feature': X_all.columns,
    'gini': rf.feature_importances_
}).sort_values('gini', ascending=False)


print("\n📊 Vẽ Gini Importance theo từng nhóm feature...")
plot_gini_by_group(
    gini_df=gini_df,
    feature_groups=feature_groups,
    top_k=None   # hoặc top_k=5 nếu muốn gọn
)


print("\n📊 Top 10 feature theo Gini:")
print(gini_df.head(10))

# =========================
# 4. CHỌN FEATURE THEO NHÓM
# =========================
selected_features = []

print("\n🔍 Chọn feature đại diện cho từng nhóm:")

for group, feats in feature_groups.items():
    sub = gini_df[gini_df['feature'].isin(feats)]

    if sub.empty:
        print(f"[{group}] ⚠️ Không có feature hợp lệ")
        continue
    
    sub = sub.sort_values('gini', ascending=False)

    if group == 'kde':
        # Giữ TOP 2 feature KDE
        top_kde = sub.head(2)
        selected_features.extend(top_kde['feature'].tolist())
    
        print(f"[{group}] → chọn 2 feature:")
        for _, row in top_kde.iterrows():
            print(f"    - {row['feature']} (gini = {row['gini']:.4f})")
    elif group == 'turning':
        top_feats = sub.head(2)       
        for _, row in top_feats.iterrows():
            selected_features.append(row['feature'])
            print(f"[{group}] → {row['feature']} (gini = {row['gini']:.4f})")
    else:
        # Các nhóm khác chỉ giữ 1
        best = sub.iloc[0]
        selected_features.append(best['feature'])

        print(
            f"[{group}] → {best['feature']} "
            f"(gini = {best['gini']:.4f})"
        )
    
    best = sub.iloc[0]
    selected_features.append(best['feature'])

    print(
        f"[{group}] → {best['feature']} "
        f"(gini = {best['gini']:.4f})"
    )

print("\n✅ FEATURE SET ĐƯỢC CHỌN:")
print(selected_features)

# =========================
# 5. TẠO DATASET CHỈ VỚI FEATURE ĐƯỢC CHỌN
# =========================
df = df[selected_features + [LABEL_COL]]

# BẮT BUỘC: đảm bảo không có cột trùng tên
df = df.loc[:, ~df.columns.duplicated()]


print("\n🧹 Đang lọc dữ liệu rác...")

original_count = len(df)

if 'dist_to_centroid_mean' in df.columns and 'mean_speed' in df.columns:
    mask_garbage = (
        (df['dist_to_centroid_mean'] < 0.1) &
        (df['mean_speed'] == 0)
    )
else:
    print("⚠️ Không đủ cột để lọc rác → bỏ qua bước này")
    mask_garbage = pd.Series(False, index=df.index)

df_clean = df.loc[~mask_garbage].copy()

removed_count = original_count - len(df_clean)

print(f"   - Tổng số dòng ban đầu: {original_count}")
print(f"   - Số dòng rác bị loại bỏ: {removed_count}")
print(f"   - Số dòng sạch còn lại: {len(df_clean)}")

deleted_outsiders = df.loc[mask_garbage, LABEL_COL].sum()
if deleted_outsiders > 0:
    print(f"⚠️ CẢNH BÁO: {deleted_outsiders} dòng Outside bị xóa")
else:
    print("✅ Không xóa nhầm dữ liệu Outside")

df = df_clean

# =========================
# 7. LƯU FILE CSV CHO FPGA
# =========================
print("\nDataset shape final:", df.shape)

output_filename = "elephant_features_selected.csv"
df.to_csv(output_filename, index=False)

print(f"🎉 Đã lưu file: {output_filename}")

print("\n5 dòng dữ liệu đầu tiên:")
print(df.head())

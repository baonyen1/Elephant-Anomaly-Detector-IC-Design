import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier

# =========================
# 1. ĐỌC VÀ CHUẨN BỊ DỮ LIỆU
# =========================
print("⏳ Đang tải dữ liệu...")
df = pd.read_csv('elephant_features_improved.csv')

if 'timestamp' in df.columns:
    df = df.drop(columns=['timestamp'])

df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

LABEL_COL = "is_outside"

# ============================================================
# LOẠI BỎ DATA LEAKAGE FEATURES
# ============================================================
# Các features này được tính TRỰC TIẾP hoặc GIÁN TIẾP từ nhãn is_outside
# Nếu giữ lại, model sẽ "học thuộc" công thức tạo nhãn → accuracy 100% giả tạo

# 1. Hard-coded leakage features (đã biết trước)
known_leakage_cols = [
    # --- Debug columns ---
    'point_is_outside_old_OR',

    # --- Anomaly score columns (tính trực tiếp từ nhãn) ---
    'anomaly_score_mean',
    'anomaly_score_max',
    'anomaly_score_2h',

    # --- KDE leakage (thành phần tính spatial_anomaly) ---
    'kde_low_prob_ratio',       # (kde_prob < 0.2) → spatial_anomaly
    'kde_very_low_prob_count',  # (kde_prob < 0.1) → spatial_anomaly

    # --- Stationary leakage (thành phần tính stationary_anomaly) ---
    'stationary_ratio',         # từ is_stationary → stationary_anomaly
    'stationary_streak_max',    # trực tiếp trong stationary_anomaly formula
    'rolling_speed_4h_mean',    # trong stationary_anomaly condition
]

# 2. Tự động phát hiện leakage features bằng pattern matching
all_cols = [c for c in df.columns if c != LABEL_COL]
leakage_patterns = [
    'anomaly_score',      # tất cả anomaly score columns
    'point_is_outside',   # tất cả point_is_outside variants
]
auto_detected_leakage = [
    c for c in all_cols
    if any(p in c for p in leakage_patterns)
]

# 3. Combine tất cả leakage features
all_leakage = list(set(known_leakage_cols + auto_detected_leakage))
all_leakage = [c for c in all_leakage if c in df.columns]  # chỉ giữ columns có thật

df = df.drop(columns=all_leakage)
print(f"\n🚫 Loại {len(all_leakage)} leakage features:")
for col in sorted(all_leakage):
    print(f"   - {col}")

print(f"   Dataset shape: {df.shape}")
print(f"   Nhãn is_outside: {df[LABEL_COL].value_counts().to_dict()}")


# =========================
# 2. KHAI BÁO NHÓM ĐẶC TRƯNG (CẬP NHẬT CHO PHIÊN BẢN CẢI TIẾN)
# =========================
feature_groups = {

    # --- Nhóm vị trí không gian ---
    # LƯU Ý: bỏ kde_low_prob_ratio và kde_very_low_prob_count vì
    # chúng là thành phần tính spatial_anomaly → gián tiếp là nhãn (LEAKAGE)
    'kde': [
        'kde_prob_mean',
        'kde_prob_std',
        'kde_prob_day_mean',
        'kde_prob_night_mean',
        'kde_prob_adaptive_mean',   # KDE thích ứng ngày/đêm
        # 'kde_low_prob_ratio',     ← LEAKAGE: dùng để tính spatial_anomaly
        # 'kde_very_low_prob_count' ← LEAKAGE: dùng để tính spatial_anomaly
    ],

    'centroid': [
        'dist_to_centroid_mean',
    ],

    # --- Nhóm di chuyển bước chân ---
    'step': [
        'step_mean',
        'step_std',
        'step_max',
        'step_median',
    ],

    # --- Nhóm tốc độ & gia tốc ---
    'speed': [
        'mean_speed',
        'accelerate',               # mean |gia tốc| tuyệt đối
        'speed_roll_var_4h_mean',
        'speed_roll_var_8h_mean',
        'accel_roll_var_4h_mean',   # MỚI: rolling variance gia tốc
        'accel_roll_var_8h_mean',
    ],

    # --- Nhóm góc rẽ (dùng turning_angle_CLEAN — đã lọc GPS noise) ---
    'turning': [
        'turning_angle_mean',       # dựa trên turning_angle_clean
        'turning_angle_std',
        'turning_angle_max',
        'turning_angle_median',
        'sharp_turns_ratio',        # tỷ lệ rẽ > 90°
        'moderate_turns_ratio',     # tỷ lệ rẽ 30°–90°
        'turning_entropy',          # Shannon entropy của pattern rẽ
    ],

    # LƯU Ý: Nhóm 'stationary' đã bị loại bỏ vì các features
    # stationary_ratio, stationary_streak_max, rolling_speed_4h_mean
    # là thành phần tính stationary_anomaly → LEAKAGE

    # --- Nhóm thời gian ---
    'time': [
        'hour',
        'is_night',
    ],
}

# Kiểm tra features có thực sự tồn tại trong df không
X_cols = [c for c in df.columns if c != LABEL_COL]
for group, feats in feature_groups.items():
    missing = [f for f in feats if f not in X_cols]
    if missing:
        print(f"⚠️  [{group}] thiếu: {missing}")


# =========================
# 3. TRAIN RF ĐỂ LẤY GINI IMPORTANCE
# =========================
print("\n🌲 Training Random Forest để tính Gini Importance...")

X_all = df[X_cols]
y     = df[LABEL_COL]

print(f"   Class distribution: {y.value_counts().to_dict()}")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=2,
    criterion='gini',
    class_weight='balanced',   # bù mất cân bằng nhãn
    n_jobs=-1,
    random_state=42
)
rf.fit(X_all, y)

gini_df = pd.DataFrame({
    'feature': X_all.columns,
    'gini':    rf.feature_importances_,
}).sort_values('gini', ascending=False).reset_index(drop=True)

print("\n📊 Top 15 feature theo Gini Importance (toàn bộ):")
print(gini_df.head(15).to_string(index=False))


# =========================
# 4. VẼ GINI THEO NHÓM
# =========================
def plot_gini_by_group(gini_df, feature_groups, top_k=None):
    n_groups = len(feature_groups)
    fig, axes = plt.subplots(
        nrows=(n_groups + 1) // 2,
        ncols=2,
        figsize=(14, 4 * ((n_groups + 1) // 2))
    )
    axes = axes.flatten()

    for idx, (group_name, feats) in enumerate(feature_groups.items()):
        sub = gini_df[gini_df['feature'].isin(feats)].copy()
        ax  = axes[idx]

        if sub.empty:
            ax.set_title(f"{group_name} — (không có data)")
            ax.axis('off')
            continue

        sub = sub.sort_values('gini', ascending=True)
        if top_k:
            sub = sub.tail(top_k)

        colors = ['#e74c3c' if g > sub['gini'].median() else '#3498db'
                  for g in sub['gini']]
        ax.barh(sub['feature'], sub['gini'], color=colors, edgecolor='white')
        ax.set_title(f"Gini — {group_name}", fontweight='bold')
        ax.set_xlabel("Gini Importance")
        ax.axvline(sub['gini'].mean(), color='orange', linestyle='--',
                   linewidth=1, label='mean')
        ax.legend(fontsize=8)

    # Ẩn subplot thừa
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Gini Importance theo nhóm Feature\n(đỏ = trên median, xanh = dưới median)",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('gini_by_group_improved.png', dpi=150, bbox_inches='tight')
    print("📊 Biểu đồ đã lưu: gini_by_group_improved.png")
    plt.show()

plot_gini_by_group(gini_df, feature_groups, top_k=None)


# =========================
# 5. CHỌN FEATURE ĐẠI DIỆN THEO NHÓM (ĐÃ SỬA BUG TRÙNG LẶP)
# =========================

# Số feature giữ lại theo nhóm — điều chỉnh tại đây
N_FEATURES_PER_GROUP = {
    'kde':      2,   # KDE có nhiều feature tương quan → giữ 2 đại diện
    'centroid': 1,
    'step':     1,
    'speed':    2,   # mean_speed + 1 variance feature
    'turning':  2,   # turning_entropy + 1 ratio
    'time':     1,
    # LƯU Ý: 'stationary' đã bị loại bỏ — leakage features
}

selected_features = []
print("\n🔍 Chọn feature đại diện cho từng nhóm:")
print(f"{'Nhóm':<18} {'Feature được chọn':<35} {'Gini':>8}")
print("-" * 65)

for group, feats in feature_groups.items():
    sub = gini_df[gini_df['feature'].isin(feats)].sort_values('gini', ascending=False)

    if sub.empty:
        print(f"{'[' + group + ']':<18} ⚠️ Không có feature hợp lệ")
        continue

    n = N_FEATURES_PER_GROUP.get(group, 1)
    top_n = sub.head(n)

    for _, row in top_n.iterrows():
        # Tránh trùng lặp (bug code cũ thêm best 2 lần)
        if row['feature'] not in selected_features:
            selected_features.append(row['feature'])
            print(f"{'[' + group + ']':<18} {row['feature']:<35} {row['gini']:>8.4f}")

print(f"\n✅ Tổng số features được chọn: {len(selected_features)}")
print(f"   {selected_features}")


# =========================
# 6. LỌC DỮ LIỆU RÁC
# =========================
print("\n🧹 Đang lọc dữ liệu rác...")

df_selected = df[selected_features + [LABEL_COL]].copy()
df_selected = df_selected.loc[:, ~df_selected.columns.duplicated()]

original_count = len(df_selected)

# Lọc dòng rác: đứng yên hoàn toàn và ở tâm — không có thông tin
has_centroid = 'dist_to_centroid_mean' in df_selected.columns
has_speed    = 'mean_speed' in df_selected.columns

if has_centroid and has_speed:
    mask_garbage = (
        (df_selected['dist_to_centroid_mean'] < 0.1) &
        (df_selected['mean_speed'] == 0)
    )
elif has_speed:
    # Fallback: chỉ dùng speed nếu không có centroid
    mask_garbage = (df_selected['mean_speed'] == 0)
else:
    print("⚠️ Không đủ cột để lọc rác → bỏ qua")
    mask_garbage = pd.Series(False, index=df_selected.index)

df_clean = df_selected.loc[~mask_garbage].copy()

removed_count    = original_count - len(df_clean)
deleted_outside  = df_selected.loc[mask_garbage, LABEL_COL].sum()

print(f"   Tổng dòng ban đầu  : {original_count}")
print(f"   Dòng rác bị loại   : {removed_count}")
print(f"   Dòng sạch còn lại  : {len(df_clean)}")

if deleted_outside > 0:
    print(f"⚠️ CẢNH BÁO: {int(deleted_outside)} dòng Outside bị xóa cùng rác!")
else:
    print("✅ Không xóa nhầm dữ liệu Outside")


# =========================
# 7. THỐNG KÊ PHÂN PHỐI NHÃN SAU LỌC
# =========================
label_dist = df_clean[LABEL_COL].value_counts()
label_pct  = df_clean[LABEL_COL].value_counts(normalize=True) * 100

print(f"\n📊 Phân phối nhãn sau khi lọc:")
print(f"   Inside  (0): {label_dist.get(0, 0):5d} dòng  ({label_pct.get(0, 0):.1f}%)")
print(f"   Outside (1): {label_dist.get(1, 0):5d} dòng  ({label_pct.get(1, 0):.1f}%)")

imbalance_ratio = label_dist.get(0, 1) / max(label_dist.get(1, 1), 1)
if imbalance_ratio > 5:
    print(f"⚠️ Mất cân bằng cao ({imbalance_ratio:.1f}:1) → nên dùng class_weight='balanced' khi train ML")
else:
    print(f"✅ Tỷ lệ cân bằng chấp nhận được ({imbalance_ratio:.1f}:1)")


# =========================
# 8. CORRELATION CHECK — phát hiện feature quá tương quan
# =========================
print("\n🔬 Kiểm tra correlation giữa các features được chọn...")

X_selected = df_clean[selected_features]
corr_matrix = X_selected.corr().abs()

# Tìm cặp có correlation > 0.85
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        val = corr_matrix.iloc[i, j]
        if val > 0.85:
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                round(val, 3)
            ))

if high_corr_pairs:
    print(f"⚠️ {len(high_corr_pairs)} cặp feature có |corr| > 0.85 (nên loại 1 trong mỗi cặp):")
    for f1, f2, c in high_corr_pairs:
        # Gợi ý loại feature nào (giữ cái có gini cao hơn)
        g1 = gini_df.loc[gini_df['feature'] == f1, 'gini'].values[0]
        g2 = gini_df.loc[gini_df['feature'] == f2, 'gini'].values[0]
        keep   = f1 if g1 >= g2 else f2
        remove = f2 if g1 >= g2 else f1
        print(f"   {f1} ↔ {f2}  corr={c}  → giữ [{keep}], cân nhắc bỏ [{remove}]")
else:
    print("✅ Không có cặp nào tương quan quá cao")

# Vẽ correlation heatmap
plt.figure(figsize=(max(8, len(selected_features)), max(6, len(selected_features) - 2)))
sns_data = X_selected.corr()
mask = np.triu(np.ones_like(sns_data, dtype=bool))
import seaborn as sns
sns.heatmap(sns_data, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Matrix — Selected Features', fontweight='bold')
plt.tight_layout()
plt.savefig('feature_correlation_improved.png', dpi=150, bbox_inches='tight')
print("📊 Heatmap đã lưu: feature_correlation_improved.png")
plt.show()


# =========================
# 9. LƯU FILE
# =========================
print(f"\n📁 Dataset shape cuối: {df_clean.shape}")

output_filename = "elephant_features_selected_improved.csv"
df_clean.to_csv(output_filename, index=False)
print(f"🎉 Đã lưu: {output_filename}")

# Lưu danh sách feature đã chọn để dùng lại
pd.Series(selected_features).to_csv('selected_feature_names.csv', index=False, header=False)
print(f"🎉 Đã lưu danh sách features: selected_feature_names.csv")

print("\n5 dòng đầu:")
print(df_clean.head())

print("\n📋 Thống kê mô tả:")
print(df_clean.describe().round(4))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# =========================
# HÀM TIỆN ÍCH
# =========================
def remove_timestamp_columns(df, label_col="is_outside"):
    """
    Loại bỏ các cột timestamp/id không cần thiết khỏi DataFrame.

    Args:
        df: DataFrame cần xử lý
        label_col: Tên cột label cần giữ lại (mặc định: "is_outside")

    Returns:
        DataFrame đã loại bỏ các cột timestamp
    """
    # Danh sách các cột cần loại bỏ
    cols_to_drop = []

    for col in df.columns:
        # Bỏ qua cột label
        if col == label_col:
            continue

        # Kiểm tra pattern timestamp
        if col.lower() in ['timestamp', 'timestep', 'time', 'date', 'datetime']:
            cols_to_drop.append(col)
        # Kiểm tra pattern id (nhưng phải match exactly hoặc gần exactly)
        elif col.lower() in ['id', 'idx', 'record_id', 'sample_id', 'row_id']:
            cols_to_drop.append(col)

    if cols_to_drop:
        print(f"   Loại bỏ các cột timestamp/id: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    return df

# =========================
# 1. ĐỌC VÀ CHUẨN BỊ DỮ LIỆU
# =========================
print("=" * 60)
print("PHÂN TÍCH: SỐ LƯỢNG FEATURES VS ĐỘ CHÍNH XÁC")
print("=" * 60)

df = pd.read_csv('elephant_features_selected_improved.csv')

# Loại bỏ timestamp
df = remove_timestamp_columns(df)

LABEL_COL = "is_outside"
X_all = df.drop(columns=[LABEL_COL])
y = df[LABEL_COL]

print(f"\nTổng số features có sẵn: {len(X_all.columns)}")
print(f"Tổng số mẫu: {len(X_all)}")
print(f"Phân phối nhãn: {y.value_counts().to_dict()}")

# =========================
# 2. LẤY GINI IMPORTANCE ĐỂ RANK FEATURES
# =========================
print("\n🌲 Tính Gini Importance để rank features...")

rf_full = RandomForestClassifier(
    n_estimators=12,
    max_depth=6,
    min_samples_leaf=2,
    criterion='gini',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf_full.fit(X_all, y)

gini_df = pd.DataFrame({
    'feature': X_all.columns,
    'gini': rf_full.feature_importances_,
}).sort_values('gini', ascending=False).reset_index(drop=True)

print("\nTop 10 features theo Gini Importance:")
print(gini_df.head(10).to_string(index=False))

# =========================
# 3. PHÂN TÍCH VỚI SỐ LƯỢNG FEATURES KHÁC NHAU
# =========================
print("\n" + "=" * 60)
print("TRAINING VỚI CÁC SỐ LƯỢNG FEATURES KHÁC NHAU")
print("=" * 60)

# Cấu hình số lượng features cần test
feature_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, len(X_all.columns)]

results = []

for n_features in feature_counts:
    # Chọn top N features theo Gini importance
    selected_features = gini_df.head(n_features)['feature'].tolist()

    X = df[selected_features + [LABEL_COL]]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(columns=[LABEL_COL]),
        X[LABEL_COL],
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Train model
    rf = RandomForestClassifier(
        n_estimators=12,
        max_depth=6,
        min_samples_leaf=1,
        criterion='gini',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
        oob_score=True
    )
    rf.fit(X_train, y_train)

    # Predict với threshold mặc định 0.5
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    # Metrics với threshold 0.5
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_class1 = f1_score(y_test, y_pred)  # F1 cho lớp 1 (Outside)
    roc_auc = roc_auc_score(y_test, y_prob)
    oob_score = rf.oob_score_

    results.append({
        'n_features': n_features,
        'features': selected_features,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_class1': f1_class1,
        'roc_auc': roc_auc,
        'oob_score': oob_score
    })

    print(f"\n{n_features} features:")
    print(f"   Features: {selected_features}")
    print(f"   F1 Macro: {f1_macro:.4f} | F1 Weighted: {f1_weighted:.4f} | F1 Class1: {f1_class1:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f} | OOB Score: {oob_score:.4f}")

# =========================
# 4. TẠO DATAFRAME KẾT QUẢ
# =========================
results_df = pd.DataFrame(results)

# =========================
# 5. VẼ BIỂU ĐỒ
# =========================
print("\n" + "=" * 60)
print("VẼ BIỂU ĐỒ SO SÁNH")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: F1 Score vs Number of Features
ax1 = axes[0, 0]
ax1.plot(results_df['n_features'], results_df['f1_macro'], 'bo-', linewidth=2, markersize=8, label='F1 Macro')
ax1.plot(results_df['n_features'], results_df['f1_weighted'], 'rs-', linewidth=2, markersize=8, label='F1 Weighted')
ax1.plot(results_df['n_features'], results_df['f1_class1'], 'g^-', linewidth=2, markersize=8, label='F1 Class 1 (Outside)')

# Find max point
max_idx = results_df['f1_macro'].idxmax()
ax1.scatter(results_df.loc[max_idx, 'n_features'], results_df.loc[max_idx, 'f1_macro'],
           s=200, c='red', marker='*', zorder=5, label=f'Max F1 Macro: {results_df.loc[max_idx, "f1_macro"]:.4f}')

ax1.set_xlabel('Number of Features', fontsize=12)
ax1.set_ylabel('F1 Score', fontsize=12)
ax1.set_title('F1 Score vs Number of Features', fontsize=14, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(results_df['n_features'])

# Plot 2: ROC-AUC vs Number of Features
ax2 = axes[0, 1]
ax2.plot(results_df['n_features'], results_df['roc_auc'], 'mo-', linewidth=2, markersize=8)

# Find max point
max_roc_idx = results_df['roc_auc'].idxmax()
ax2.scatter(results_df.loc[max_roc_idx, 'n_features'], results_df.loc[max_roc_idx, 'roc_auc'],
           s=200, c='red', marker='*', zorder=5, label=f'Max ROC-AUC: {results_df.loc[max_roc_idx, "roc_auc"]:.4f}')

ax2.set_xlabel('Number of Features', fontsize=12)
ax2.set_ylabel('ROC-AUC Score', fontsize=12)
ax2.set_title('ROC-AUC Score vs Number of Features', fontsize=14, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(results_df['n_features'])

# Plot 3: OOB Score vs Number of Features
ax3 = axes[1, 0]
ax3.plot(results_df['n_features'], results_df['oob_score'], 'cd-', linewidth=2, markersize=8)

# Find max point
max_oob_idx = results_df['oob_score'].idxmax()
ax3.scatter(results_df.loc[max_oob_idx, 'n_features'], results_df.loc[max_oob_idx, 'oob_score'],
           s=200, c='red', marker='*', zorder=5, label=f'Max OOB: {results_df.loc[max_oob_idx, "oob_score"]:.4f}')

ax3.set_xlabel('Number of Features', fontsize=12)
ax3.set_ylabel('OOB Score', fontsize=12)
ax3.set_title('Out-of-Bag Score vs Number of Features', fontsize=14, fontweight='bold')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(results_df['n_features'])

# Plot 4: F1 Class 1 (Outside) vs Number of Features
ax4 = axes[1, 1]
ax4.plot(results_df['n_features'], results_df['f1_class1'], 'yp-', linewidth=2, markersize=8)

# Find max point
max_f1_idx = results_df['f1_class1'].idxmax()
ax4.scatter(results_df.loc[max_f1_idx, 'n_features'], results_df.loc[max_f1_idx, 'f1_class1'],
           s=200, c='red', marker='*', zorder=5, label=f'Max F1 Class1: {results_df.loc[max_f1_idx, "f1_class1"]:.4f}')

ax4.set_xlabel('Number of Features', fontsize=12)
ax4.set_ylabel('F1 Score (Class 1 - Outside)', fontsize=12)
ax4.set_title('F1 Score for Outside Class vs Number of Features', fontsize=14, fontweight='bold')
ax4.legend(loc='best')
ax4.grid(True, alpha=0.3)
ax4.set_xticks(results_df['n_features'])

plt.suptitle('PHÂN TÍCH: ẢNH HƯỞNG CỦA SỐ LƯỢNG FEATURES ĐẾN ĐỘ CHÍNH XÁC',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('feature_count_analysis.png', dpi=150, bbox_inches='tight')
print("\n📊 Biểu đồ đã lưu: feature_count_analysis.png")
plt.show()

# =========================
# 6. TÓM TẮT KẾT QUẢ
# =========================
print("\n" + "=" * 60)
print("TÓM TẮT KẾT QUẢ")
print("=" * 60)

# Tìm số lượng features tối ưu theo F1 Macro
best_row = results_df.loc[results_df['f1_macro'].idxmax()]
print(f"\n✅ SỐ LƯỢNG FEATURES TỐI ƯU (theo F1 Macro): {int(best_row['n_features'])}")
print(f"   F1 Macro cao nhất: {best_row['f1_macro']:.4f}")
print(f"   F1 Weighted: {best_row['f1_weighted']:.4f}")
print(f"   F1 Class 1 (Outside): {best_row['f1_class1']:.4f}")
print(f"   ROC-AUC: {best_row['roc_auc']:.4f}")
print(f"   OOB Score: {best_row['oob_score']:.4f}")
print(f"   Features: {best_row['features']}")

# Tìm số lượng features tối ưu theo ROC-AUC
best_roc_row = results_df.loc[results_df['roc_auc'].idxmax()]
print(f"\n✅ SỐ LƯỢNG FEATURES TỐI ƯU (theo ROC-AUC): {int(best_roc_row['n_features'])}")
print(f"   ROC-AUC cao nhất: {best_roc_row['roc_auc']:.4f}")
print(f"   F1 Macro: {best_roc_row['f1_macro']:.4f}")
print(f"   Features: {best_roc_row['features']}")

# Tìm số lượng features tối ưu theo F1 Class 1 (Outside)
best_f1c1_row = results_df.loc[results_df['f1_class1'].idxmax()]
print(f"\n✅ SỐ LƯỢNG FEATURES TỐI ƯU (theo F1 Class 1 - Outside): {int(best_f1c1_row['n_features'])}")
print(f"   F1 Class 1 cao nhất: {best_f1c1_row['f1_class1']:.4f}")
print(f"   F1 Macro: {best_f1c1_row['f1_macro']:.4f}")
print(f"   Features: {best_f1c1_row['features']}")

# So sánh với tất cả features
all_features_row = results_df.loc[results_df['n_features'] == len(X_all.columns)].iloc[0]
print(f"\n📊 TẤT CẢ FEATURES ({len(X_all.columns)} features):")
print(f"   F1 Macro: {all_features_row['f1_macro']:.4f}")
print(f"   F1 Class 1: {all_features_row['f1_class1']:.4f}")
print(f"   ROC-AUC: {all_features_row['roc_auc']:.4f}")
print(f"   Features: {all_features_row['features']}")

# =========================
# 7. LƯU KẾT QUẢ RA FILE
# =========================
results_df.to_csv('feature_count_analysis_results.csv', index=False)
print(f"\n📁 Kết quả đã lưu: feature_count_analysis_results.csv")

# Lưu top features
gini_df.to_csv('feature_gini_ranking.csv', index=False)
print(f"📁 Ranking features đã lưu: feature_gini_ranking.csv")

print("\n" + "=" * 60)
print("PHÂN TÍCH HOÀN TẤT!")
print("=" * 60)

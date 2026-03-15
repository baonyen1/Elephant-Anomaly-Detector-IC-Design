import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# ====================== 1. LOAD DATA ======================
print("Đang load dữ liệu...")
df = pd.read_csv('Quantized_Features.csv')

# Tách X, y
X = df.drop(columns=['is_outside'])
y = df['is_outside']

# Split 80/20 (stratify vì imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train: {X_train.shape[0]} mẫu | Test: {X_test.shape[0]} mẫu")

# ====================== 2. GRID SEARCH ======================
results = []

print("Đang chạy bảng so sánh n_estimators & max_depth...\n")

for depth in range(1, 7):                    # depth từ 1 → 6 (bước 1)
    for n_est in range(2, 13, 2):            # n_estimators từ 2 → 12 (bước 2)
        
        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=depth,
            min_samples_leaf=1,
            class_weight='balanced',   # rất quan trọng vì imbalance
            random_state=42,
            n_jobs=-1,
            oob_score=True
        )
        
        rf.fit(X_train, y_train)
        
        # Dự đoán trên Test
        y_pred = rf.predict(X_test)
        
        # Tính metric
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='macro')
        prec = precision_score(y_test, y_pred, pos_label=1)
        rec  = recall_score(y_test, y_pred, pos_label=1)
        oob  = rf.oob_score_
        
        results.append({
            'max_depth': depth,
            'n_estimators': n_est,
            'OOB_Score': round(oob, 4),
            'Test_Accuracy': round(acc, 4),
            'F1_Macro': round(f1, 4),
            'Precision_Anomaly': round(prec, 4),
            'Recall_Anomaly': round(rec, 4)
        })
        
        print(f"depth={depth:2d} | n_est={n_est:2d} → OOB={oob:.4f} | F1={f1:.4f}")

# ====================== 3. TẠO BẢNG SO SÁNH ======================
df_compare = pd.DataFrame(results)

# Sắp xếp theo F1_Macro giảm dần (quan trọng nhất cho anomaly detection)
df_compare = df_compare.sort_values(by='F1_Macro', ascending=False)

print("\n" + "="*100)
print("BẢNG SO SÁNH n_estimators & max_depth (đã sắp xếp theo F1_Macro)")
print("="*100)
print(df_compare.to_string(index=False))

# Lưu ra file để đưa vào báo cáo Vivado
df_compare.to_csv('RF_Comparison_Depth1-6_nEst2-12.csv', index=False)
print("\nĐã lưu bảng vào: RF_Comparison_Depth1-6_nEst2-12.csv")
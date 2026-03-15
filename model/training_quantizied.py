import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from importlib import reload

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

print("Training Quantized Model")
df = pd.read_csv('Quantized_Features.csv')

X = df.drop(columns=['is_outside'])
y = df['is_outside']

#2 training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# ko su dung lai SMOTE/UnderSampler vi du lieu da duoc luong tu hoa truoc do
# ko su dung gridsearch vi da chot dc feature va hyperparameter tot nhat truoc do
rf_clf = RandomForestClassifier(
    n_estimators=12, #so cay la 80
    max_depth=6, # do sau la 6
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',# su dung class_weight de xu ly du lieu khong can bang
    criterion='gini', # su dung gini de tang toc do training
    oob_score=True
)
rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)
y_prob = rf_clf.predict_proba(X_test)[:, 1]
BEST_THRESHOLD = 0.73
print("Using fixed threshold:", BEST_THRESHOLD)

y_pred_opt = (y_prob >= BEST_THRESHOLD).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_pred_opt))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_opt)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

f1 = f1_score(y_test, y_pred_opt)
roc_auc = roc_auc_score(y_test, y_pred_opt)
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

import numpy as np
from sklearn.tree import _tree
import os

def export_tree_to_verilog_hex(tree, feature_names, tree_idx):
    tree_ = tree.tree_
    
    def recurse(node, depth):
        indent = "    " * (depth + 1)
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            
            # Chuyển đổi sang số nguyên và định dạng Hex 4 ký tự (16-bit)
            val_int = int(round(threshold))
            # Đảm bảo xử lý số âm nếu có bằng cách dùng bitwise & 0xFFFF
            hex_val = "{:04X}".format(val_int & 0xFFFF)

            code = f"{indent}if ({name} <= 16'h{hex_val}) begin\n"
            code += recurse(tree_.children_left[node], depth + 1)
            code += f"{indent}end else begin\n"
            code += recurse(tree_.children_right[node], depth + 1)
            code += f"{indent}end\n"
            return code
        else:
            # Kết quả đầu ra của cây: 1'b1 là bất thường, 1'b0 là bình thường
            res = "1'b1" if np.argmax(tree_.value[node]) == 1 else "1'b0"
            return f"{indent}tree_out = {res};\n"

    # Header module
    header = f"module decision_tree_{tree_idx} (\n"
    header += "    input wire [15:0] " + ", ".join(feature_names) + ",\n"
    header += "    output reg tree_out\n);\n\n"
    header += ""
    header += "always @(*) begin\n"
    
    body = recurse(0, 0)
    
    footer = "end\nendmodule\n"
    return header + body + footer

# Lấy danh sách đặc trưng từ X (mean_speed, kde_prob_min, v.v.)
feature_list = list(X.columns)

# Tạo thư mục lưu trữ nếu chưa có
if not os.path.exists('verilog_trees'):
    os.makedirs('verilog_trees')

print("--- ĐANG TRÍCH XUẤT LOGIC HEX ---")
for i, estimator in enumerate(rf_clf.estimators_):
    v_code = export_tree_to_verilog_hex(estimator, feature_list, i+1)
    file_path = f"verilog_trees/decision_tree_{i+1}.v"
    with open(file_path, 'w') as f:
        f.write(v_code)
    print(f"Đã tạo: {file_path}")

print("\nThành công! Hãy copy nội dung trong thư mục 'verilog_trees' vào Vivado.")
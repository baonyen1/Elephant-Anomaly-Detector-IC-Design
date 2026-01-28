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
df = pd.read_csv('Quantized_Combined_Features.csv')

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
    n_estimators=80, #so cay la 80
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
BEST_THRESHOLD = 0.7
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

# save model
reload(joblib)
class ElephantAnomalyDetector:
    def __init__(self, model, threshold, feature_names):
        self.model = model
        self.threshold = threshold
        self.feature_names = feature_names
    def predict(self, X):
        X = X[self.feature_names]
        prob = self.model.predict_proba(X)[:, 1]
        return (prob >= self.threshold).astype(int)

    def predict_proba(self, X):
        X = X[self.feature_names]
        return self.model.predict_proba(X)
final_model_package = ElephantAnomalyDetector(
    model=rf_clf,
    threshold=BEST_THRESHOLD,
    feature_names=list(X.columns)
)
print('Xuat file model')
filename = 'quantization_rf_model.pkl'
joblib.dump(final_model_package, filename)
print('Xuat file model thanh cong')
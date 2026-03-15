# Elephant Behavior Classification System

## 🎯 Mục đích dự án
Hệ thống phân loại hành vi voi dựa trên dữ liệu GPS tracking, sử dụng Machine Learning để phát hiện khi voi di chuyển ra ngoài home range (Outside) hoặc ở trong home range (Inside).

## 📊 Tổng quan
- **Dataset**: Dữ liệu GPS từ elephant collar tracking (Collar 1630 - Ivory Coast)
- **Problem**: Binary classification với highly imbalanced data
- **Approach**: Random Forest với **Weighted Anomaly Score** features và **Quantization** cho hardware deployment
- **Pipeline**: Filter → Feature Selection → Quantization → Training → Verilog Export

## 🏗️ Kiến trúc hệ thống

### Data Pipeline (CURRENT FLOW)
```
Raw GPS Data → Feature Engineering → Feature Selection → Quantization → Training → Verilog Export
     ↓              ↓                      ↓                ↓            ↓          ↓
  .csv files   filter.py          training.py       quantization.py  training   Vivado
               (6 anomaly                                              quantizied.py
               features +
               turning angle)
```

### Feature Engineering Pipeline (filter.py)
```
1. Adaptive KDE (Silverman bandwidth + IQR outlier filtering)
2. Persistence Anomaly (rolling 2h window)
3. Turning Angle (filtered when stationary - dist < 50m)
4. Behavioral Anomaly (AND logic + percentile thresholds)
5. Temporal Anomaly (night-specific thresholds + KDE low)
6. Stationary Anomaly (NEW - phát hiện đứng yên bất thường)
7. Acceleration Anomaly (NEW - tín hiệu gia tốc đột ngột)
8. Weighted Score (thay vì OR logic)
```

### Feature Selection (training.py)
```python
feature_groups = {
    'kde': ['kde_prob_mean', 'kde_prob_std', ...],
    'centroid': ['dist_to_centroid_mean'],
    'step': ['step_mean', 'step_std', ...],
    'speed': ['mean_speed', 'accelerate', ...],
    'turning': ['turning_angle_mean', 'turning_entropy', ...],
    'stationary': ['stationary_ratio', 'stationary_streak_max', ...],
    'time': ['hour', 'is_night']
}
# Selected via Gini Importance → ~12 features
```

### Quantization Pipeline (quantization.py)
```python
# Label Encoding cho categorical columns
# uint32 quantization cho numeric features
# Export scale table để dequantize
```

### Model Training (training_quantizied.py)
```python
RandomForestClassifier(
    n_estimators=12,  # 80 cây
    max_depth=6,
    class_weight='balanced',
    criterion='gini',
    oob_score=True
)
threshold_optimization → Verilog export
```

## 🔑 Key Concepts

### 1. Weighted Anomaly Score (CẢI TIẾN)
**OLD**: OR logic (spatial OR persistence OR behavioral OR temporal)
**NEW**: Weighted sum với các weights:
```python
weights = {
    'spatial_anomaly':      0.20,
    'persistence_anomaly':  0.20,
    'behavioral_anomaly':   0.20,
    'temporal_anomaly':     0.15,
    'acceleration_anomaly': 0.10,
    'stationary_anomaly':   0.15,
}
SCORE_THRESHOLD = 0.25  # cần ~2 anomalies đồng ý
```

### 2. Adaptive KDE (CẢI TIẾN)
```python
# Silverman bandwidth tự động
bw = 1.06 * std_mean * (n ** (-1/5))

# IQR outlier filtering trước khi fit KDE
# Percentile normalization thay vì min-max
prob_normalized = np.clip((prob - p1) / (p99 - p1), 0, 1)
```

### 3. Behavioral Anomaly - AND Logic (CẢI TIẾN)
**OLD**: `(speed > TH) | (turning > TH)` - quá nhạy
**NEW**: `(speed > TH) & (turning_clean > TH)` - panic behavior

### 4. Turning Angle Cleaning
```python
# Lọc GPS noise khi voi đứng yên
turning_angle_clean = np.where(dist < 50m, 0, turning_angle)
```

### 5. Quantization cho Hardware
```python
# uint32 quantization
quantized = ((value - min) / scale).round().astype('uint32')
scale = (max - min) / (2^32 - 1)

# Export Verilog decision trees
# Hex threshold values cho FPGA
```

## 📁 Cấu trúc dữ liệu

### Input Files
- `Elephant Research - Ivory Coast - Collar 1630.csv`: Raw GPS data
  - Columns: timestamp, location-lat, location-long, ...

### Output Files (filter.py)
- `elephant_features_improved.csv`: Engineered features (2h intervals)
- `elephant_raw_improved.csv`: Raw data với tất cả anomaly flags

### Output Files (training.py)
- `elephant_features_selected_improved.csv`: Selected features
- `selected_feature_names.csv`: Feature list

### Output Files (quantization.py)
- `Quantized_Features.csv`: Quantized features cho training
- `Quantization_Scales.csv`: Scale table để dequantize
- `label_encoding_mapping.json/csv`: Label encoder mapping

### Output Files (training_quantizied.py)
- `model.pkl`: Trained model với optimal threshold
- `verilog_trees/`: Decision trees dưới dạng Verilog modules

## 🛠️ Tech Stack
- **Python 3.8+**
- **Core Libraries**:
  - pandas, numpy: Data manipulation
  - scikit-learn: ML models, metrics
  - geopy: Geographic distance calculations
  - matplotlib, seaborn: Visualization
- **ML Algorithms**:
  - Random Forest Classifier
  - SMOTE + RandomUnderSampler cho imbalance
  - GridSearchCV cho hyperparameter tuning
- **Hardware**:
  - Verilog export cho FPGA implementation

## 🚀 Quick Start

### 1. Feature Engineering
```bash
cd software/data
python filter.py
# Output: elephant_features_improved.csv
```

### 2. Feature Selection
```bash
cd software/data
python training.py
# Output: elephant_features_selected_improved.csv
```

### 3. Quantization
```bash
cd software/quantization
python quantization.py
# Output: Quantized_Features.csv
```

### 4. Model Training + Verilog Export
```bash
cd software/model
python training_quantizied.py
# Output: model.pkl + verilog_trees/*.v
```

## ⚙️ Configuration

### Feature Engineering Parameters (filter.py)
```python
KDE: Silverman bandwidth (auto) + IQR outlier filtering
DBSCAN: eps=0.005, min_samples=10
Turning Angle: lọc khi dist < 50m
Behavioral: AND logic + 97th/95th percentile
Temporal: 95th percentile night speed + KDE < 0.3
Stationary: streak >= 3 + KDE < 0.4
Acceleration: 97th percentile
Score Threshold: 0.25
```

### Model Hyperparameters (training_quantizied.py)
```python
n_estimators = 12  # số cây
max_depth = 6
min_samples_leaf = [1, 2, 5] - tuned via GridSearch
criterion = 'gini'
class_weight = 'balanced'
sampling_strategy SMOTE = 0.5
sampling_strategy Under = 0.8
```

### Threshold Optimization
```python
# Tìm threshold tối ưu cho F1-score
thresholds = np.arange(0.01, 1.0, 0.01)
best_threshold = threshold[argmax(f1_scores)]
# Typical: 0.70-0.75
```

## 📝 Notes for Claude

### When working on this project:
1. **Follow the flow**: filter → training (selection) → quantization → training_quantizied
2. **Weighted Score > OR logic**: Understand the weight contributions
3. **Quantization matters**: uint32 scaling cho hardware deployment
4. **Turning angle clean**: Đã lọc GPS noise khi đứng yên
5. **Verilog export**: Hex thresholds cho FPGA implementation

### Best Practices
- Always check feature correlations before selection
- Validate anomaly score distribution
- Test threshold sensitivity
- Verify quantization scales match original ranges
- Check Verilog threshold values match training

## 🎯 Future Improvements

### Short-term
- [ ] Multi-collar training
- [ ] Real-time prediction pipeline
- [ ] Threshold adaptive theo context

### Long-term
- [ ] FPGA deployment verification
- [ ] Multi-class classification (feeding, resting, traveling)
- [ ] Online learning cho new collars

---

**Last Updated**: 2026-03-07
**Version**: 2.0.0
**Status**: Production Ready ✅ với Hardware Deployment
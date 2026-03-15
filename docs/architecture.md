# System Architecture

## 🏗️ Overview

Elephant Behavior Classification System được thiết kế theo kiến trúc pipeline 4 giai đoạn: **Feature Engineering → Feature Selection → Quantization → Model Training**, với output là model.pkl và Verilog code cho FPGA deployment.

## 📊 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│  Raw GPS Data (.csv) → timestamp, lat, long                    │
│  File: Elephant Research - Ivory Coast - Collar 1630.csv       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 1: FEATURE ENGINEERING (filter.py)           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   KDE        │→ │   Anomaly    │→ │   Weighted   │         │
│  │  Adaptive    │  │   Features   │  │    Score     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
│  Output: elephant_features_improved.csv (~50 features)         │
│          elephant_raw_improved.csv (anomaly flags)             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            STAGE 2: FEATURE SELECTION (training.py)             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Gini       │→ │   Group      │→ │   Correlation│         │
│  │ Importance   │  │   Selection  │  │   Check      │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
│  Output: elephant_features_selected_improved.csv (~12 features)│
│          selected_feature_names.csv                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 3: QUANTIZATION (quantization.py)            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Label      │→ │   uint32     │→ │   Scale      │         │
│  │  Encoding    │  │ Quantization │  │   Export     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
│  Output: Quantized_Features.csv                                │
│          Quantization_Scales.csv                               │
│          label_encoding_mapping.json/csv                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           STAGE 4: MODEL TRAINING (training_quantizied.py)      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   SMOTE +    │→ │  GridSearch  │→ │  Threshold   │         │
│  │  Undersample │  │     CV       │  │  Optimizer  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                         │                                        │
│                         ▼                                        │
│              ┌──────────────────────┐                           │
│              │   Verilog Export     │                           │
│              │   (decision trees)   │                           │
│              └──────────────────────┘                           │
│                                                                 │
│  Output: model.pkl (Random Forest + threshold)                 │
│          verilog_trees/*.v (12 decision tree modules)          │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DEPLOYMENT LAYER                            │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │   Python     │  │   FPGA       │                            │
│  │   Inference  │  │   (Vivado)   │                            │
│  └──────────────┘  └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Component Details

### Stage 1: Feature Engineering (filter.py)

**Responsibility**: Tạo anomaly features từ raw GPS data

**Input**:
```python
{
    "filepath": "Elephant Research - Ivory Coast - Collar 1630.csv",
    "columns": ["timestamp", "location-lat", "location-long", ...]
}
```

**Processing**:
1. **Adaptive KDE**:
   - IQR outlier filtering trước khi fit
   - Silverman bandwidth tự động
   - Percentile normalization

2. **Six Anomaly Types**:
   ```python
   weights = {
       'spatial_anomaly':      0.20,  # KDE-based
       'persistence_anomaly':  0.20,  # Rolling 2h window
       'behavioral_anomaly':   0.20,  # AND logic (speed & turning)
       'temporal_anomaly':     0.15,  # Night-specific
       'acceleration_anomaly': 0.10,  # Gia tốc đột ngột
       'stationary_anomaly':   0.15,  # Đứng yên bất thường
   }
   ```

3. **Weighted Score**:
   ```python
   anomaly_score = sum(anomaly_i * weight_i)
   point_is_outside = (anomaly_score >= 0.25).astype(int)
   ```

4. **Feature Engineering** (2h intervals):
   - Step length statistics
   - Turning angle (cleaned)
   - KDE probabilities
   - Rolling variance
   - Temporal features

**Output**:
```python
{
    "elephant_features_improved.csv": DataFrame with ~50 features,
    "elephant_raw_improved.csv": Raw data + anomaly flags,
    "elephant_anomaly_improved.png": Visualization
}
```

### Stage 2: Feature Selection (training.py)

**Responsibility**: Chọn features tốt nhất qua Gini Importance

**Input**: `elephant_features_improved.csv`

**Processing**:
1. **Remove Leakage Columns**:
   ```python
   drop_cols = [
       'point_is_outside_old_OR',  # Debug
       'anomaly_score_2h',         # Trùng lặp
       'anomaly_score_mean',       # LEAKAGE
       'anomaly_score_max',        # LEAKAGE
   ]
   ```

2. **Feature Groups**:
   ```python
   feature_groups = {
       'kde': ['kde_prob_mean', 'kde_prob_std', ...],
       'centroid': ['dist_to_centroid_mean'],
       'step': ['step_mean', 'step_std', ...],
       'speed': ['mean_speed', 'accelerate', ...],
       'turning': ['turning_angle_mean', 'turning_entropy', ...],
       'stationary': ['stationary_ratio', 'stationary_streak_max', ...],
       'time': ['hour', 'is_night'],
   }
   ```

3. **Gini Importance**:
   ```python
   rf = RandomForestClassifier(n_estimators=300, ...)
   rf.fit(X_all, y)
   gini_df = pd.DataFrame({
       'feature': X_all.columns,
       'gini': rf.feature_importances_
   }).sort_values('gini', ascending=False)
   ```

4. **Group-wise Selection**:
   ```python
   N_FEATURES_PER_GROUP = {
       'kde': 2,
       'centroid': 1,
       'step': 1,
       'speed': 2,
       'turning': 2,
       'stationary': 1,
       'time': 1,
   }
   ```

5. **Correlation Check**:
   ```python
   corr_matrix = X_selected.corr().abs()
   # Remove pairs with |corr| > 0.85
   ```

6. **Garbage Data Filtering**:
   ```python
   mask_garbage = (dist_to_centroid < 0.1) & (mean_speed == 0)
   df_clean = df_selected.loc[~mask_garbage]
   ```

**Output**:
```python
{
    "elephant_features_selected_improved.csv": DataFrame (~12 features),
    "selected_feature_names.csv": List of selected features,
    "gini_by_group_improved.png": Feature importance plot,
    "feature_correlation_improved.png": Correlation heatmap
}
```

### Stage 3: Quantization (quantization.py)

**Responsibility**: Lượng tử hóa features cho hardware deployment

**Input**: `elephant_features_selected_improved.csv`

**Processing**:
1. **Label Encoding**:
   ```python
   for col in df.select_dtypes(include='object').columns:
       le = LabelEncoder()
       df[col] = le.fit_transform(df[col])
       # Export mapping to JSON
   ```

2. **uint32 Quantization**:
   ```python
   def quantize_column_to_uint32(series):
       min_val = series.min()
       max_val = series.max()
       scale = (max_val - min_val) / (2**32 - 1)
       quantized = ((series - min_val) / scale).round().astype('uint32')
       return quantized, scale, min_val, max_val
   ```

3. **Scale Table Export**:
   ```python
   scale_table.append({
       "feature": col,
       "scale": scale,
       "min": min_val,
       "max": max_val
   })
   ```

**Output**:
```python
{
    "Quantized_Features.csv": DataFrame with uint32 columns,
    "Quantization_Scales.csv": Scale/min/max per feature,
    "label_encoding_mapping.json": Label encoder mapping,
    "label_encoding_mapping.csv": Label encoder CSV
}
```

### Stage 4: Model Training (training_quantizied.py)

**Responsibility**: Train Random Forest + Export Verilog

**Input**: `Quantized_Features.csv`

**Processing**:
1. **Train/Test Split**:
   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, stratify=y, random_state=42
   )
   ```

2. **SMOTE + Undersampling**:
   ```python
   pipeline = ImbPipeline([
       ('smote', SMOTE(sampling_strategy=0.5)),
       ('under', RandomUnderSampler(sampling_strategy=0.8)),
       ('rf', RandomForestClassifier(
           n_estimators=12,
           max_depth=6,
           class_weight='balanced',
           criterion='gini',
           oob_score=True
       ))
   ])
   ```

3. **GridSearchCV**:
   ```python
   param_grid = {
       'rf__n_estimators': [12],
       'rf__max_depth': [6],
       'rf__min_samples_leaf': [1, 2, 5]
   }
   grid_search = GridSearchCV(..., scoring='f1_macro', cv=5)
   ```

4. **Threshold Optimization**:
   ```python
   thresholds = np.arange(0.01, 1.0, 0.01)
   f1_scores = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]
   best_threshold = thresholds[np.argmax(f1_scores)]
   # Typical: 0.70-0.75
   ```

5. **Verilog Export**:
   ```python
   def export_tree_to_verilog_hex(tree, feature_names, tree_idx):
       # Convert thresholds to 32-bit hex
       hex_val = "{:08X}".format(int(round(threshold)) & 0xFFFFFFFF)
       # Generate if-else Verilog code
       code = f"if ({name} <= 32'h{hex_val}) begin\n"
       ...
   ```

**Output**:
```python
{
    "model.pkl": ElephantAnomalyDetector(model, threshold),
    "verilog_trees/": [
        "decision_tree_1.v",
        "decision_tree_2.v",
        ...
        "decision_tree_12.v"
    ]
}
```

## 🔄 Data Flow

### Training Flow
```
Raw GPS CSV
    ↓
[filter.py] → elephant_features_improved.csv
    ↓
[training.py] → elephant_features_selected_improved.csv
    ↓
[quantization.py] → Quantized_Features.csv
    ↓
[training_quantizied.py] → model.pkl + verilog_trees/
    ↓
[FPGA Deployment] → Vivado project
```

### Inference Flow (Python)
```
New GPS Data
    ↓
[Apply same feature engineering]
    ↓
[Quantize using scale table]
    ↓
[Load model.pkl]
    ↓
[Predict with threshold]
    ↓
[Outside/Inside label]
```

### Inference Flow (FPGA)
```
New GPS Data → Feature Engineering → Quantization
    ↓
[All 12 decision trees in parallel]
    ↓
[Voting: majority wins]
    ↓
[Outside/Inside label]
```

## 🗄️ Data Schemas

### Selected Features Schema
```python
{
    # KDE Features (2)
    "kde_prob_mean": uint32,
    "kde_prob_std": uint32,

    # Centroid (1)
    "dist_to_centroid_mean": uint32,

    # Step (1)
    "step_mean": uint32,

    # Speed (2)
    "mean_speed": uint32,
    "accelerate": uint32,

    # Turning (2)
    "turning_angle_mean": uint32,
    "turning_entropy": uint32,

    # Stationary (1)
    "stationary_ratio": uint32,

    # Time (1)
    "is_night": uint32,  # Already 0/1

    # Target
    "is_outside": int    # 0=Inside, 1=Outside
}
```

### Quantization Scale Table Schema
```python
{
    "feature": str,      # Feature name
    "scale": float,      # Scaling factor
    "min": float,        # Original minimum
    "max": float         # Original maximum
}
```

### Verilog Module Schema
```verilog
module decision_tree_N (
    input wire [31:0] feature1,
    input wire [31:0] feature2,
    // ... all features
    output reg tree_out  // 1'b1 = Outside, 1'b0 = Inside
);

always @(*) begin
    if (feature1 <= 32'hXXXXXXXX) begin
        // ... nested if-else
    end
end
endmodule
```

## 🔐 Design Decisions

### Decision 1: 4-Stage Pipeline
**Rationale**:
- Separation of concerns
- Each stage independently testable
- Quantization enables hardware deployment

### Decision 2: Gini-based Feature Selection
**Rationale**:
- Model-based importance
- Handles non-linear relationships
- Interpretable rankings

### Decision 3: uint32 Quantization
**Rationale**:
- Fits FPGA fixed-point arithmetic
- Preserves precision (32-bit)
- Reversible with scale table

### Decision 4: Verilog Export
**Rationale**:
- Direct FPGA deployment
- No dependency on Python/sklearn
- Low-latency inference

## 📈 Scalability Considerations

### Current Limitations
- Single collar processing
- In-memory processing
- No distributed training

### Scale-Up Path
1. **Multiple Collars**: Batch processing with shared scale table
2. **Streaming**: Real-time feature engineering pipeline
3. **Model Compression**: Prune trees for smaller FPGA

## 🔧 Technology Stack

### Core
- **Python 3.8+**: Main language
- **pandas**: Data manipulation
- **NumPy**: Numerical operations
- **scikit-learn**: ML algorithms

### Specialized
- **imblearn**: SMOTE + Undersampling
- **geopy**: Geographic distance
- **joblib**: Model persistence

### Visualization
- **matplotlib**: Base plotting
- **seaborn**: Statistical visualizations

### Hardware
- **Verilog**: FPGA implementation
- **Vivado**: Xilinx FPGA tools

## 📝 Future Architecture Improvements

### Short-Term
- [ ] Configuration file (YAML/JSON)
- [ ] Logging throughout pipeline
- [ ] Input validation decorators

### Medium-Term
- [ ] REST API for predictions
- [ ] Docker containerization
- [ ] CI/CD pipeline

### Long-Term
- [ ] Multi-collar ensemble
- [ ] Real-time streaming
- [ ] Online learning

---

**Document Version**: 2.0
**Last Updated**: 2026-03-07
**Author**: System Architect
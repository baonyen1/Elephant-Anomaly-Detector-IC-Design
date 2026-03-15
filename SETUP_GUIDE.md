# 🚀 Elephant Project Setup Guide

Hướng dẫn chi tiết để setup và chạy dự án Elephant Behavior Classification.

## 📋 Prerequisites

### System Requirements
- **OS**: Windows 10/11, macOS, or Linux
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Disk Space**: 5GB for code, data, and models

### Required Software
- Python 3.8+
- pip (Python package manager)
- Git

## 🔧 Installation Steps

### 1. Clone Repository
```bash
cd C:\Users\nguye\Documents\TKVM
# Or clone if starting fresh:
# git clone https://github.com/yourusername/elephant-behavior-classification.git
```

### 2. Create Virtual Environment
#### Using venv (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

#### Using conda
```bash
conda create -n elephant python=3.8
conda activate elephant
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Verify installation:
```bash
python -c "import pandas, sklearn, geopy, matplotlib, seaborn; print('✅ All packages installed')"
```

### 4. Verify Project Structure
```bash
# Check that these directories exist:
software/data/
software/quantization/
software/model/
src/
```

### 5. Download Data
Place your GPS data in the working directory:
```bash
# Required file:
# Elephant Research - Ivory Coast - Collar 1630.csv
```

### 6. Verify Setup
```bash
python src/config.py
```

Expected output:
```
==============================================================
ELEPHANT BEHAVIOR CLASSIFICATION - CONFIGURATION
==============================================================
📁 Paths: ...
📊 Data: ...
🔧 Features: ...
🤖 Model: ...
```

## 🏃 Quick Start - Full Pipeline

### Run All Steps in Order
```bash
# Step 1: Feature Engineering
cd software/data
python filter.py

# Step 2: Feature Selection
python training.py

# Step 3: Quantization
cd ../quantization
python quantization.py

# Step 4: Model Training + Verilog Export
cd ../model
python training_quantizied.py
```

### Expected Outputs
After full pipeline, you should have:
```
elephant_features_improved.csv       # From filter.py
elephant_raw_improved.csv            # From filter.py
elephant_features_selected_improved.csv  # From training.py
selected_feature_names.csv           # From training.py
Quantized_Features.csv               # From quantization.py
Quantization_Scales.csv              # From quantization.py
label_encoding_mapping.json          # From quantization.py
model.pkl                            # From training_quantizied.py
verilog_trees/                       # From training_quantizied.py
  ├── decision_tree_1.v
  ├── decision_tree_2.v
  └── ...
```

## 📂 Detailed Step-by-Step

### Step 1: Feature Engineering (filter.py)

**Purpose**: Tạo 6 anomaly features + weighted score từ raw GPS data

**Input**: `Elephant Research - Ivory Coast - Collar 1630.csv`

**Output**:
- `elephant_features_improved.csv` - Features cho ML
- `elephant_raw_improved.csv` - Raw data + anomaly flags
- `elephant_anomaly_improved.png` - Visualization

**What it does**:
1. Adaptive KDE với Silverman bandwidth
2. Tính 6 anomaly scores:
   - Spatial Anomaly (KDE-based)
   - Persistence Anomaly (rolling window)
   - Behavioral Anomaly (AND logic)
   - Temporal Anomaly (night-specific)
   - Acceleration Anomaly (gia tốc đột ngột)
   - Stationary Anomaly (đứng yên bất thường)
3. Weighted score combination
4. Feature engineering (2h intervals)

**Duration**: ~2-5 minutes

### Step 2: Feature Selection (training.py)

**Purpose**: Chọn features tốt nhất qua Gini Importance

**Input**: `elephant_features_improved.csv`

**Output**:
- `elephant_features_selected_improved.csv` - Selected features
- `selected_feature_names.csv` - Feature list
- `gini_by_group_improved.png` - Feature importance plot
- `feature_correlation_improved.png` - Correlation heatmap

**What it does**:
1. Train Random Forest để tính Gini importance
2. Chọn top features theo nhóm (kde, step, turning, etc.)
3. Kiểm tra correlation - loại features tương quan cao
4. Lọc data rác (đứng yên ở tâm)
5. Thống kê phân phối nhãn

**Duration**: ~1-2 minutes

### Step 3: Quantization (quantization.py)

**Purpose**: Lượng tử hóa features cho hardware deployment

**Input**: `elephant_features_selected_improved.csv`

**Output**:
- `Quantized_Features.csv` - Quantized data (uint32)
- `Quantization_Scales.csv` - Scale/min/max table
- `label_encoding_mapping.json` - Label encoder mapping
- `label_encoding_mapping.csv` - Label encoder CSV

**What it does**:
1. Label encoding cho categorical columns
2. uint32 quantization cho numeric features
3. Export scale table để dequantize
4. Save label mapping

**Duration**: ~30 seconds

### Step 4: Model Training (training_quantizied.py)

**Purpose**: Train Random Forest + Export Verilog

**Input**: `Quantized_Features.csv`

**Output**:
- `model.pkl` - Trained model với optimal threshold
- `verilog_trees/` - Decision trees dưới dạng Verilog

**What it does**:
1. Split train/test (80/20, stratified)
2. SMOTE + Undersampling cho imbalance
3. GridSearchCV tuning hyperparameters
4. Threshold optimization cho F1-score
5. Export decision trees sang Verilog (hex thresholds)

**Duration**: ~2-5 minutes

## 🔍 Common Issues & Solutions

### Issue 1: File Not Found
```
FileNotFoundError: 'Elephant Research - Ivory Coast - Collar 1630.csv'
```
**Solution**: Copy data file to current directory
```bash
cp /path/to/data.csv .
```

### Issue 2: Module Import Error
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solution**: Activate venv and install dependencies
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

### Issue 3: Output Files Missing
**Solution**: Run scripts in correct order:
1. `filter.py` first
2. `training.py` second
3. `quantization.py` third
4. `training_quantizied.py` fourth

### Issue 4: Low F1 Score
**Symptoms**: F1 < 0.90

**Solutions**:
- Check feature selection removed leakage columns
- Verify SMOTE/Undersampling ratios
- Re-run threshold optimization
- Check data quality (GPS drift, outliers)

### Issue 5: Verilog Trees Not Generated
**Solution**: Check that `verilog_trees/` directory is created
```bash
# Directory should be auto-created by script
ls verilog_trees/
```

## 📊 Expected Performance

### Step 1 (filter.py)
- Creates ~50 features
- 6 anomaly types
- Weighted score distribution

### Step 2 (training.py)
- Selects ~12 features
- Removes correlated features
- Balances class distribution

### Step 3 (quantization.py)
- All features quantized to uint32
- Scale factors saved

### Step 4 (training_quantizied.py)
- F1 Score: ~0.98
- ROC AUC: ~0.99
- Optimal threshold: 0.70-0.75
- 12 decision trees exported to Verilog

## ⚙️ Configuration

### Modify Parameters

**filter.py** - Feature engineering:
```python
# KDE bandwidth (Silverman auto)
# Behavioral thresholds (percentiles)
# Score weights
SCORE_THRESHOLD = 0.25
```

**training.py** - Feature selection:
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

**training_quantizied.py** - Model training:
```python
n_estimators = 12  # số cây
max_depth = 6
SMOTE sampling_strategy = 0.5
UnderSampler sampling_strategy = 0.8
```

## 🧪 Testing

### Test Individual Components
```bash
# Test feature engineering
cd software/data
python filter.py

# Test feature selection
python training.py

# Test quantization
cd ../quantization
python quantization.py

# Test model training
cd ../model
python training_quantizied.py
```

### Verify Model
```python
import joblib
import pandas as pd

# Load model
model = joblib.load('model.pkl')

# Load data
df = pd.read_csv('Quantized_Features.csv')
X = df.drop(columns=['is_outside'])

# Predict
probs = model.model.predict_proba(X)[:, 1]
preds = (probs >= model.threshold).astype(int)

# Check F1
from sklearn.metrics import f1_score
print(f"F1: {f1_score(df['is_outside'], preds):.4f}")
```

## 📚 Next Steps

1. ✅ **Verify Installation**: `pip install -r requirements.txt`
2. ✅ **Prepare Data**: Copy GPS data to directory
3. ✅ **Run Pipeline**: filter → training → quantization → training_quantizied
4. ✅ **Verify Outputs**: Check all CSV and Verilog files
5. ✅ **Test Model**: Load and test model.pkl
6. ✅ **Deploy to FPGA**: Copy verilog_trees/ to Vivado project

## 🆘 Getting Help

### Documentation
- **CLAUDE.md**: Main context file
- **README.md**: Project overview
- **docs/architecture.md**: System design
- **docs/decisions/**: Architecture decisions

### Debug Tips
```bash
# Check file exists
ls -la *.csv

# Check data quality
python -c "import pandas as pd; df = pd.read_csv('file.csv'); print(df.describe())"

# Check model
python -c "import joblib; m = joblib.load('model.pkl'); print(m.threshold)"
```

## ✅ Verification Checklist

Before declaring success:
- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] Data file in place
- [ ] `elephant_features_improved.csv` created
- [ ] `elephant_features_selected_improved.csv` created
- [ ] `Quantized_Features.csv` created
- [ ] `model.pkl` created
- [ ] `verilog_trees/*.v` files created
- [ ] F1 score > 0.95
- [ ] No critical errors in logs

## 🎉 Success Criteria

You're done when:
1. All 4 scripts run without errors
2. F1 score > 0.95 on test set
3. Verilog files generated
4. Model can load and predict

---

**Happy coding!** 🐘🎉
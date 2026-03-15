# 🐘 Elephant Behavior Classification System

> AI-powered system for detecting elephant movement patterns using GPS tracking data
> Now with **Quantization** and **Verilog Export** for FPGA deployment

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

## 📋 Quick Links

- [Overview](#overview)
- [Pipeline Flow](#pipeline-flow)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Results](#results)

## 🎯 Overview

This project implements a machine learning pipeline to classify elephant behavior based on GPS collar tracking data. The system detects when elephants move outside their home range - critical information for wildlife conservation and human-elephant conflict prevention.

**NEW in v2.0**:
- ✅ Quantization support for hardware deployment
- ✅ Verilog decision tree export for FPGA
- ✅ Improved feature selection via Gini importance
- ✅ Weighted anomaly score instead of OR logic

## 🌟 Key Features

### Advanced Feature Engineering (filter.py)
- **Adaptive KDE**: Silverman bandwidth + IQR outlier filtering
- **6 Anomaly Types**: Spatial, Persistence, Behavioral, Temporal, Acceleration, Stationary
- **Weighted Score**: Combines anomalies with learned weights (not simple OR)
- **Turning Angle Clean**: Filters GPS noise when elephant is stationary

### Feature Selection (training.py)
- **Gini Importance**: Random Forest-based feature ranking
- **Group-wise Selection**: Selects top features per group (kde, step, turning, etc.)
- **Correlation Check**: Removes highly correlated features (|corr| > 0.85)
- **Data Cleaning**: Removes garbage data (stationary at centroid)

### Quantization (quantization.py)
- **uint32 Quantization**: Scales features to [0, 2^32-1]
- **Label Encoding**: Converts categorical columns to integers
- **Scale Table Export**: Saves min/max/scale for dequantization

### Model Training (training_quantizied.py)
- **SMOTE + Undersampling**: Handles class imbalance
- **GridSearchCV**: Tunes hyperparameters
- **Threshold Optimization**: Finds best F1-score threshold
- **Verilog Export**: Converts decision trees to FPGA-compatible code

## 📊 Performance

| Metric | Value |
|--------|-------|
| **F1 Score (Optimized)** | ~0.98 |
| **ROC AUC** | ~0.99 |
| **Threshold** | 0.70-0.75 (optimized) |
| **Trees** | 12 (80 cây) |
| **Max Depth** | 6 |

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/elephant-behavior-classification.git
cd elephant-behavior-classification
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## ⚡ Quick Start

### Step 1: Feature Engineering
```bash
cd software/data
python filter.py
```
**Output**:
- `elephant_features_improved.csv` - Full features
- `elephant_raw_improved.csv` - Raw data + anomalies
- `elephant_anomaly_improved.png` - Visualization

### Step 2: Feature Selection
```bash
cd software/data
python training.py
```
**Output**:
- `elephant_features_selected_improved.csv` - Selected features
- `selected_feature_names.csv` - Feature list
- `gini_by_group_improved.png` - Feature importance
- `feature_correlation_improved.png` - Correlation heatmap

### Step 3: Quantization
```bash
cd software/quantization
python quantization.py
```
**Output**:
- `Quantized_Features.csv` - Quantized data
- `Quantization_Scales.csv` - Scale table
- `label_encoding_mapping.json` - Label mapping

### Step 4: Model Training + Verilog Export
```bash
cd software/model
python training_quantizied.py
```
**Output**:
- `model.pkl` - Trained model with optimal threshold
- `verilog_trees/*.v` - Decision trees in Verilog

## 📁 Project Structure

```
TKVM/
│
├── CLAUDE.md                          # Context cho AI assistant
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── PROJECT_STRUCTURE.txt              # Directory structure
├── SETUP_GUIDE.md                     # Setup guide
│
├── .claude/
│   ├── settings.json                  # Claude Code settings
│   └── skills/
│       └── testing/
│           └── SKILL.md               # Testing skill
│
├── docs/
│   ├── architecture.md                # System architecture
│   └── decisions/
│       └── 001-kde-vs-dbscan.md       # Architecture decision
│
├── software/                          # Main code directory
│   ├── data/
│   │   ├── filter.py                  # Feature engineering
│   │   └── training.py                # Feature selection
│   │
│   ├── quantization/
│   │   └── quantization.py            # Data quantization
│   │
│   └── model/
│       └── training_quantizied.py     # Model training + Verilog
│
├── src/
│   └── config.py                      # Configuration
│
└── verilog_trees/                     # Output: Verilog modules
    ├── decision_tree_1.v
    ├── decision_tree_2.v
    └── ...
```

## 💻 Usage

### Python Example
```python
import joblib
import pandas as pd

# Load model
model = joblib.load('software/model/model.pkl')

# Load test data
df = pd.read_csv('Quantized_Features.csv')
X = df.drop(columns=['is_outside'])

# Predict
predictions = model.predict(X)
probabilities = model.model.predict_proba(X)[:, 1]

# Apply threshold
threshold = model.threshold
y_pred = (probabilities >= threshold).astype(int)
```

### Verilog Example
```verilog
// Generated decision tree module
module decision_tree_1 (
    input wire [31:0] mean_speed,
    input wire [31:0] kde_prob_mean,
    // ... other features
    output reg tree_out
);

always @(*) begin
    if (mean_speed <= 32'h00001A2B) begin
        if (kde_prob_mean <= 32'h00002C4D) begin
            tree_out = 1'b1;  // Outside
        end else begin
            tree_out = 1'b0;  // Inside
        end
    end else begin
        // ... more conditions
    end
end
endmodule
```

## 📈 Results

### Feature Importance (Typical)
```
kde_prob_mean          - #1 most important
dist_to_centroid_mean  - #2
turning_entropy        - Top 3
mean_speed             - Top 5
```

### Ablation Study
| Configuration | F1 Score |
|--------------|----------|
| Full pipeline | ~0.98 |
| Without quantization | ~0.98 |
| Without SMOTE | ~0.95 |
| Default threshold (0.5) | ~0.92 |
| Optimized threshold | ~0.98 |

## 🧪 Testing

```bash
# Test feature engineering
cd software/data
python filter.py

# Test feature selection
python training.py

# Test full training
cd ../model
python training_quantizied.py
```

## 🔧 Configuration

Key parameters can be adjusted in each script:

**filter.py**:
```python
KDE: Silverman bandwidth (auto)
Behavioral: AND logic + percentiles
Score Threshold: 0.25
```

**training.py**:
```python
N_FEATURES_PER_GROUP = {
    'kde': 2,
    'speed': 2,
    'turning': 2,
    # ...
}
```

**training_quantizied.py**:
```python
n_estimators = 12
max_depth = 6
SMOTE sampling_strategy = 0.5
UnderSampler sampling_strategy = 0.8
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Multi-collar training
- Real-time pipeline
- FPGA deployment verification
- Additional anomaly types

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- Wildlife conservation team for GPS collar data
- Ivory Coast National Park for collaboration

## 📞 Contact

Project maintainer: [Your contact info]

---

**Made with ❤️ for wildlife conservation** 🐘🌍

**v2.0**: Now with FPGA support!
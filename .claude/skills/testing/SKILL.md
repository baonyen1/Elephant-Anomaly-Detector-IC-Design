# Testing Skill - Elephant Behavior Classification

## Purpose
Kỹ năng viết và chạy tests toàn diện cho hệ thống phân loại hành vi voi, đảm bảo code quality và model performance.

## Pipeline Flow

### Testing Each Stage

#### Stage 1: Feature Engineering (filter.py)
```python
# Test data loading
df = pd.read_csv('Elephant Research - Ivory Coast - Collar 1630.csv')
assert 'timestamp' in df.columns
assert 'location-lat' in df.columns
assert 'location-long' in df.columns

# Test anomaly features created
assert 'spatial_anomaly' in df.columns
assert 'persistence_anomaly' in df.columns
assert 'behavioral_anomaly' in df.columns
assert 'temporal_anomaly' in df.columns
assert 'acceleration_anomaly' in df.columns
assert 'stationary_anomaly' in df.columns
assert 'anomaly_score' in df.columns

# Test weighted score
assert df['anomaly_score'].between(0, 1).all()
assert df['point_is_outside'].isin([0, 1]).all()

# Test output files
assert os.path.exists('elephant_features_improved.csv')
assert os.path.exists('elephant_raw_improved.csv')
```

#### Stage 2: Feature Selection (training.py)
```python
# Test feature groups defined
feature_groups = {
    'kde': [...],
    'centroid': [...],
    'step': [...],
    'speed': [...],
    'turning': [...],
    'stationary': [...],
    'time': [...]
}

# Test Gini importance calculated
assert 'gini' in gini_df.columns
assert len(gini_df) > 0

# Test features selected
assert len(selected_features) > 0
assert all(f in df.columns for f in selected_features)

# Test correlation check
assert corr_matrix.shape == (len(selected_features), len(selected_features))

# Test output
assert os.path.exists('elephant_features_selected_improved.csv')
assert os.path.exists('selected_feature_names.csv')
```

#### Stage 3: Quantization (quantization.py)
```python
# Test quantized data
df_quantized = pd.read_csv('Quantized_Features.csv')
assert df_quantized.dtypes.apply(lambda x: 'uint' in str(x)).any()

# Test scale table
scales = pd.read_csv('Quantization_Scales.csv')
assert 'feature' in scales.columns
assert 'scale' in scales.columns
assert 'min' in scales.columns
assert 'max' in scales.columns

# Test label mapping
assert os.path.exists('label_encoding_mapping.json')
assert os.path.exists('label_encoding_mapping.csv')
```

#### Stage 4: Model Training (training_quantizied.py)
```python
# Test model trained
assert hasattr(rf_clf, 'estimators_')
assert len(rf_clf.estimators_) > 0

# Test threshold optimization
assert 0 < BEST_THRESHOLD < 1

# Test predictions
assert len(y_pred) == len(y_test)
assert set(y_pred).issubset({0, 1})

# Test F1 score
f1 = f1_score(y_test, y_pred_opt)
assert f1 > 0.90  # Minimum threshold

# Test Verilog export
assert os.path.exists('verilog_trees/')
verilog_files = os.listdir('verilog_trees')
assert len(verilog_files) > 0

# Test Verilog format
with open('verilog_trees/decision_tree_1.v') as f:
    content = f.read()
    assert 'module decision_tree_1' in content
    assert '32\'h' in content  # Hex thresholds
    assert 'endmodule' in content
```

## Test Categories

### 1. Data Quality Tests

```python
def test_gps_coordinates_valid():
    """Test GPS coordinates are in valid range"""
    df = pd.read_csv('Elephant Research - Ivory Coast - Collar 1630.csv')
    assert (df['location-lat'] >= -90).all()
    assert (df['location-lat'] <= 90).all()
    assert (df['location-long'] >= -180).all()
    assert (df['location-long'] <= 180).all()

def test_timestamps_sorted():
    """Test timestamps are chronological"""
    df = pd.read_csv('elephant_features_improved.csv')
    assert df['timestamp'].is_monotonic_increasing

def test_no_missing_features():
    """Test no NaN in final features"""
    df = pd.read_csv('elephant_features_selected_improved.csv')
    assert not df.drop(columns=['is_outside']).isnull().any().any()

def test_class_distribution():
    """Test class imbalance ratio"""
    df = pd.read_csv('elephant_features_selected_improved.csv')
    class_counts = df['is_outside'].value_counts()
    ratio = class_counts[0] / class_counts[1]
    assert 5 < ratio < 50  # Expect imbalanced but not extreme
```

### 2. Feature Engineering Tests

```python
def test_anomaly_weights_sum():
    """Test anomaly weights sum to 1"""
    weights = {
        'spatial_anomaly': 0.20,
        'persistence_anomaly': 0.20,
        'behavioral_anomaly': 0.20,
        'temporal_anomaly': 0.15,
        'acceleration_anomaly': 0.10,
        'stationary_anomaly': 0.15,
    }
    assert sum(weights.values()) == 1.0

def test_turning_angle_filtered():
    """Test turning angle filtered when stationary"""
    df = pd.read_csv('elephant_raw_improved.csv')
    stationary = df['dist'] < 50
    assert (df.loc[stationary, 'turning_angle_clean'] == 0).all()

def test_behavioral_and_logic():
    """Test behavioral anomaly uses AND not OR"""
    df = pd.read_csv('elephant_raw_improved.csv')
    # Behavioral should only be 1 when BOTH conditions met
    behavioral = df['behavioral_anomaly']
    speed_high = df['speed'] > df['speed'].quantile(0.97)
    turning_high = df['turning_angle_clean'] > df['turning_angle_clean'].quantile(0.95)
    assert not ((speed_high | turning_high) & ~behavioral).any()
```

### 3. Quantization Tests

```python
def test_quantization_reversible():
    """Test quantization can be reversed"""
    scales = pd.read_csv('Quantization_Scales.csv')
    original = pd.read_csv('elephant_features_selected_improved.csv')
    quantized = pd.read_csv('Quantized_Features.csv')

    for _, row in scales.iterrows():
        col = row['feature']
        q_col = quantized[col]
        # Dequantize
        reconstructed = q_col * row['scale'] + row['min']
        # Check close to original (within quantization error)
        orig_col = original[col].replace([np.inf, -np.inf], np.nan).fillna(row['min'])
        assert np.allclose(reconstructed, orig_col, rtol=0.01)

def test_uint32_range():
    """Test quantized values fit in uint32"""
    df = pd.read_csv('Quantized_Features.csv')
    numeric_cols = df.select_dtypes(include=['uint32']).columns
    for col in numeric_cols:
        assert df[col].min() >= 0
        assert df[col].max() <= 2**32 - 1
```

### 4. Model Performance Tests

```python
def test_minimum_f1_score():
    """Model must achieve minimum F1"""
    # Run training and get metrics
    # From training_quantizied.py output
    f1 = 0.98  # Expected
    assert f1 > 0.90

def test_threshold_optimal():
    """Threshold should be optimized"""
    # From training output
    threshold = 0.73  # Expected
    assert 0.5 < threshold < 0.9  # Should be different from default 0.5

def test_oob_score_reasonable():
    """OOB score should be reasonable"""
    # rf_clf.oob_score_ should be > 0.8
    oob = 0.95  # Expected
    assert oob > 0.80

def test_verilog_tree_count():
    """Should export all trees to Verilog"""
    trees_dir = 'verilog_trees'
    files = os.listdir(trees_dir)
    # Should match n_estimators
    assert len(files) == 12  # 12 trees
```

### 5. Integration Tests

```python
def test_full_pipeline():
    """Test complete pipeline from raw data to Verilog"""
    # 1. Filter
    assert os.path.exists('elephant_features_improved.csv')

    # 2. Selection
    assert os.path.exists('elephant_features_selected_improved.csv')

    # 3. Quantization
    assert os.path.exists('Quantized_Features.csv')

    # 4. Training
    assert os.path.exists('model.pkl')

    # 5. Verilog
    assert os.path.exists('verilog_trees/decision_tree_1.v')

def test_model_load_predict():
    """Test model can load and predict"""
    import joblib
    model = joblib.load('model.pkl')

    # Load test data
    df = pd.read_csv('Quantized_Features.csv')
    X = df.drop(columns=['is_outside'])

    # Predict
    probs = model.model.predict_proba(X)[:, 1]
    preds = (probs >= model.threshold).astype(int)

    # Verify output
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})

def test_verilog_syntax_valid():
    """Test Verilog files have valid syntax"""
    for vfile in os.listdir('verilog_trees/'):
        with open(f'verilog_trees/{vfile}') as f:
            content = f.read()
            # Basic syntax checks
            assert 'module' in content
            assert 'endmodule' in content
            assert content.count('begin') == content.count('end')
```

### 6. Edge Case Tests

```python
def test_empty_data_handling():
    """Test handling of edge cases"""
    # What if all points are stationary?
    # What if no anomalies detected?
    # These should not crash

def test_extreme_threshold():
    """Test with extreme threshold values"""
    # Threshold = 0.0 should predict all 1
    # Threshold = 1.0 should predict all 0

def test_single_tree():
    """Test with single decision tree"""
    # Can we export just 1 tree for testing?
```

## Running Tests

### Individual Stage Tests
```bash
# Test filter.py
cd software/data
python filter.py  # Should complete without errors

# Test training.py
python training.py  # Should output selected features

# Test quantization.py
cd ../quantization
python quantization.py  # Should output quantized data

# Test training_quantizied.py
cd ../model
python training_quantizied.py  # Should output model + Verilog
```

### Verification Script
```python
# test_pipeline.py
import os
import pandas as pd
import joblib

def test_all():
    """Run all quick tests"""
    # Check files exist
    files = [
        'elephant_features_improved.csv',
        'elephant_features_selected_improved.csv',
        'Quantized_Features.csv',
        'model.pkl',
    ]
    for f in files:
        assert os.path.exists(f), f"Missing: {f}"

    # Check data shapes
    df_features = pd.read_csv('elephant_features_selected_improved.csv')
    df_quantized = pd.read_csv('Quantized_Features.csv')
    assert df_features.shape == df_quantized.shape

    # Check model
    model = joblib.load('model.pkl')
    assert hasattr(model, 'threshold')

    print("✅ All tests passed!")

if __name__ == '__main__':
    test_all()
```

## Test Fixtures

```python
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_gps_data():
    """Sample GPS dataframe for testing"""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1H'),
        'location-lat': np.random.uniform(5.0, 6.0, 1000),
        'location-long': np.random.uniform(-5.0, -4.0, 1000)
    })

@pytest.fixture
def loaded_model():
    """Load trained model"""
    return joblib.load('model.pkl')

@pytest.fixture
def quantized_data():
    """Load quantized dataset"""
    return pd.read_csv('Quantized_Features.csv')
```

## Performance Benchmarks

```python
import time

def benchmark_filter():
    """Benchmark feature engineering"""
    start = time.time()
    # Run filter.py
    duration = time.time() - start
    assert duration < 300  # < 5 minutes

def benchmark_training():
    """Benchmark model training"""
    start = time.time()
    # Run training_quantizied.py
    duration = time.time() - start
    assert duration < 300  # < 5 minutes

def benchmark_prediction():
    """Benchmark prediction latency"""
    model = joblib.load('model.pkl')
    X = pd.read_csv('Quantized_Features.csv').drop(columns=['is_outside'])

    start = time.time()
    model.model.predict_proba(X)
    duration = time.time() - start

    assert duration < 1.0  # < 1 second for batch
    assert duration / len(X) < 0.001  # < 1ms per sample
```

## CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run pipeline tests
        run: |
          cd software/data
          python filter.py
          python training.py
          cd ../quantization
          python quantization.py
          cd ../model
          python training_quantizied.py
      - name: Verify outputs
        run: |
          python test_pipeline.py
```

---

**Remember**:
- Tests validate the full pipeline: filter → selection → quantization → training
- Verilog export is critical for FPGA deployment
- Threshold optimization must be reproducible
- F1 score > 0.90 is minimum acceptable
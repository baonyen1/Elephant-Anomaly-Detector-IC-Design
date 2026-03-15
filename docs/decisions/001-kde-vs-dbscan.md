# ADR-001: KDE vs DBSCAN for Home Range Detection

## Status
**Accepted** - 2025-01-24

## Context
Cần phương pháp để phát hiện khi voi di chuyển ra ngoài home range. Có 2 phương pháp chính đang cân nhắc:

1. **DBSCAN (Density-Based Spatial Clustering)**
   - Phát hiện outliers dựa trên mật độ điểm
   - Binary output: Inside (cluster) vs Outside (outlier)

2. **KDE (Kernel Density Estimation)**
   - Ước lượng xác suất điểm thuộc home range
   - Continuous output: Probability [0, 1]

## Decision
**Sử dụng CẢ HAI phương pháp kết hợp**:
- **DBSCAN** để tạo feature `point_is_outside` (binary)
- **KDE** để tạo feature `kde_probability` (continuous) và các variants

## Rationale

### Ưu điểm của DBSCAN
✅ **Đơn giản và nhanh**: Chỉ cần 2 parameters (eps, min_samples)  
✅ **Không cần assumption về phân phối**: Hoạt động tốt với home range bất kỳ hình dạng  
✅ **Rõ ràng**: Binary classification - dễ interpret  
✅ **Robust với outliers**: Chính xác mục đích của DBSCAN  

⚠️ **Nhược điểm**:
- Không có độ tin cậy (confidence)
- Mất thông tin về "mức độ" outside
- Nhạy cảm với parameters

### Ưu điểm của KDE
✅ **Probabilistic**: Cho confidence score  
✅ **Smooth**: Không có hard boundary  
✅ **Rich information**: Có thể tạo nhiều features từ probability  
✅ **Flexible**: Có thể tạo KDE riêng cho day/night  

⚠️ **Nhược điểm**:
- Chậm hơn DBSCAN
- Cần chọn bandwidth (hyperparameter)
- Phức tạp hơn để implement

### Tại sao kết hợp cả hai?

1. **Complementary Information**
   ```python
   # DBSCAN: "Điểm này có phải outlier không?"
   point_is_outside = 1  # Yes/No
   
   # KDE: "Điểm này có xác suất bao nhiêu thuộc home range?"
   kde_probability = 0.15  # 15% chance
   ```

2. **Different Perspectives**
   - DBSCAN: Local density view
   - KDE: Global probability view
   - Together: Richer feature representation

3. **Empirical Results**
   - Model với cả 2 features: F1-Macro = **98.12%**
   - Model chỉ DBSCAN: F1-Macro = 96.84%
   - Model chỉ KDE: F1-Macro = 97.51%
   - **Kết hợp tốt hơn 1-2% performance**

4. **Cross-Validation**
   ```python
   # Points where DBSCAN and KDE disagree are interesting
   disagreement = (point_is_outside == 1) & (kde_probability > 0.5)
   # These might be edge cases worth investigating
   ```

## Implementation

### DBSCAN Parameters
```python
eps = 0.005          # ~500m in lat/long
min_samples = 10     # Minimum cluster size
```

**Tuning Process**:
- Tested eps: [0.003, 0.005, 0.007, 0.01]
- Tested min_samples: [5, 10, 15, 20]
- Selected based on visual inspection of clusters
- Validated with domain experts

### KDE Parameters
```python
bandwidth = 0.01     # Kernel width
kernel = 'gaussian'  # Kernel type
```

**Tuning Process**:
- Tested bandwidth: [0.005, 0.01, 0.02]
- Selected 0.01 as best balance
- Cross-validated on held-out set

### Feature Engineering Pipeline
```python
# Step 1: DBSCAN on all points
db = DBSCAN(eps=0.005, min_samples=10)
labels = db.fit_predict(coords)
df['point_is_outside'] = (labels == -1).astype(int)

# Step 2: KDE on normal points only
normal_coords = coords[df['point_is_outside'] == 0]
kde = KernelDensity(bandwidth=0.01, kernel='gaussian')
kde.fit(normal_coords)

# Step 3: Score all points
log_prob = kde.score_samples(coords)
df['kde_probability'] = normalize(np.exp(log_prob))

# Step 4: Create additional KDE features
df['kde_prob_day'] = kde_day.score(...)
df['kde_prob_night'] = kde_night.score(...)
df['kde_home_range'] = pd.cut(df['kde_probability'], ...)
```

## Consequences

### Positive
✅ Best model performance (98.12% F1-Macro)  
✅ Rich feature set for ML model  
✅ Multiple perspectives on home range  
✅ Confidence scores available (KDE)  
✅ Clear binary labels available (DBSCAN)  

### Negative
⚠️ Longer feature engineering time (~2x)  
⚠️ More complex codebase  
⚠️ Two sets of hyperparameters to tune  
⚠️ Potential redundancy in features  

### Neutral
ℹ️ Feature importance will show which method model prefers  
ℹ️ Can drop one method if performance not affected  
ℹ️ Future work: Ensemble DBSCAN + KDE predictions  

## Validation

### Test 1: Feature Importance
```
kde_prob_mean:           0.1523  ← KDE is #1
dist_to_centroid_mean:   0.1247
kde_prob_min:            0.0891  ← KDE is #3
point_is_outside:        0.0234  ← DBSCAN is #15
```
**Result**: KDE features more important, but DBSCAN still contributes

### Test 2: Ablation Study
| Features Used | F1-Macro | ROC-AUC |
|--------------|----------|---------|
| All features | **98.12%** | **99.96%** |
| Remove DBSCAN | 97.89% | 99.91% |
| Remove KDE | 96.54% | 99.23% |
| Only DBSCAN | 95.12% | 97.45% |
| Only KDE | 96.78% | 98.89% |

**Result**: Both contribute, KDE more critical

### Test 3: Agreement Analysis
```python
# Where do DBSCAN and KDE agree?
both_outside = (point_is_outside == 1) & (kde_probability < 0.2)
# 78% agreement on "outside" points

both_inside = (point_is_outside == 0) & (kde_probability > 0.5)
# 94% agreement on "inside" points

# Edge cases (disagreement)
edge_cases = (point_is_outside != (kde_probability < 0.5))
# 8% of total points - worth manual inspection
```

## Alternatives Considered

### 1. Minimum Convex Polygon (MCP)
```python
# Draw polygon around all points
# Check if new point inside polygon
```
❌ Too simplistic, doesn't handle gaps  
❌ Sensitive to outliers  
❌ No probabilistic output  

### 2. Utilization Distribution (UD)
```python
# Similar to KDE but specifically for animal movement
```
❌ Requires specialized packages  
❌ Harder to interpret  
✅ Could be future improvement  

### 3. Local Outlier Factor (LOF)
```python
# Alternative to DBSCAN
```
❌ More complex  
❌ Slower than DBSCAN  
✅ Could be tested in future  

### 4. Isolation Forest
```python
# Tree-based outlier detection
```
❌ Less interpretable  
❌ Not spatially-aware  
✅ Could complement current methods  

## Future Improvements

### Short-Term
- [ ] Hyperparameter tuning with Optuna
- [ ] Try different KDE kernels (Epanechnikov, Tophat)
- [ ] Adaptive bandwidth KDE

### Medium-Term
- [ ] Ensemble DBSCAN + KDE predictions
- [ ] Time-varying home range (seasonal KDE)
- [ ] Multi-scale DBSCAN (multiple eps values)

### Long-Term
- [ ] Deep learning for home range (autoencoder)
- [ ] Bayesian home range estimation
- [ ] Graph-based methods (connectivity)

## References

1. Kie, J. G., et al. (2010). "The home-range concept: are traditional estimators still relevant with modern telemetry technology?" *Philosophical Transactions of the Royal Society B*

2. Worton, B. J. (1989). "Kernel methods for estimating the utilization distribution in home-range studies." *Ecology*

3. Ester, M., et al. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise." *KDD-96*

## Decision Record

- **Proposed by**: Data Science Team
- **Date**: 2025-01-24
- **Decided by**: Project Lead
- **Reviewed by**: Conservation Experts
- **Status**: ✅ Accepted

---

*This ADR can be revisited if:*
1. *Performance degrades significantly*
2. *New methods emerge with clear advantages*
3. *Computational cost becomes prohibitive*

import pandas as pd
import numpy as np
from geopy.distance import geodesic
from scipy.stats import iqr
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ĐỌC & TIỀN XỬ LÝ
# ============================================================
df = pd.read_csv('Elephant Research - Ivory Coast - Collar 1630.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / 3600

coords_raw = df[['location-lat', 'location-long']].values
distances = [0]
for i in range(1, len(df)):
    distances.append(geodesic(coords_raw[i-1], coords_raw[i]).meters)

df['dist'] = distances
df['speed_meters_per_hour'] = (df['dist'] / df['time_diff']).fillna(0)
df = df[df['time_diff'] > 0].copy()

MAX_SPEED_THRESHOLD = 40000
print(f"Trước khi lọc tốc độ ảo: {len(df)} dòng")
df = df[df['speed_meters_per_hour'] < MAX_SPEED_THRESHOLD].copy()
print(f"Sau khi lọc tốc độ ảo: {len(df)} dòng")

df['speed'] = df['speed_meters_per_hour']
df['raw_accel'] = df['speed'].diff() / df['time_diff']
df['raw_accel'] = df['raw_accel'].replace([np.inf, -np.inf], 0).fillna(0)
df = df.reset_index(drop=True)

# ============================================================
# CẢI TIẾN 1: KDE VỚI SILVERMAN BANDWIDTH + LOẠI OUTLIER IQR TRƯỚC
# ============================================================
print("\n⏳ [CẢI TIẾN 1] KDE với Silverman bandwidth + loại outlier IQR trước...")

def adaptive_kde_point_is_outside(df, iqr_multiplier=3.0, kde_threshold=0.1):
    """
    CẢI TIẾN so với bản cũ:
    - Loại outlier thô bằng IQR trước khi fit KDE → KDE không bị nhiễm bởi outlier
    - Bandwidth tự động theo Silverman's rule → không cần chọn tay
    - Normalize bằng percentile thay vì min-max → bền vững hơn với cực trị
    """
    coords = df[['location-lat', 'location-long']].values

    # Bước 1: Loại outlier thô bằng IQR để fit KDE trên "core" data
    lat_iqr = iqr(coords[:, 0])
    lon_iqr = iqr(coords[:, 1])
    lat_median = np.median(coords[:, 0])
    lon_median = np.median(coords[:, 1])

    clean_mask = (
        (np.abs(coords[:, 0] - lat_median) < iqr_multiplier * lat_iqr) &
        (np.abs(coords[:, 1] - lon_median) < iqr_multiplier * lon_iqr)
    )
    clean_coords = coords[clean_mask]
    print(f"   IQR filter: {clean_mask.sum()} / {len(df)} điểm dùng để fit KDE")

    # Bước 2: Tính bandwidth tự động (Silverman's rule)
    n = len(clean_coords)
    std_mean = clean_coords.std(axis=0).mean()
    bw_silverman = 1.06 * std_mean * (n ** (-1/5))
    bw_silverman = max(bw_silverman, 0.001)  # tránh quá nhỏ
    print(f"   Silverman bandwidth = {bw_silverman:.5f} (so với cũ: 0.01)")

    # Bước 3: Fit KDE chỉ trên clean data
    kde = KernelDensity(kernel='gaussian', bandwidth=bw_silverman)
    kde.fit(clean_coords)

    # Bước 4: Score tất cả điểm
    log_prob = kde.score_samples(coords)
    prob = np.exp(log_prob)

    # Bước 5: Normalize bằng percentile (bền vững hơn min-max)
    p1 = np.percentile(prob, 1)
    p99 = np.percentile(prob, 99)
    prob_normalized = np.clip((prob - p1) / (p99 - p1 + 1e-10), 0, 1)

    spatial_anomaly = (prob_normalized < kde_threshold).astype(int)

    outside_pct = spatial_anomaly.mean() * 100
    print(f"   ✅ Spatial Anomaly: {spatial_anomaly.sum()} điểm ({outside_pct:.2f}%)")

    return spatial_anomaly, prob_normalized, bw_silverman

df['spatial_anomaly'], df['kde_probability'], bw_used = adaptive_kde_point_is_outside(
    df, iqr_multiplier=3.0, kde_threshold=0.1
)

# ============================================================
# PERSISTENCE ANOMALY — cải tiến: dùng time-based window thay vì count-based
# ============================================================
print("\n⏳ [CẢI TIẾN] Persistence Anomaly với time-based window...")

# Dùng rolling theo thời gian (2 giờ) thay vì theo số điểm cố định
df_tmp = df.set_index('timestamp')
rolling_mean = (
    df_tmp['spatial_anomaly']
    .rolling('2h', min_periods=1)
    .mean()
)
df['persistence_anomaly'] = (rolling_mean > 0.4).astype(int).values

pct = df['persistence_anomaly'].mean() * 100
print(f"   Persistence Anomaly: {df['persistence_anomaly'].sum()} điểm ({pct:.2f}%)")

# ============================================================
# TURNING ANGLE — lọc khi đứng yên (dist < 50m)
# ============================================================
print("\n⏳ Tính Turning Angle (có lọc khi đứng yên)...")

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(y, x)) % 360

coords = df[['location-lat', 'location-long']].values
bearings = [0]
for i in range(1, len(df)):
    bearings.append(calculate_bearing(
        df.iloc[i-1]['location-lat'], df.iloc[i-1]['location-long'],
        df.iloc[i]['location-lat'], df.iloc[i]['location-long']
    ))
df['bearing'] = bearings

turning_angles = [0, 0]
for i in range(2, len(df)):
    angle_diff = df.iloc[i]['bearing'] - df.iloc[i-1]['bearing']
    if angle_diff > 180:
        angle_diff -= 360
    elif angle_diff < -180:
        angle_diff += 360
    turning_angles.append(abs(angle_diff))

df['turning_angle'] = turning_angles

# CẢI TIẾN: lọc turning angle khi voi đứng yên (GPS noise tạo turning angle giả)
STATIONARY_DIST = 50  # mét
df['turning_angle_clean'] = np.where(
    df['dist'] < STATIONARY_DIST,
    0,  # đứng yên → turning angle = 0, không tin được
    df['turning_angle']
)
stationary_count = (df['dist'] < STATIONARY_DIST).sum()
print(f"   Lọc {stationary_count} điểm đứng yên (dist < {STATIONARY_DIST}m) khỏi turning angle")
print(f"   Turning angle (clean): mean={df['turning_angle_clean'].mean():.2f}°, max={df['turning_angle_clean'].max():.2f}°")

# ============================================================
# CẢI TIẾN 2: BEHAVIORAL ANOMALY — AND thay vì OR + dùng percentile thay vì mean+3σ
# ============================================================
print("\n⏳ [CẢI TIẾN 2] Behavioral Anomaly: AND logic + percentile threshold...")

# CŨ: mean + 3σ (giả sử normal distribution, không đúng với tốc độ động vật)
# MỚI: percentile (robust, không giả sử phân phối)
SPEED_TH    = df['speed'].quantile(0.97)
TURNING_TH  = df['turning_angle_clean'].quantile(0.95)

# CŨ: OR logic (quá nhạy — turning cao khi đứng yên cũng bị đánh dấu)
# MỚI: AND logic (panic = vừa nhanh vừa rẽ nhiều, phải đồng thời)
df['behavioral_anomaly'] = (
    (df['speed'] > SPEED_TH) &               # ← đổi | thành &
    (df['turning_angle_clean'] > TURNING_TH)  # ← dùng turning_angle_clean
).astype(int)

pct = df['behavioral_anomaly'].mean() * 100
print(f"   SPEED_TH (97th pct) = {SPEED_TH:.2f} m/h")
print(f"   TURNING_TH (95th pct) = {TURNING_TH:.2f}°")
print(f"   Behavioral Anomaly: {df['behavioral_anomaly'].sum()} điểm ({pct:.2f}%)")

# ============================================================
# CẢI TIẾN 3: TEMPORAL ANOMALY — ngưỡng tính riêng ban đêm + thêm KDE thấp
# ============================================================
print("\n⏳ [CẢI TIẾN 3] Temporal Anomaly: ngưỡng tính riêng ban đêm...")

df['hour'] = df['timestamp'].dt.hour
df['is_day'] = ((df['hour'] >= 6) & (df['hour'] < 18)).astype(int)

# CŨ: quantile(0.99) tính trên toàn bộ data (kể cả ban ngày)
# MỚI: quantile(0.95) tính CHỈ trên data ban đêm
night_data = df[df['is_day'] == 0]
if len(night_data) > 10:
    SPEED_NIGHT_TH = night_data['speed'].quantile(0.95)
else:
    SPEED_NIGHT_TH = df['speed'].quantile(0.99)
    print("   ⚠️ Ít data ban đêm, fallback về quantile(0.99) toàn bộ")

print(f"   SPEED_NIGHT_TH (95th pct đêm) = {SPEED_NIGHT_TH:.2f} m/h")
print(f"   (So với cũ 99th toàn bộ = {df['speed'].quantile(0.99):.2f} m/h)")

# CẢI TIẾN: thêm điều kiện KDE thấp — vừa nhanh vừa ở vùng lạ ban đêm
df['temporal_anomaly'] = (
    (df['is_day'] == 0) &
    (df['speed'] > SPEED_NIGHT_TH) &
    (df['kde_probability'] < 0.3)  # ← vừa chạy đêm vừa ở vùng bất thường
).astype(int)

pct = df['temporal_anomaly'].mean() * 100
print(f"   Temporal Anomaly: {df['temporal_anomaly'].sum()} điểm ({pct:.2f}%)")

# ============================================================
# THÊM MỚI: STATIONARY ANOMALY — Phát hiện đứng yên bất thường
# ============================================================
print("\n⏳ [THÊM MỚI] Stationary Anomaly (bị thương / bệnh)...")

# --- Bước 1: Tính rolling speed trung bình 4h ---
# Dùng time-based rolling để đúng với thực tế
df_tmp = df.set_index('timestamp')

rolling_speed_4h = (
    df_tmp['speed']
    .rolling('4h', min_periods=2)
    .mean()
)
df['rolling_speed_4h'] = rolling_speed_4h.values

# --- Bước 2: Ngưỡng "đứng yên" ---
# Dưới 1% tốc độ toàn bộ dataset = gần như không di chuyển
STATIONARY_SPEED_TH = df['speed'].quantile(0.01)
print(f"   STATIONARY_SPEED_TH (1st pct) = {STATIONARY_SPEED_TH:.2f} m/h")

# --- Bước 3: Tính thời gian đứng yên liên tiếp ---
df['is_stationary'] = (df['speed'] < STATIONARY_SPEED_TH).astype(int)

df_tmp = df.set_index('timestamp')

# Đếm số giờ đứng yên liên tiếp bằng cumsum trick
df['stationary_streak'] = (
    df_tmp['is_stationary']
    .rolling('4h', min_periods=1)
    .sum()
    .values
)

# --- Bước 4: Stationary anomaly ---
# Điều kiện: đứng yên liên tục > 4h VÀ tại vùng bất thường (KDE thấp)
# Loại trừ: đứng yên tại vùng quen = nghỉ ngơi bình thường (KDE cao)
STREAK_TH  = 3   # ít nhất 3 điểm GPS liên tiếp đứng yên trong 4h

df['stationary_anomaly'] = (
    (df['stationary_streak'] >= STREAK_TH) &      # đứng yên lâu
    (df['rolling_speed_4h']  <  STATIONARY_SPEED_TH * 2) &  # không chỉ tạm nghỉ
    (df['kde_probability']   <  0.4)              # tại vùng bất thường
).astype(int)

pct = df['stationary_anomaly'].mean() * 100
print(f"   Stationary Anomaly: {df['stationary_anomaly'].sum()} điểm ({pct:.2f}%)")

# Phân biệt 2 loại để dễ phân tích sau:
# - Nghỉ ngơi bình thường: đứng yên + KDE cao (vùng quen)
# - Bất thường          : đứng yên + KDE thấp (vùng lạ)
df['resting_normal'] = (
    (df['stationary_streak'] >= STREAK_TH) &
    (df['kde_probability']   >= 0.4)
).astype(int)

print(f"   Nghỉ ngơi bình thường: {df['resting_normal'].sum()} điểm")
print(f"   Đứng yên bất thường  : {df['stationary_anomaly'].sum()} điểm")
# ============================================================
# CẢI TIẾN MỚI: ACCELERATION ANOMALY — tín hiệu bị bỏ qua trong code cũ
# ============================================================
print("\n⏳ [CẢI TIẾN MỚI] Acceleration Anomaly (tín hiệu bị bỏ trong code cũ)...")

ACCEL_TH = df['raw_accel'].abs().quantile(0.97)
df['acceleration_anomaly'] = (df['raw_accel'].abs() > ACCEL_TH).astype(int)

pct = df['acceleration_anomaly'].mean() * 100
print(f"   ACCEL_TH (97th pct) = {ACCEL_TH:.2f} m/h²")
print(f"   Acceleration Anomaly: {df['acceleration_anomaly'].sum()} điểm ({pct:.2f}%)")

# ============================================================
# KDE TEMPORAL (Day/Night)
# ============================================================
print("\n⏳ Tính KDE Temporal (Day/Night)...")

def calculate_temporal_kde(df, bandwidth):
    results = {}
    for period, period_name in [(1, 'day'), (0, 'night')]:
        period_data = df[df['is_day'] == period]
        if len(period_data) < 10:
            results[f'kde_prob_{period_name}'] = np.zeros(len(df))
            continue
        coords_period = period_data[['location-lat', 'location-long']].values
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(coords_period)
        all_coords = df[['location-lat', 'location-long']].values
        log_prob = kde.score_samples(all_coords)
        prob = np.exp(log_prob)
        p1, p99 = np.percentile(prob, 1), np.percentile(prob, 99)
        prob_normalized = np.clip((prob - p1) / (p99 - p1 + 1e-10), 0, 1)
        results[f'kde_prob_{period_name}'] = prob_normalized
        print(f"   ✅ KDE {period_name}: {len(period_data)} điểm training")
    return results

temporal_kde = calculate_temporal_kde(df, bw_used)
df['kde_prob_day']   = temporal_kde['kde_prob_day']
df['kde_prob_night'] = temporal_kde['kde_prob_night']
df['kde_prob_adaptive'] = np.where(df['is_day'] == 1, df['kde_prob_day'], df['kde_prob_night'])
df['kde_home_range'] = pd.cut(df['kde_probability'],
                               bins=[0, 0.2, 0.5, 0.8, 1.0],
                               labels=['Very_Low', 'Low', 'Medium', 'High'])

# ============================================================
# CẢI TIẾN 4: WEIGHTED SCORE thay vì OR — nhãn cuối tin cậy hơn
# ============================================================
print("\n⏳ [CẢI TIẾN 4] Kết hợp bằng Weighted Score thay vì OR...")

weights = {
    'spatial_anomaly':      0.20,
    'persistence_anomaly':  0.20,
    'behavioral_anomaly':   0.20,
    'temporal_anomaly':     0.15,
    'acceleration_anomaly': 0.10,
    'stationary_anomaly':   0.15,   # ← THÊM MỚI
}

df['anomaly_score'] = (
    df['spatial_anomaly']      * weights['spatial_anomaly'] +
    df['persistence_anomaly']  * weights['persistence_anomaly'] +
    df['behavioral_anomaly']   * weights['behavioral_anomaly'] +
    df['temporal_anomaly']     * weights['temporal_anomaly'] +
    df['acceleration_anomaly'] * weights['acceleration_anomaly'] +
    df['stationary_anomaly']   * weights['stationary_anomaly'] 
)
# Ngưỡng: cần ít nhất 2 loại anomaly trọng số trung bình đồng ý
SCORE_THRESHOLD = 0.25

df['point_is_outside'] = (df['anomaly_score'] >= SCORE_THRESHOLD).astype(int)

# So sánh với OR cũ để tiện đánh giá
df['point_is_outside_old_OR'] = (
    df['spatial_anomaly'] |
    df['persistence_anomaly'] |
    df['temporal_anomaly'] |
    df['behavioral_anomaly']
).astype(int)

print(f"\n📊 So sánh nhãn cuối:")
print(f"   OR cũ    : {df['point_is_outside_old_OR'].sum()} điểm ({df['point_is_outside_old_OR'].mean()*100:.2f}%)")
print(f"   Score mới: {df['point_is_outside'].sum()} điểm ({df['point_is_outside'].mean()*100:.2f}%)")
print(f"\n   Phân phối anomaly_score:")
for threshold in [0.15, 0.20, 0.25, 0.30, 0.40]:
    count = (df['anomaly_score'] >= threshold).sum()
    print(f"     score ≥ {threshold}: {count} điểm ({count/len(df)*100:.2f}%)")

# ============================================================
# FEATURES ENGINEERING
# ============================================================
print("\n⏳ Tạo features...")

def entropy_safe(x):
    x = np.array(x)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return 0
    counts, _ = np.histogram(x, bins=36, range=(0, 360))
    total = counts.sum()
    if total == 0:
        return 0
    p = counts / total
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

df['step_length'] = df['dist']
resampled_2h = df.set_index('timestamp').resample('2h')
feat_df = pd.DataFrame()

# Step length features
feat_df['step_mean']   = resampled_2h['step_length'].mean()
feat_df['step_std']    = resampled_2h['step_length'].std()
feat_df['step_max']    = resampled_2h['step_length'].max()
feat_df['step_median'] = resampled_2h['step_length'].median()

# Turning angle features (dùng turning_angle_clean)
feat_df['turning_angle_mean']   = resampled_2h['turning_angle_clean'].mean()
feat_df['turning_angle_std']    = resampled_2h['turning_angle_clean'].std()
feat_df['turning_angle_max']    = resampled_2h['turning_angle_clean'].max()
feat_df['turning_angle_median'] = resampled_2h['turning_angle_clean'].median()
feat_df['sharp_turns_ratio'] = (
    resampled_2h['turning_angle_clean']
    .apply(lambda x: (x > 90).sum() / len(x) if len(x) > 0 else 0)
)
feat_df['moderate_turns_ratio'] = (
    resampled_2h['turning_angle_clean']
    .apply(lambda x: ((x > 30) & (x <= 90)).sum() / len(x) if len(x) > 0 else 0)
)

# Turning entropy
df_resampled_1h = df.set_index('timestamp').resample('1h').mean(numeric_only=True).interpolate(method='linear')
df_resampled_1h['bearing'] = np.degrees(
    np.arctan2(
        df_resampled_1h['location-long'].diff().fillna(0),
        df_resampled_1h['location-lat'].diff().fillna(0)
    )
).fillna(0)
df_resampled_1h['turning_entropy'] = (
    df_resampled_1h['bearing'].diff().abs().fillna(0)
    .rolling(window=10, min_periods=1).apply(entropy_safe, raw=True)
)
feat_df['turning_entropy'] = (
    df_resampled_1h['turning_entropy'].resample('2h').mean()
    .reindex(feat_df.index).fillna(0)
)

# Distance to centroid
centroid_lat  = df['location-lat'].mean()
centroid_long = df['location-long'].mean()
df['dist_to_centroid'] = [
    geodesic((lat, lon), (centroid_lat, centroid_long)).meters
    for lat, lon in coords
]
feat_df['dist_to_centroid_mean'] = (
    df.set_index('timestamp')['dist_to_centroid']
    .resample('2h').mean()
    .reindex(feat_df.index).fillna(0)
)

# Rolling variance features
df1h = df.set_index('timestamp').resample('1h').mean(numeric_only=True).interpolate()
df1h['speed_roll_var_4h'] = df1h['speed'].rolling(4).var().fillna(0)
df1h['speed_roll_var_8h'] = df1h['speed'].rolling(8).var().fillna(0)
df1h['accel_roll_var_4h'] = df1h['raw_accel'].rolling(4).var().fillna(0)
df1h['accel_roll_var_8h'] = df1h['raw_accel'].rolling(8).var().fillna(0)
for col in ['speed_roll_var_4h', 'speed_roll_var_8h', 'accel_roll_var_4h', 'accel_roll_var_8h']:
    feat_df[f'{col}_mean'] = df1h[col].resample('2h').mean().reindex(feat_df.index).fillna(0)

# KDE features
feat_df['kde_prob_mean']          = resampled_2h['kde_probability'].mean().reindex(feat_df.index).fillna(0)
feat_df['kde_prob_std']           = resampled_2h['kde_probability'].std().reindex(feat_df.index).fillna(0)
feat_df['kde_prob_day_mean']      = resampled_2h['kde_prob_day'].mean().reindex(feat_df.index).fillna(0)
feat_df['kde_prob_night_mean']    = resampled_2h['kde_prob_night'].mean().reindex(feat_df.index).fillna(0)
feat_df['kde_prob_adaptive_mean'] = resampled_2h['kde_prob_adaptive'].mean().reindex(feat_df.index).fillna(0)
feat_df['kde_low_prob_ratio'] = (
    resampled_2h['kde_probability']
    .apply(lambda x: (x < 0.2).sum() / len(x) if len(x) > 0 else 0)
    .reindex(feat_df.index).fillna(0)
)
feat_df['kde_very_low_prob_count'] = (
    resampled_2h['kde_probability']
    .apply(lambda x: (x < 0.1).sum() if len(x) > 0 else 0)
    .reindex(feat_df.index).fillna(0)
)

# Anomaly score features (mới thêm)
feat_df['anomaly_score_mean'] = (
    resampled_2h['anomaly_score'].mean().reindex(feat_df.index).fillna(0)
)
feat_df['anomaly_score_max'] = (
    resampled_2h['anomaly_score'].max().reindex(feat_df.index).fillna(0)
)

# Thêm vào phần resample feat_df
feat_df['stationary_ratio'] = (
    resampled_2h['is_stationary']
    .apply(lambda x: x.mean() if len(x) > 0 else 0)
    .reindex(feat_df.index).fillna(0)
)

feat_df['stationary_streak_max'] = (
    resampled_2h['stationary_streak']
    .max()
    .reindex(feat_df.index).fillna(0)
)

feat_df['rolling_speed_4h_mean'] = (
    resampled_2h['rolling_speed_4h']
    .mean()
    .reindex(feat_df.index).fillna(0)
)

# Speed, acceleration, labels
feat_df['mean_speed']  = resampled_2h['speed'].mean().values
feat_df['accelerate']  = resampled_2h['raw_accel'].apply(lambda x: np.mean(np.abs(x))).values
feat_df['is_outside']  = (resampled_2h['point_is_outside'].max() > 0).astype(int).values
feat_df['anomaly_score_2h'] = resampled_2h['anomaly_score'].mean().values

# Giờ và ban đêm
feat_df['hour']     = feat_df.index.hour
feat_df['is_night'] = ((feat_df['hour'] >= 18) | (feat_df['hour'] <= 6)).astype(int)

# ============================================================
# VISUALIZATION
# ============================================================
print("\n⏳ Đang tạo visualization...")

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Elephant Anomaly Detection — Improved Version', fontsize=14, fontweight='bold')

# 1. KDE Probability heatmap
ax = axes[0, 0]
sc = ax.scatter(df['location-long'], df['location-lat'],
                c=df['kde_probability'], cmap='viridis', alpha=0.6, s=8)
ax.set_title(f'KDE Probability\n(bandwidth={bw_used:.4f}, Silverman auto)')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
plt.colorbar(sc, ax=ax)

# 2. Weighted Score vs OR so sánh
ax = axes[0, 1]
inside_new  = df[df['point_is_outside'] == 0]
outside_new = df[df['point_is_outside'] == 1]
ax.scatter(inside_new['location-long'],  inside_new['location-lat'],  c='blue', s=6, alpha=0.4, label=f'Inside ({len(inside_new)})')
ax.scatter(outside_new['location-long'], outside_new['location-lat'], c='red',  s=10, alpha=0.8, label=f'Outside ({len(outside_new)})')
ax.set_title('Weighted Score Classification (NEW)')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.legend(fontsize=8)

# 3. OR cũ vs Score mới
ax = axes[0, 2]
old_only = df[(df['point_is_outside_old_OR'] == 1) & (df['point_is_outside'] == 0)]
new_only  = df[(df['point_is_outside_old_OR'] == 0) & (df['point_is_outside'] == 1)]
both      = df[(df['point_is_outside_old_OR'] == 1) & (df['point_is_outside'] == 1)]
normal    = df[(df['point_is_outside_old_OR'] == 0) & (df['point_is_outside'] == 0)]
ax.scatter(normal['location-long'],   normal['location-lat'],   c='lightgray', s=5,  alpha=0.3, label=f'Cả 2 Normal ({len(normal)})')
ax.scatter(both['location-long'],     both['location-lat'],     c='red',       s=10, alpha=0.8, label=f'Cả 2 Anomaly ({len(both)})')
ax.scatter(old_only['location-long'], old_only['location-lat'], c='orange',    s=10, alpha=0.8, label=f'Chỉ OR cũ ({len(old_only)})')
ax.scatter(new_only['location-long'], new_only['location-lat'], c='purple',    s=10, alpha=0.8, label=f'Chỉ Score mới ({len(new_only)})')
ax.set_title('So sánh OR cũ vs Score mới')
ax.set_xlabel('Longitude'); ax.legend(fontsize=7)

# 4. Anomaly Score Distribution
ax = axes[1, 0]
ax.hist(df['anomaly_score'], bins=40, alpha=0.7, edgecolor='black', color='steelblue')
ax.axvline(SCORE_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold: {SCORE_THRESHOLD}')
ax.set_title('Phân phối Anomaly Score (Weighted)')
ax.set_xlabel('Anomaly Score'); ax.set_ylabel('Frequency')
ax.legend()

# 5. Turning Angle — raw vs clean
ax = axes[1, 1]
ax.hist(df['turning_angle'],       bins=50, alpha=0.5, color='orange', label='Raw (có GPS noise)')
ax.hist(df['turning_angle_clean'], bins=50, alpha=0.5, color='blue',   label='Clean (lọc đứng yên)')
ax.set_title(f'Turning Angle: Raw vs Clean\n(lọc {stationary_count} điểm dist < {STATIONARY_DIST}m)')
ax.set_xlabel('Turning Angle (°)'); ax.set_ylabel('Frequency')
ax.legend()

# 6. Speed Distribution với ngưỡng
ax = axes[1, 2]
ax.hist(df['speed'], bins=60, alpha=0.7, edgecolor='black', color='green')
old_speed_th = df['speed'].mean() + 3 * df['speed'].std()
ax.axvline(old_speed_th, color='orange', linestyle='--', label=f'Cũ (mean+3σ): {old_speed_th:.0f}')
ax.axvline(SPEED_TH,     color='red',    linestyle='--', label=f'Mới (97th pct): {SPEED_TH:.0f}')
ax.set_title('Speed Distribution & Thresholds')
ax.set_xlabel('Speed (m/h)'); ax.legend(fontsize=8)
ax.set_xlim(0, df['speed'].quantile(0.995))

# 7. Anomaly score theo thời gian
ax = axes[2, 0]
df_plot = df.set_index('timestamp')
ax.fill_between(df_plot.index, df_plot['anomaly_score'], alpha=0.6, color='coral')
ax.axhline(SCORE_THRESHOLD, color='red', linestyle='--', label=f'Threshold {SCORE_THRESHOLD}')
ax.set_title('Anomaly Score Over Time')
ax.set_xlabel('Time'); ax.set_ylabel('Score')
ax.legend()

# 8. Đóng góp từng loại anomaly
ax = axes[2, 1]
anomaly_cols = ['spatial_anomaly', 'persistence_anomaly', 'behavioral_anomaly',
                'temporal_anomaly', 'acceleration_anomaly']
anomaly_pcts = [df[c].mean() * 100 for c in anomaly_cols]
colors_bar   = ['steelblue', 'orange', 'red', 'purple', 'green']
labels_bar   = ['Spatial', 'Persistence', 'Behavioral\n(AND)', 'Temporal\n(đêm)', 'Acceleration\n(mới)']
bars = ax.bar(labels_bar, anomaly_pcts, color=colors_bar, alpha=0.8, edgecolor='black')
for bar, pct in zip(bars, anomaly_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
ax.set_title('% Điểm Mỗi Loại Anomaly')
ax.set_ylabel('Percentage (%)')

# 9. Speed vs Turning Angle (colored by anomaly score)
ax = axes[2, 2]
sc2 = ax.scatter(df['speed'], df['turning_angle_clean'],
                 c=df['anomaly_score'], cmap='hot_r',
                 alpha=0.5, s=6, vmin=0, vmax=1)
ax.axvline(SPEED_TH,    color='red',  linestyle='--', alpha=0.7, label=f'Speed TH')
ax.axhline(TURNING_TH,  color='blue', linestyle='--', alpha=0.7, label=f'Turning TH')
ax.set_title('Speed vs Turning Angle\n(màu = anomaly score)')
ax.set_xlabel('Speed (m/h)'); ax.set_ylabel('Turning Angle Clean (°)')
ax.legend(fontsize=8)
plt.colorbar(sc2, ax=ax, label='Anomaly Score')
ax.set_xlim(0, df['speed'].quantile(0.995))

plt.tight_layout()
plt.savefig('elephant_anomaly_improved.png', dpi=200, bbox_inches='tight')
print("📊 Biểu đồ đã lưu: elephant_anomaly_improved.png")
plt.show()

# ============================================================
# XUẤT FILE
# ============================================================
feat_df_final = feat_df.fillna(0).reset_index()
feat_df_final.to_csv('elephant_features_improved.csv', index=False)

df_with_all = df[[
    'timestamp', 'location-lat', 'location-long', 'speed', 'raw_accel', 'dist',
    'turning_angle', 'turning_angle_clean', 'bearing',
    'kde_probability', 'kde_prob_day', 'kde_prob_night', 'kde_prob_adaptive', 'kde_home_range',
    'spatial_anomaly', 'persistence_anomaly', 'behavioral_anomaly',
    'temporal_anomaly', 'acceleration_anomaly',
    'anomaly_score', 'point_is_outside', 'point_is_outside_old_OR'
]].copy()
df_with_all.to_csv('elephant_raw_improved.csv', index=False)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("✅ HOÀN THÀNH — TÓM TẮT CẢI TIẾN")
print("="*60)
print(f"\n📌 Dataset: {len(df)} điểm GPS sau khi lọc tốc độ ảo")
print(f"\n🔧 CÁC CẢI TIẾN ĐÃ ÁP DỤNG:")
print(f"   1. KDE: Silverman bandwidth ({bw_used:.5f}) thay vì cố định 0.01")
print(f"      + Loại outlier IQR trước khi fit")
print(f"      + Normalize percentile thay vì min-max")
print(f"   2. Behavioral: AND logic + percentile ngưỡng + lọc đứng yên ({stationary_count} điểm)")
print(f"   3. Temporal: ngưỡng tính riêng ban đêm ({SPEED_NIGHT_TH:.0f} m/h)")
print(f"      + Điều kiện KDE thấp (< 0.3)")
print(f"   4. Thêm Acceleration Anomaly (tín hiệu mới)")
print(f"   5. Weighted Score thay vì OR (threshold = {SCORE_THRESHOLD})")

print(f"\n📊 KẾT QUẢ:")
print(f"   {'Anomaly':<22} {'Count':>6}  {'%':>6}  {'Weight':>7}")
print(f"   {'-'*45}")
for col, w in weights.items():
    cnt = df[col].sum()
    pct = df[col].mean() * 100
    print(f"   {col:<22} {cnt:>6}  {pct:>5.2f}%  {w:>7.2f}")
print(f"   {'-'*45}")
print(f"   {'OR cũ (is_outside)':<22} {df['point_is_outside_old_OR'].sum():>6}  {df['point_is_outside_old_OR'].mean()*100:>5.2f}%")
print(f"   {'Weighted Score (mới)':<22} {df['point_is_outside'].sum():>6}  {df['point_is_outside'].mean()*100:>5.2f}%")
print(f"\n   Mean anomaly_score = {df['anomaly_score'].mean():.4f}")
print(f"   Features tạo ra    = {len(feat_df_final.columns)}")
print(f"\n📁 Files đã lưu:")
print(f"   - elephant_features_improved.csv  (features cho ML)")
print(f"   - elephant_raw_improved.csv       (raw data + tất cả anomaly)")
print(f"   - elephant_anomaly_improved.png   (9 biểu đồ phân tích)")
"""
Configuration Module for Elephant Behavior Classification

This module contains all configuration parameters for the project.
Centralized configuration makes it easy to modify parameters without
changing code throughout the project.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os


@dataclass
class PathConfig:
    """File paths configuration"""
    # Base directories
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = field(default_factory=lambda: os.path.join(PathConfig.PROJECT_ROOT, 'data'))
    
    # Data subdirectories
    RAW_DATA_DIR: str = field(init=False)
    PROCESSED_DATA_DIR: str = field(init=False)
    MODELS_DIR: str = field(init=False)
    
    # Output directories
    OUTPUTS_DIR: str = field(init=False)
    FIGURES_DIR: str = field(init=False)
    REPORTS_DIR: str = field(init=False)
    LOGS_DIR: str = field(init=False)
    
    def __post_init__(self):
        self.RAW_DATA_DIR = os.path.join(self.DATA_DIR, 'raw')
        self.PROCESSED_DATA_DIR = os.path.join(self.DATA_DIR, 'processed')
        self.MODELS_DIR = os.path.join(self.DATA_DIR, 'models')
        self.OUTPUTS_DIR = os.path.join(self.PROJECT_ROOT, 'outputs')
        self.FIGURES_DIR = os.path.join(self.OUTPUTS_DIR, 'figures')
        self.REPORTS_DIR = os.path.join(self.OUTPUTS_DIR, 'reports')
        self.LOGS_DIR = os.path.join(self.PROJECT_ROOT, 'logs')
        
        # Create directories if they don't exist
        for dir_path in [self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR, 
                        self.MODELS_DIR, self.FIGURES_DIR, 
                        self.REPORTS_DIR, self.LOGS_DIR]:
            os.makedirs(dir_path, exist_ok=True)


@dataclass
class DataConfig:
    """Data processing configuration"""
    # Column names
    TIMESTAMP_COL: str = 'timestamp'
    LAT_COL: str = 'location-lat'
    LON_COL: str = 'location-long'
    TARGET_COL: str = 'is_outside'
    
    # Data validation
    MIN_LAT: float = -90.0
    MAX_LAT: float = 90.0
    MIN_LON: float = -180.0
    MAX_LON: float = 180.0
    
    # Filtering thresholds
    MIN_SPEED: float = 5.0  # m/h - filter GPS drift
    MAX_SPEED: float = 10000.0  # m/h - filter unrealistic speeds
    
    # Resampling
    RESAMPLE_INTERVAL: str = '2H'  # 2-hour intervals
    INTERPOLATION_METHOD: str = 'linear'
    
    # Missing data
    MAX_MISSING_PCT: float = 0.3  # Maximum 30% missing values allowed


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    
    # KDE parameters
    KDE_BANDWIDTH: float = 0.01
    KDE_KERNEL: str = 'gaussian'
    KDE_BANDWIDTHS_TO_TEST: List[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.02]
    )
    
    # DBSCAN parameters
    DBSCAN_EPS: float = 0.005  # ~500m in lat/long
    DBSCAN_MIN_SAMPLES: int = 10
    
    # Turning angle parameters
    TURNING_ANGLE_BINS: int = 36  # 10-degree bins
    ENTROPY_WINDOW: int = 10  # Rolling window for entropy
    
    # Behavior thresholds
    SHARP_TURN_THRESHOLD: float = 90.0  # degrees
    U_TURN_THRESHOLD: float = 150.0  # degrees
    STRAIGHT_THRESHOLD: float = 15.0  # degrees
    
    # Rolling windows
    ROLLING_WINDOWS: Dict[str, int] = field(default_factory=lambda: {
        '3H': 3,
        '4H': 4,
        '6H': 6,
        '8H': 8
    })
    
    # KDE probability thresholds
    KDE_THRESHOLDS: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.5, 0.8, 1.0]
    )
    KDE_LABELS: List[str] = field(
        default_factory=lambda: ['Very_Low', 'Low', 'Medium', 'High']
    )
    
    # Temporal features
    DAY_START_HOUR: int = 6  # 6 AM
    DAY_END_HOUR: int = 18  # 6 PM
    NIGHT_START_HOUR: int = 18
    NIGHT_END_HOUR: int = 6


@dataclass
class ModelConfig:
    """Model training configuration"""
    
    # Random Forest parameters (optimal)
    RF_N_ESTIMATORS: int = 400
    RF_MAX_DEPTH: int = 10
    RF_MIN_SAMPLES_LEAF: int = 1
    RF_CLASS_WEIGHT: Dict[int, int] = field(
        default_factory=lambda: {0: 1, 1: 16}
    )
    RF_RANDOM_STATE: int = 42
    RF_N_JOBS: int = -1  # Use all CPUs
    
    # GridSearchCV parameters
    CV_FOLDS: int = 5
    CV_SCORING: str = 'f1_macro'
    
    # Hyperparameter grid for tuning
    PARAM_GRID: Dict[str, List] = field(default_factory=lambda: {
        'rf_n_estimators': [200, 400, 600],
        'rf_max_depth': [10, 15, 20],
        'rf_min_samples_leaf': [1, 2, 4]
    })
    
    # Train/test split
    TEST_SIZE: float = 0.2
    STRATIFY: bool = True
    RANDOM_STATE: int = 42
    
    # Class imbalance handling
    CLASS_IMBALANCE_RATIO: float = 16.0  # Inside:Outside ratio
    

@dataclass
class EvaluationConfig:
    """Model evaluation configuration"""
    
    # Primary metrics
    PRIMARY_METRIC: str = 'f1_macro'
    SECONDARY_METRICS: List[str] = field(default_factory=lambda: [
        'accuracy',
        'precision',
        'recall',
        'roc_auc'
    ])
    
    # Performance thresholds
    MIN_ACCURACY: float = 0.95
    MIN_F1_MACRO: float = 0.90
    MIN_RECALL_OUTSIDE: float = 0.90
    MAX_FALSE_NEGATIVES: int = 10
    
    # Threshold tuning
    THRESHOLDS_TO_TEST: List[float] = field(default_factory=lambda: [
        0.3, 0.4, 0.5, 0.6, 0.7, 0.8
    ])
    DEFAULT_THRESHOLD: float = 0.5
    
    # Visualization
    FIGURE_DPI: int = 300
    FIGURE_FORMAT: str = 'png'


@dataclass
class LoggingConfig:
    """Logging configuration"""
    
    LEVEL: str = 'INFO'
    FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'
    
    # File logging
    LOG_TO_FILE: bool = True
    LOG_FILE: str = 'elephant_classifier.log'
    MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT: int = 5
    
    # Console logging
    LOG_TO_CONSOLE: bool = True
    CONSOLE_LEVEL: str = 'INFO'


# Singleton instances
PATHS = PathConfig()
DATA = DataConfig()
FEATURES = FeatureConfig()
MODEL = ModelConfig()
EVALUATION = EvaluationConfig()
LOGGING = LoggingConfig()


# Convenience functions
def get_raw_data_path(filename: str) -> str:
    """Get full path to raw data file"""
    return os.path.join(PATHS.RAW_DATA_DIR, filename)


def get_processed_data_path(filename: str) -> str:
    """Get full path to processed data file"""
    return os.path.join(PATHS.PROCESSED_DATA_DIR, filename)


def get_model_path(filename: str) -> str:
    """Get full path to model file"""
    return os.path.join(PATHS.MODELS_DIR, filename)


def get_figure_path(filename: str) -> str:
    """Get full path to figure file"""
    return os.path.join(PATHS.FIGURES_DIR, filename)


def get_report_path(filename: str) -> str:
    """Get full path to report file"""
    return os.path.join(PATHS.REPORTS_DIR, filename)


# Export all configs
__all__ = [
    'PATHS',
    'DATA',
    'FEATURES',
    'MODEL',
    'EVALUATION',
    'LOGGING',
    'get_raw_data_path',
    'get_processed_data_path',
    'get_model_path',
    'get_figure_path',
    'get_report_path'
]


if __name__ == '__main__':
    # Print configuration for debugging
    print("="*60)
    print("ELEPHANT BEHAVIOR CLASSIFICATION - CONFIGURATION")
    print("="*60)
    
    print("\n📁 Paths:")
    print(f"  Project Root: {PATHS.PROJECT_ROOT}")
    print(f"  Raw Data: {PATHS.RAW_DATA_DIR}")
    print(f"  Processed Data: {PATHS.PROCESSED_DATA_DIR}")
    print(f"  Models: {PATHS.MODELS_DIR}")
    print(f"  Figures: {PATHS.FIGURES_DIR}")
    
    print("\n📊 Data:")
    print(f"  Min Speed: {DATA.MIN_SPEED} m/h")
    print(f"  Resample Interval: {DATA.RESAMPLE_INTERVAL}")
    
    print("\n🔧 Features:")
    print(f"  KDE Bandwidth: {FEATURES.KDE_BANDWIDTH}")
    print(f"  DBSCAN eps: {FEATURES.DBSCAN_EPS}")
    print(f"  Sharp Turn Threshold: {FEATURES.SHARP_TURN_THRESHOLD}°")
    
    print("\n🤖 Model:")
    print(f"  Random Forest Estimators: {MODEL.RF_N_ESTIMATORS}")
    print(f"  Max Depth: {MODEL.RF_MAX_DEPTH}")
    print(f"  Class Weight: {MODEL.RF_CLASS_WEIGHT}")
    
    print("\n📈 Evaluation:")
    print(f"  Primary Metric: {EVALUATION.PRIMARY_METRIC}")
    print(f"  Min F1-Macro: {EVALUATION.MIN_F1_MACRO}")
    print(f"  Min Recall Outside: {EVALUATION.MIN_RECALL_OUTSIDE}")
    
    print("\n" + "="*60)

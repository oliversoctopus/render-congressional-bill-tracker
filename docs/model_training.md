# Model Training Documentation

## Overview
The Predictive Bill Tracker uses a sophisticated two-phase machine learning system trained on **76,897 bills from 6 U.S. Congresses (113th-118th, 2013-2025)**. This comprehensive dataset provides robust historical patterns for accurate predictions.

## Training Pipeline

### 1. Data Preparation
- **Source**: `bills_6congress_training.csv` (preprocessed from Congress API data)
- **Features**: 40+ engineered features including:
  - Sponsor characteristics (party, count)
  - Bill metadata (title complexity, policy area)
  - Activity metrics (actions, committees, cosponsors)
  - Temporal indicators (days_active, momentum)
  - Congress-specific features

### 2. Target Definition
- **Viability**: Bills gaining legislative traction (16% positive rate)
  - Criteria: Passed, high activity, strong support, or key milestones
- **Passage**: Bills becoming law (2.7% overall, 16.7% of viable bills)

### 3. Model Architecture

#### Phase 1: Viability Models
Three time-aware stages with increasing feature complexity:
- **New Bill** (17 features): ROC-AUC 0.886
- **Early Stage** (25 features): ROC-AUC 0.994  
- **Progressive** (39 features): ROC-AUC 0.996

#### Phase 2: Passage Models
Trained only on viable bills for focused predictions:
- **New Bill**: ROC-AUC 0.812
- **Early Stage**: ROC-AUC 0.976
- **Progressive**: ROC-AUC 0.982

### 4. Training Process

```python
# Ensemble configuration
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=10,
    class_weight='balanced_subsample'
)

gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8
)

lr_model = LogisticRegression(
    penalty='l2',
    C=1.0,
    max_iter=1000,
    class_weight='balanced'
)

# Voting ensemble with calibration
ensemble = VotingClassifier(
    estimators=[('rf', rf_model), ('gb', gb_model), ('lr', lr_model)],
    voting='soft',
    weights=[0.4, 0.4, 0.2]
)

# Isotonic calibration for probability adjustment
calibrated_model = CalibratedClassifierCV(
    ensemble,
    method='isotonic',
    cv=3
)
```

### 5. Feature Selection
- **Method**: Mutual Information with SelectKBest
- **Top features**: 
  - Viability: early_activity, support_velocity, bipartisan_score
  - Passage: action_count, committee involvement, sponsor characteristics

### 6. Model Validation
- **Cross-validation**: 5-fold stratified CV
- **Metrics tracked**:
  - ROC-AUC (primary)
  - Precision/Recall
  - F1 Score
  - Accuracy

### 7. Model Storage
Optimized split component architecture:
```
models/
├── metadata.pkl                    # Encoders and metadata
├── viability_new_bill/
│   ├── rf_model.pkl               # Random Forest (largest)
│   ├── components.pkl             # GB, LR, scaler, selector
│   ├── ensemble_config.pkl        # Ensemble configuration
│   └── calibration.pkl           # Calibration data
├── viability_early_stage/...
└── passage_progressive/...
```

## Running Training

### Prerequisites
```bash
pip install scikit-learn pandas numpy joblib matplotlib
```

### Training Script
```bash
cd models
jupyter notebook model.ipynb
# Run all cells to train models
```

### Training Time
- Full training: ~30-45 minutes on standard hardware
- Includes all 6 model variants (3 viability + 3 passage)

## Model Performance Summary

| Model Stage | Viability ROC-AUC | Passage ROC-AUC |
|------------|-------------------|------------------|
| New Bill | 0.886 | 0.812 |
| Early Stage | 0.994 | 0.976 |
| Progressive | 0.996 | 0.982 |

## Key Improvements (6-Congress Version)
1. **Larger dataset**: 77K bills vs 16K (4.5x increase)
2. **Historical validation**: Patterns validated across 12+ years
3. **Congress-aware features**: Accounts for legislative period differences
4. **Better calibration**: More data enables robust probability adjustment
5. **Reduced overfitting**: Cross-congress validation ensures generalization

## Usage in Application
Models are loaded dynamically based on bill age:
- Day 1: New bill model
- Days 2-30: Early stage model  
- Day 30+: Progressive model

The application handles model loading, feature engineering, and prediction serving automatically.
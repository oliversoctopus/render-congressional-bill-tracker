# Congressional Bill Prediction Models - Documentation

**Version:** 5.0 (Temporal Snapshots)
**Training Date:** November 3, 2025
**Training Data:** 273,113 temporal snapshots from 76,854 unique bills (Congresses 113-118)

---

## Overview

This system uses a **two-phase prediction pipeline** with **time-aware model stages** to predict both the viability and passage probability of U.S. Congressional bills.

### Two-Phase Prediction

1. **Phase 1: Viability Prediction**
   Predicts whether a bill will receive significant legislative activity (committee hearings, floor votes, amendments)
   - Base rate: 10.0% of bills are viable

2. **Phase 2: Passage Prediction** (Only for viable bills)
   Predicts whether a viable bill will ultimately pass both chambers
   - Base rate: 80.5% of viable bills pass
   - Overall passage rate: 8.1% of all bills

### Time-Aware Model Stages

Each phase uses **three models** specialized for different bill ages:

| Stage | Bill Age | Features | Use Case |
|-------|----------|----------|----------|
| **New Bill** | Day 1 | 16 basic | Initial assessment at introduction |
| **Early Stage** | 2-30 days | 24 extended | After initial committee activity |
| **Progressive** | 30+ days | 41 comprehensive | Long-lived bills with substantial history |

---

## Model Architecture

### Calibrated Ensemble (All 6 Models)

Each model uses a **weighted voting ensemble** with **isotonic calibration**:

```
Ensemble = 0.4 × Random Forest + 0.4 × Gradient Boosting + 0.2 × Logistic Regression
                    ↓
            Isotonic Calibration
                    ↓
        Calibrated Probability (0-100%)
```

**Components:**
- **Random Forest**: 100 trees, captures complex feature interactions
- **Gradient Boosting**: 100 estimators, sequential error correction
- **Logistic Regression**: Linear baseline, interpretability
- **Isotonic Calibration**: Ensures probabilities match true frequencies

---

## Training Data - Temporal Snapshots

### Why Temporal Snapshots?

**Problem:** Previous models trained on bills at their final state, creating survivorship bias.

**Solution:** Generate multiple training examples per bill at different time points:
- Day 1 (introduction)
- Day 7, 14, 30 (early monitoring)
- Day 60, 90, 120, 180 (mid-stage)
- Day 365, 730 (long-term)

**Result:** 273,113 snapshots from 76,854 bills (3.6 avg per bill)

### Snapshot Distribution

| Days Active | Snapshots | % of Total |
|-------------|-----------|------------|
| ≤ 1 day     | 76,854    | 28.1%      |
| ≤ 7 days    | 114,632   | 42.0%      |
| ≤ 30 days   | 178,437   | 65.3%      |
| ≤ 90 days   | 224,377   | 82.2%      |
| ≤ 180 days  | 256,525   | 93.9%      |

### Hybrid Stage Filtering

To reduce survivorship bias, models filter training data:

- **New Bill**: All snapshots (no filter)
- **Early Stage**: Exclude snapshots <2 days
- **Progressive**:
  - Exclude snapshots <7 days
  - Down-weight 7-30 day snapshots (0.3× weight)
  - Full weight for 30+ day snapshots (1.0× weight)

This ensures the progressive model learns from bills that genuinely survived long-term, not just early snapshots.

---

## Model Performance

### Phase 1: Viability Prediction

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 Score |
|-------|---------|----------|-----------|--------|----------|
| New Bill | 0.9410 | 93.50% | 73.81% | 54.33% | 62.59% |
| Early Stage | 0.9644 | 93.09% | 75.11% | 53.21% | 62.25% |
| Progressive | 0.8664 | 90.84% | 75.11% | 36.73% | 49.33% |

**Note:** Progressive model has lower metrics because it only predicts on bills that survived 7+ days (harder task, pre-filtered set).

### Phase 2: Passage Prediction (Viable Bills Only)

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 Score |
|-------|---------|----------|-----------|--------|----------|
| New Bill | 0.9793 | 94.20% | 94.76% | 98.23% | 96.46% |
| Early Stage | 0.9644 | 92.09% | 93.80% | 96.66% | 95.21% |
| Progressive | 0.9437 | 90.97% | 93.01% | 96.15% | 94.55% |

**Note:** High accuracy reflects that 80.5% of viable bills pass (imbalanced dataset).

---

## Feature Engineering

### Base Features (16) - New Bill Model

Used on day 1 when only basic information is available:

**Bill Metadata:**
- `congress_numeric` - Congress number (113-120)
- `bill_type_encoded` - HR/S/HJRES/SJRES (encoded)
- `sponsor_party_encoded` - Democrat/Republican/Independent (encoded)
- `policy_area_encoded` - Policy category (encoded)

**Title Analysis:**
- `title_length` - Character count
- `title_word_count` - Word count
- `title_complexity` - Average word length

**Subject/Committee:**
- `subject_count` - Number of legislative subjects
- `committee_count` - Number of referred committees

**Temporal Features:**
- `month_introduced` - Month (1-12)
- `quarter_introduced` - Quarter (1-4)
- `is_election_year` - Congressional election year (0/1)

**Derived Features:**
- `log_title_length`, `log_subject_count`, `log_committee_count`

### Extended Features (+8) - Early Stage Model

Adds activity-based features (days 2-30):

- `days_active` - Days from introduction to last action
- `action_count` - Total legislative actions
- `cosponsor_count` - Number of cosponsors
- `activity_rate` - Actions per day
- `log_days_active`, `log_action_count`, `log_cosponsor_count`, `log_activity_rate`

### Progressive Features (+17) - Progressive Model

Adds advanced metrics (30+ days):

**Engagement Metrics:**
- `bipartisan_support` - Has both D and R cosponsors (0/1)
- `sponsor_cosponsor_ratio` - Relative engagement
- `cosponsorship_rate` - Cosponsors per day

**Activity Patterns:**
- `days_since_last_action` - Days since most recent action
- `action_momentum` - Recent activity trend
- `action_density` - Clustered vs spread activity

**Logarithmic Transforms:**
- `log_days_since_last_action`, `log_action_momentum`, etc.

**Total:** 41 features for progressive models

---

## Model Files Structure

Each model is saved in a directory with these components:

```
models/
├── viability_new_bill/
│   ├── rf_model.pkl           # Random Forest (56.4 MB)
│   ├── components.pkl          # GB + LR + scaler + selector (0.9 MB)
│   ├── ensemble_config.pkl     # Voting weights (1 KB)
│   └── calibration.pkl         # Isotonic calibrator (1 KB)
├── viability_early_stage/
│   ├── rf_model.pkl           # Compressed (100.4 MB → 37.2 MB)
│   ├── components.pkl
│   ├── ensemble_config.pkl
│   └── calibration.pkl
├── viability_progressive/
│   ├── rf_model.pkl           # Compressed (92.2 MB → 38.0 MB)
│   ├── components.pkl
│   ├── ensemble_config.pkl
│   └── calibration.pkl
├── passage_new_bill/
├── passage_early_stage/
├── passage_progressive/
└── metadata.pkl               # Training metadata + label encoders
```

**Compression:** Models >80 MB are compressed using `joblib compress=9` (lossless, 30-50% size reduction)

---

## How Predictions Work

### 1. User Selects a Bill

The app (`src/app.py`) fetches bill data from Congress.gov API:
- Bill metadata (title, sponsor, type, introduced date)
- Actions timeline (committee referrals, votes, amendments)
- Cosponsors list with dates

### 2. Calculate Bill Age

```python
days_active = (last_action_date - introduced_date).days
days_since_last_action = (current_date - last_action_date).days
```

### 3. Select Appropriate Model Stage

```python
if days_active == 1:
    model_stage = 'new_bill'
elif days_active <= 30:
    model_stage = 'early_stage'
else:
    model_stage = 'progressive'
```

### 4. Engineer Features

Extract features matching the model stage (16/24/41 features)

### 5. Run Two-Phase Prediction

**Phase 1: Viability**
```python
viability_prob = viability_models[model_stage].predict(features)

if viability_prob >= 50%:
    # Phase 2: Passage (only for viable bills)
    passage_prob = passage_models[model_stage].predict(features)
else:
    passage_prob = 0  # Non-viable bills don't pass
```

### 6. Display Results

- Viability probability: 0-100%
- Passage probability: 0-100% (0% if not viable)
- Confidence intervals
- Feature importance

---

## Key Improvements Over v4.0

### 1. Temporal Snapshots
- **v4.0:** 76,854 training examples (1 per bill at final state)
- **v5.0:** 273,113 training examples (3.6 per bill at multiple time points)
- **Impact:** Eliminates survivorship bias, models learn from bills at various stages

### 2. Hybrid Stage Filtering
- **v4.0:** Progressive model trained on all bills including quick deaths
- **v5.0:** Progressive model excludes <7 day bills, down-weights 7-30 day bills
- **Impact:** Better predictions for long-lived bills

### 3. New Feature: days_since_last_action
- **v4.0:** No abandonment detection
- **v5.0:** Tracks time since last activity, warns users about stalled bills
- **Impact:** Identifies "zombie bills" with no recent movement

### 4. Fixed Temporal Bugs
- **v4.0:** days_active = (now - introduction) ❌ WRONG
- **v5.0:** days_active = (last_action - introduction) ✅ CORRECT
- **Impact:** Accurate feature calculations matching training data

### 5. Removed Deprecated Features
- **v4.0:** Used `is_recent_congress` (congress >= 117)
- **v5.0:** Removed (becomes stale over time)
- **Impact:** Models remain accurate as new congresses begin

---

## Usage Guidelines

### When to Trust Predictions

**High Confidence:**
- Day 1 viability predictions (94.1% ROC-AUC)
- Passage predictions for clearly viable bills (97.9% ROC-AUC at day 1)
- Bills with typical activity patterns

**Lower Confidence:**
- Progressive viability predictions (86.6% ROC-AUC) - harder task
- Bills from ended congresses (educational mode only)
- Bills with unusual characteristics (symbolic resolutions, commemorations)

### Interpreting Probabilities

- **Viability < 30%:** Likely to die in committee, minimal activity
- **Viability 30-70%:** Uncertain, monitor for committee action
- **Viability > 70%:** Strong chance of substantive activity

- **Passage < 40%:** (if viable) Likely to stall or fail vote
- **Passage 40-60%:** (if viable) Uncertain outcome
- **Passage > 60%:** (if viable) Strong chance of becoming law

### Educational Mode (Ended Congresses)

For bills from congresses that have ended (113-118), the app shows:
- "What-if" predictions at historical dates
- Actual outcome comparison
- Educational disclaimer

These are for learning only, as the bill can no longer pass.

---

## Technical Details

### Training Configuration

**Environment:**
- Python 3.13
- scikit-learn (latest)
- pandas, numpy, joblib

**Hyperparameters:**
- Random Forest: 100 trees, max_depth=None, min_samples_split=2
- Gradient Boosting: 100 estimators, learning_rate=0.1, max_depth=3
- Logistic Regression: max_iter=1000, L2 regularization
- Ensemble Weights: [0.4, 0.4, 0.2]

**Cross-Validation:**
- Stratified K-Fold, k=5
- Trained on 80% of data
- Tested on held-out 20%

**Feature Selection:**
- SelectKBest with mutual information
- Top 15 features (new_bill)
- Top 20 features (early_stage, progressive)

### File Sizes

| Model | RF Model | Components | Total |
|-------|----------|------------|-------|
| viability_new_bill | 56.4 MB | 0.9 MB | 57.3 MB |
| viability_early_stage | 37.2 MB* | 0.9 MB | 38.1 MB |
| viability_progressive | 38.0 MB* | 0.8 MB | 38.8 MB |
| passage_new_bill | 24.7 MB | 0.8 MB | 25.5 MB |
| passage_early_stage | 22.7 MB | 0.8 MB | 23.5 MB |
| passage_progressive | 21.5 MB | 0.8 MB | 22.3 MB |

*Compressed with `joblib compress=9`

**Total:** ~206 MB (all 6 models)

---

## Future Improvements

### Potential Enhancements

1. **Incremental Updates:** Retrain only on new congresses (119+) instead of full retraining
2. **Additional Features:**
   - Bill sponsor seniority/committee positions
   - Current political composition (majority party control)
   - Similar bill history (text similarity to past bills)
3. **Real-Time Updates:** Daily model updates with latest bill actions
4. **Explainability:** SHAP values for individual predictions
5. **Uncertainty Quantification:** Confidence intervals per prediction

### Maintenance

- **As needed:** Model retraining, bug fixes, feature additions, data updates

---

## References

**Data Source:** Congress.gov via GovInfo Bulk Downloads
**Model Training Script:** `models/train_models.py`
**Web Application:** `src/app.py` (Streamlit)

**For questions or issues:** See GitHub repository

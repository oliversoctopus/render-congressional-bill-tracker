# Model Architecture and Training

## Two-Phase Prediction System

### Phase 1: Viability Prediction
Determines if a bill will gain traction (16.0% of bills are viable across 6 congresses)

### Phase 2: Passage Prediction
For viable bills only, predicts likelihood of becoming law (16.7% of viable bills pass)

## Time-Aware Model Stages

### 1. New Bill Model (Day 1)
**Features**: 17 base features
- Sponsor party and characteristics
- Title complexity
- Policy area
- Initial bipartisan support
- Introduction timing
- Congress-specific indicators

**Performance (6-Congress Dataset)**:
- ROC-AUC: 0.886
- Accuracy: 88.8%
- Cross-validation ROC-AUC: 0.888 (±0.008)

### 2. Early Stage Model (2-30 days)
**Features**: 25 extended features (base + 8 additional)
- Cosponsor growth rate
- Early momentum indicators  
- Support velocity
- Initial activity patterns
- Early activity flag

**Performance (6-Congress Dataset)**:
- ROC-AUC: 0.994
- Accuracy: 98.5%
- Cross-validation ROC-AUC: 0.994 (±0.002)

### 3. Progressive Model (30+ days)
**Features**: 39 comprehensive features
- Full legislative history
- Committee engagement metrics
- Activity rate normalization
- Momentum indicators
- Sustained activity patterns

**Performance (6-Congress Dataset)**:
- ROC-AUC: 0.996
- Accuracy: 99.1%
- Cross-validation ROC-AUC: 0.996 (±0.001)

## Ensemble Architecture
Each stage uses a calibrated ensemble of:
- **Random Forest** (40% weight): 300 trees, balanced sampling, max_depth=20
- **Gradient Boosting** (40% weight): 200 trees, learning rate 0.05, max_depth=5
- **Logistic Regression** (20% weight): L2 regularization for stability

## Key Innovations
1. **Isotonic Calibration**: Adjusts probabilities for realistic predictions using 3-fold CV
2. **Mutual Information Feature Selection**: Identifies top 20 most predictive features per stage
3. **Congress-Aware Features**: Includes congress_numeric and is_recent_congress for temporal context
4. **Confidence Intervals**: Bootstrap-based uncertainty quantification
5. **Optimized Component Storage**: Split models into <100MB files for efficient loading

## Training Process
- **Dataset**: 76,897 bills from 6 congresses (113th-118th)
- **Cross-validation**: 5-fold stratified CV
- **Class balancing**: Addresses 2.7% positive class imbalance  
- **Feature scaling**: StandardScaler normalization
- **Model persistence**: Saved as optimized split components:
  - RF models saved separately (largest components)
  - GB, LR, scalers, selectors combined in components.pkl
  - Ensemble configs and calibration data in separate files

## Performance by Congress
Consistent performance across different congresses demonstrates model robustness:
- 113th: 18.9% viability, 3.2% passage
- 114th: 19.2% viability, 3.3% passage
- 115th: 19.6% viability, 3.8% passage
- 116th: 15.0% viability, 2.4% passage
- 117th: 14.4% viability, 2.4% passage
- 118th: 12.4% viability, 1.7% passage

## Model Sizes (Optimized)
- Viability models: 57.3MB (new), 29.2MB (early), 14.2MB (progressive)
- Passage models: 15.6MB (new), 9.1MB (early), 6.7MB (progressive)
- Total: ~132MB across all models
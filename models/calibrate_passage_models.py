"""
Calibrate existing passage models (post-training fix)

Issue: Passage models have 80.5% positive rate but weren't calibrated because
training script only calibrated models with <30% positive rate.

This script applies isotonic calibration to the existing passage models.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CALIBRATING PASSAGE MODELS (POST-TRAINING FIX)")
print("="*70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load temporal snapshots
print("\nLoading temporal snapshots dataset...")
df = pd.read_csv('data/bills_temporal_snapshots.csv')
print(f"Loaded {len(df)} snapshots")

# Filter to viable bills only (passage models only predict on viable bills)
df_viable = df[df['viable'] == True].copy()
print(f"Filtered to {len(df_viable)} viable bill snapshots")

# Load metadata
metadata_package = joblib.load('models/metadata.pkl')
label_encoders = metadata_package['label_encoders']

print(f"\nPositive rate (passed): {df_viable['passed'].mean():.1%}")

# Engineer features for all viable bills (only those not already in temporal snapshots)
print("\nEngineering features...")

# Encode categorical variables that need encoding
df_viable['sponsor_party_encoded'] = df_viable['sponsor_party'].map(
    lambda x: label_encoders['party'].transform([x])[0] if x in label_encoders['party'].classes_ else 0
)

df_viable['policy_area_encoded'] = df_viable['policy_area'].map(
    lambda x: label_encoders['policy'].transform([x])[0] if x in label_encoders['policy'].classes_ else 0
)

# All other features should already exist in the temporal snapshots CSV
# Just ensure numeric types and fill NaNs
for col in df_viable.columns:
    if df_viable[col].dtype in ['float64', 'int64']:
        df_viable[col] = df_viable[col].fillna(0).replace([np.inf, -np.inf], 0)

print("✅ Features prepared")

# Models to calibrate
passage_models = ['passage_new_bill', 'passage_early_stage', 'passage_progressive']

for model_name in passage_models:
    print("\n" + "="*70)
    print(f"CALIBRATING: {model_name}")
    print("="*70)

    stage = model_name.replace('passage_', '')
    model_dir = f'models/{model_name}'

    # Load existing model components
    print("Loading existing model components...")
    rf_model = joblib.load(f'{model_dir}/rf_model.pkl')
    components = joblib.load(f'{model_dir}/components.pkl')
    ensemble_config = joblib.load(f'{model_dir}/ensemble_config.pkl')

    features = components['metadata']['features']
    selected_features = components['metadata']['selected_features']
    scaler = components['scaler']
    selector = components['selector']

    # Reconstruct ensemble
    print("Reconstructing ensemble...")
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('gb', components['gb_model']),
            ('lr', components['lr_model'])
        ],
        voting='soft',
        weights=ensemble_config['weights']
    )

    # Set fitted attributes
    ensemble.estimators_ = [rf_model, components['gb_model'], components['lr_model']]
    ensemble.classes_ = rf_model.classes_
    ensemble.le_ = None

    # Filter data by stage
    print(f"Filtering data for {stage} stage...")
    if stage == 'new_bill':
        stage_df = df_viable[df_viable['days_active'] <= 1].copy()
    elif stage == 'early_stage':
        stage_df = df_viable[(df_viable['days_active'] >= 2) & (df_viable['days_active'] <= 30)].copy()
    else:  # progressive
        stage_df = df_viable[df_viable['days_active'] >= 7].copy()

    print(f"Stage data: {len(stage_df)} snapshots")
    print(f"Positive rate: {stage_df['passed'].mean():.1%}")

    if len(stage_df) < 100:
        print(f"⚠️  WARNING: Not enough data for {stage}, skipping calibration")
        continue

    # Prepare features
    print("Preparing features...")
    X = stage_df[features].copy()
    y = stage_df['passed'].values

    # Handle missing values
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    # Scale and select features
    X_scaled = scaler.transform(X)
    X_selected = selector.transform(X_scaled)

    print(f"Feature shape: {X_selected.shape}")

    # Apply sample weights if progressive model
    sample_weight = None
    if stage == 'progressive':
        print("Applying hybrid sample weights...")
        sample_weight = np.ones(len(stage_df))

        # Down-weight 7-30 day snapshots
        early_mask = (stage_df['days_active'] >= 7) & (stage_df['days_active'] < 30)
        sample_weight[early_mask] = 0.3

        print(f"  Full weight (30+ days): {(~early_mask).sum()} snapshots")
        print(f"  Reduced weight (7-30 days): {early_mask.sum()} snapshots")

    # Calibrate the ensemble
    print("Applying isotonic calibration...")
    calibrated_ensemble = CalibratedClassifierCV(
        ensemble,
        method='isotonic',
        cv=3
    )

    # Fit calibration
    calibrated_ensemble.fit(X_selected, y, sample_weight=sample_weight)

    print("✅ Calibration complete!")

    # Test predictions before/after calibration
    print("\nComparing predictions (before vs after calibration):")

    # Sample 10 random examples
    sample_indices = np.random.choice(len(X_selected), min(10, len(X_selected)), replace=False)

    uncalibrated_probs = ensemble.predict_proba(X_selected[sample_indices])[:, 1]
    calibrated_probs = calibrated_ensemble.predict_proba(X_selected[sample_indices])[:, 1]
    actual_labels = y[sample_indices]

    print("\n  Uncalib  Calib  Actual")
    print("  -------  -----  ------")
    for i in range(len(sample_indices)):
        print(f"  {uncalibrated_probs[i]:.1%}   {calibrated_probs[i]:.1%}   {actual_labels[i]}")

    # Save the calibrated model by updating the ensemble config
    print("\nSaving calibrated model...")

    # Save the calibrated ensemble as the main model
    # We need to save the entire CalibratedClassifierCV object
    calibration_data = {
        'method': 'isotonic',
        'cv': 3,
        'calibrators': []
    }

    # Extract calibrators from the fitted model
    if hasattr(calibrated_ensemble, 'calibrated_classifiers_'):
        for cal_clf in calibrated_ensemble.calibrated_classifiers_:
            if hasattr(cal_clf, 'calibrator'):
                calibration_data['calibrators'].append({
                    'calibrator': cal_clf.calibrator
                })

    joblib.dump(calibration_data, f'{model_dir}/calibration.pkl')

    # Update metadata to indicate calibration
    components['metadata']['is_calibrated'] = True
    components['metadata']['calibration_date'] = datetime.now().isoformat()
    joblib.dump(components, f'{model_dir}/components.pkl')

    print(f"✅ Saved calibration data to {model_dir}/calibration.pkl")
    print(f"✅ Updated components.pkl with calibration metadata")

print("\n" + "="*70)
print("CALIBRATION COMPLETE FOR ALL PASSAGE MODELS")
print("="*70)
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nCalibrated models:")
for model_name in passage_models:
    print(f"  - {model_name}")
print("\nNote: The app needs to be updated to use the calibration data.")
print("="*70)

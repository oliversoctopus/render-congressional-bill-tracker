"""
Train Time-Aware Dual Models on Temporal Snapshots

This script trains 6 models (viability/passage × 3 stages) using temporal snapshot data.
The temporal snapshots eliminate survivorship bias by training on bills at various stages.

Input: data/bills_temporal_snapshots.csv
Output: models/{viability|passage}_{new_bill|early_stage|progressive}/

Usage:
    python train_models.py

Expected Runtime: 4-6 hours (training 6 models on 273K snapshots)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TIME-AWARE MODEL TRAINING - TEMPORAL SNAPSHOTS")
print("="*70)

# Load temporal snapshot data
dataset_path = '../data/bills_temporal_snapshots.csv'
print(f"Loading dataset from: {dataset_path}")

df = pd.read_csv(dataset_path)
print(f"Loaded {len(df):,} snapshots from {df['bill_id'].nunique():,} unique bills")
print(f"Passed: {df['passed'].sum():,} ({df['passed'].mean()*100:.1f}%)")
print(f"Viable: {df['viable'].sum():,} ({df['viable'].mean()*100:.1f}%)")

# Viability definition
print(f"\nSnapshot distribution by days_active:")
for threshold in [1, 7, 30, 90, 180]:
    count = len(df[df['days_active'] <= threshold])
    print(f"  ≤{threshold:3d} days: {count:6,} ({count/len(df)*100:.1f}%)")

# Label encoding for categorical features
print("\nEncoding categorical features...")
le_party = LabelEncoder()
le_policy = LabelEncoder()

# Handle unknown categories
df['sponsor_party'] = df['sponsor_party'].fillna('Unknown')
df['policy_area'] = df['policy_area'].fillna('Unknown')

all_parties = list(df['sponsor_party'].unique()) + ['Unknown']
le_party.fit(all_parties)
df['sponsor_party_encoded'] = le_party.transform(df['sponsor_party'])

all_policies = list(df['policy_area'].unique()) + ['Unknown']
le_policy.fit(all_policies)
df['policy_area_encoded'] = le_policy.transform(df['policy_area'])

# Define feature sets for each stage
base_features = [
    'sponsor_party_encoded',
    'sponsor_count',
    'original_cosponsor_count',
    'month_introduced',
    'quarter_introduced',
    'is_election_year',
    'title_length',
    'title_word_count',
    'title_complexity',
    'subject_count',
    'policy_area_encoded',
    'party_balance',
    'party_dominance',
    'bipartisan_score',
    'has_bipartisan_support',
    'congress_numeric',
]

extended_features = base_features + [
    'cosponsor_count',
    'total_sponsors',
    'is_fresh',
    'support_velocity',
    'cosponsor_growth',
    'dem_total',
    'rep_total',
    'early_activity'
]

progressive_features = extended_features + [
    'days_active',
    'log_days_active',
    'sqrt_days_active',
    'activity_rate',
    'normalized_activity',
    'sustained_activity',
    'is_active',
    'is_stale',
    'committee_count',
    'has_committee',
    'multi_committee',
    'committee_density',
    'action_count',
    'bipartisan_momentum',
    'committee_activity',
    'days_since_last_action',
    'log_days_since_last_action'
]

# Verify features exist
print("\nVerifying features...")
for feature_set_name, features in [('base', base_features), ('extended', extended_features), ('progressive', progressive_features)]:
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"⚠️ Missing in {feature_set_name}: {missing}")

# Remove missing features
base_features = [f for f in base_features if f in df.columns]
extended_features = [f for f in extended_features if f in df.columns]
progressive_features = [f for f in progressive_features if f in df.columns]

print(f"\nFinal feature counts:")
print(f"- Base (new_bill): {len(base_features)}")
print(f"- Extended (early_stage): {len(extended_features)}")
print(f"- Progressive (30+ days): {len(progressive_features)}")


def save_model_with_compression(model, path):
    """Save model, using compression if it would exceed 80 MB"""
    import os

    # Save temporarily to check size
    temp_path = f"{path}.temp"
    joblib.dump(model, temp_path)
    size_mb = os.path.getsize(temp_path) / (1024 * 1024)

    if size_mb > 80:
        # Use compression to stay under 100 MB GitHub limit
        joblib.dump(model, path, compress=9)
        compressed_size = os.path.getsize(path) / (1024 * 1024)
        print(f"    Saved with compression: {size_mb:.1f} MB → {compressed_size:.1f} MB")
        os.unlink(temp_path)
    else:
        # No compression needed
        os.replace(temp_path, path)
        print(f"    Saved without compression: {size_mb:.1f} MB")


def train_stage_model(df, target_col, model_name, stage_name, features, use_calibration=True):
    """Train model for a specific stage with hybrid filtering"""

    print(f"\n{'='*70}")
    print(f"TRAINING {model_name.upper()} - {stage_name.upper()} MODEL")
    print(f"{'='*70}")

    # HYBRID STAGE FILTERING
    if stage_name == 'new_bill':
        # All snapshots (they all start here)
        stage_df = df.copy()
        sample_weights = None
        print("Using all snapshots")

    elif stage_name == 'early_stage':
        # Exclude very quick deaths (<2 days)
        stage_df = df[df['days_active'] >= 2].copy()
        sample_weights = None
        print(f"Filtered to ≥2 days: {len(stage_df):,} snapshots")

    elif stage_name == 'progressive':
        # HYBRID: Exclude <7 days, down-weight 7-30 days
        stage_df = df[df['days_active'] >= 7].copy()

        # Create sample weights
        sample_weights = np.ones(len(stage_df))
        sample_weights[stage_df['days_active'] < 30] = 0.3

        print(f"Filtered to ≥7 days: {len(stage_df):,} snapshots")
        print(f"  - 7-30 days (0.3 weight): {(stage_df['days_active'] < 30).sum():,}")
        print(f"  - 30+ days (1.0 weight): {(stage_df['days_active'] >= 30).sum():,}")

    # Filter available features
    available_features = [f for f in features if f in stage_df.columns]
    print(f"Using {len(available_features)} features")

    # Prepare data
    X = stage_df[available_features].fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    y = stage_df[target_col]

    # Check class distribution
    pos_rate = y.mean()
    print(f"Positive class rate: {pos_rate:.1%}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Split sample weights if they exist
    if sample_weights is not None:
        # Create a series with the same index as stage_df for proper alignment
        weights_series = pd.Series(sample_weights, index=stage_df.index)
        weights_train = weights_series.loc[X_train.index].values
        weights_test = weights_series.loc[X_test.index].values
    else:
        weights_train = None
        weights_test = None

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # Feature selection
    k_features = min(20, len(available_features) - 1)
    selector = SelectKBest(mutual_info_classif, k=k_features)
    selector.fit(X_train_scaled, y_train)

    selected_indices = selector.get_support(indices=True)
    selected_features = [available_features[i] for i in selected_indices]

    X_train_selected = X_train_scaled[selected_features]
    X_test_selected = X_test_scaled[selected_features]

    print(f"Selected top {k_features} features")

    # Create ensemble models
    print("Training ensemble models...")

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )

    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )

    lr_model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )

    # Train individual models with sample weights
    rf_model.fit(X_train_selected, y_train, sample_weight=weights_train)
    gb_model.fit(X_train_selected, y_train, sample_weight=weights_train)
    lr_model.fit(X_train_selected, y_train, sample_weight=weights_train)

    # Create voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('gb', gb_model),
            ('lr', lr_model)
        ],
        voting='soft',
        weights=[0.4, 0.4, 0.2]
    )

    # Train ensemble
    ensemble.fit(X_train_selected, y_train, sample_weight=weights_train)

    # Calibrate if needed
    if use_calibration and pos_rate < 0.3:
        print("Calibrating probabilities...")
        calibrated_ensemble = CalibratedClassifierCV(
            ensemble,
            method='isotonic',
            cv=3
        )
        calibrated_ensemble.fit(X_train_selected, y_train, sample_weight=weights_train)
        final_model = calibrated_ensemble
    else:
        final_model = ensemble

    # Make predictions
    y_pred_proba = final_model.predict_proba(X_test_selected)[:, 1]
    threshold = 0.5
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\nPerformance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    # Cross-validation
    print("Running cross-validation...")
    cv_params = {'sample_weight': weights_train} if weights_train is not None else {}
    cv_scores = cross_val_score(
        ensemble, X_train_selected, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc',
        params=cv_params
    )
    print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # Feature importance
    if hasattr(rf_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': selected_features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTop 5 features:")
        for _, row in importance_df.head(5).iterrows():
            print(f"  - {row['feature']}: {row['importance']:.4f}")

    # Store model components
    model_data = {
        'model': final_model,
        'ensemble': ensemble,
        'rf_model': rf_model,
        'gb_model': gb_model,
        'lr_model': lr_model,
        'scaler': scaler,
        'selector': selector,
        'features': available_features,
        'selected_features': selected_features,
        'threshold': threshold,
        'performance': {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_roc_auc': cv_scores.mean(),
            'cv_std': cv_scores.std()
        },
        'is_calibrated': use_calibration and pos_rate < 0.3
    }

    return model_data


def save_model_components(model_data, model_type, stage_name):
    """Save model as optimized split components"""

    # Create directory
    model_dir = f'../models/{model_type}_{stage_name}'
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n--- Saving {model_type}_{stage_name} ---")

    # Save RF model separately (largest component)
    rf_file = f'{model_dir}/rf_model.pkl'
    save_model_with_compression(model_data['rf_model'], rf_file)

    # Combine smaller components
    small_components = {
        'gb_model': model_data['gb_model'],
        'lr_model': model_data['lr_model'],
        'scaler': model_data['scaler'],
        'selector': model_data['selector'],
        'metadata': {
            'features': model_data['features'],
            'selected_features': model_data['selected_features'],
            'threshold': model_data['threshold'],
            'performance': model_data['performance'],
            'is_calibrated': model_data['is_calibrated']
        }
    }

    components_file = f'{model_dir}/components.pkl'
    joblib.dump(small_components, components_file)
    components_size = os.path.getsize(components_file) / (1024 * 1024)
    print(f"    components.pkl: {components_size:.1f} MB")

    # Save ensemble config
    ensemble_config = {
        'voting': 'soft',
        'weights': [0.4, 0.4, 0.2],
        'estimator_names': ['rf', 'gb', 'lr']
    }
    joblib.dump(ensemble_config, f'{model_dir}/ensemble_config.pkl')

    # Save calibration if exists
    if model_data['is_calibrated']:
        calibrated_model = model_data['model']
        calibration_data = {
            'method': 'isotonic',
            'cv': 3,
            'calibrators': []
        }

        if hasattr(calibrated_model, 'calibrated_classifiers_'):
            for cal_clf in calibrated_model.calibrated_classifiers_:
                if hasattr(cal_clf, 'calibrator_'):
                    calibration_data['calibrators'].append({
                        'calibrator': cal_clf.calibrator_
                    })

        joblib.dump(calibration_data, f'{model_dir}/calibration.pkl')


# Train viability models
print("\n" + "="*70)
print("PHASE 1: VIABILITY PREDICTION")
print("="*70)

feature_sets = {
    'new_bill': base_features,
    'early_stage': extended_features,
    'progressive': progressive_features
}

viability_models = {}

# First two models already trained, skip them
print("\n[SKIPPED] viability_new_bill - already trained")
print("[SKIPPED] viability_early_stage - already trained")

# Only train progressive model (the one that failed)
for stage_name, features in list(feature_sets.items())[2:]:  # Start from index 2 (progressive)
    model_data = train_stage_model(df, 'viable', 'viability', stage_name, features)
    viability_models[stage_name] = model_data
    save_model_components(model_data, 'viability', stage_name)

# Train passage models on viable snapshots only
viable_snapshots = df[df['viable'] == 1].copy()
print(f"\n" + "="*70)
print(f"PHASE 2: PASSAGE PREDICTION")
print(f"Training on {len(viable_snapshots):,} viable snapshots ({viable_snapshots['passed'].mean():.1%} passed)")
print("="*70)

passage_models = {}
for stage_name, features in feature_sets.items():
    model_data = train_stage_model(viable_snapshots, 'passed', 'passage', stage_name, features)
    passage_models[stage_name] = model_data
    save_model_components(model_data, 'passage', stage_name)

# Save metadata
metadata = {
    'training_date': datetime.now().isoformat(),
    'dataset_size': len(df),
    'unique_bills': df['bill_id'].nunique(),
    'viable_rate': df['viable'].mean(),
    'passage_rate': df['passed'].mean(),
    'model_version': '7.0-temporal-snapshots',
    'improvements': [
        'Trained on temporal snapshots (273K examples from 77K bills)',
        'Hybrid stage filtering to reduce survivorship bias',
        'Added days_since_last_action feature',
        'Removed is_recent_congress (deprecated)',
        'Models trained on bills at multiple time points'
    ]
}

joblib.dump({
    'metadata': metadata,
    'label_encoders': {
        'party': le_party,
        'policy': le_policy
    }
}, '../models/metadata.pkl')

print("\n" + "="*70)
print("MODEL TRAINING COMPLETE!")
print(f"Trained 6 models on {len(df):,} temporal snapshots")
print(f"Models saved in models/ directory")
print("="*70)

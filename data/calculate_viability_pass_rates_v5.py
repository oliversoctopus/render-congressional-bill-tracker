"""
VIABILITY PASS RATE ANALYZER v5.0
Analyzes the training dataset to compute actual pass rates for different viability score ranges
Uses v5.0 models with temporal snapshots
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("VIABILITY PASS RATE ANALYSIS v5.0")
print("="*70)

# Load the temporal snapshots dataset
dataset_path = 'data/bills_temporal_snapshots.csv'
print(f"Loading dataset from: {dataset_path}")
df = pd.read_csv(dataset_path)
print(f"Loaded {len(df)} temporal snapshots")

# Get unique bills only (use the last snapshot for each bill)
print("\nExtracting unique bills (using last snapshot for each)...")
df_sorted = df.sort_values(['bill_id', 'days_active'])
df_unique = df_sorted.groupby('bill_id').last().reset_index()
print(f"Extracted {len(df_unique)} unique bills")

# Load metadata and encoders
print("\nLoading model metadata...")
metadata_package = joblib.load('models/metadata.pkl')
label_encoders = metadata_package['label_encoders']

# Function to load a model stage
def load_model_stage(model_type, stage):
    """Load a single model stage with all components"""
    model_dir = f'models/{model_type}_{stage}'

    print(f"  Loading {model_type}_{stage}...")

    # Load components
    rf_model = joblib.load(f'{model_dir}/rf_model.pkl')
    components = joblib.load(f'{model_dir}/components.pkl')
    ensemble_config = joblib.load(f'{model_dir}/ensemble_config.pkl')
    calibration_data = joblib.load(f'{model_dir}/calibration.pkl')

    # Extract calibrators
    calibrators = None
    if calibration_data['calibrators']:
        calibrators = [cal['calibrator'] for cal in calibration_data['calibrators']]

    return {
        'rf_model': rf_model,
        'gb_model': components['gb_model'],
        'lr_model': components['lr_model'],
        'scaler': components['scaler'],
        'selector': components['selector'],
        'features': components['metadata']['features'],
        'selected_features': components['metadata']['selected_features'],
        'calibrators': calibrators,
        'weights': ensemble_config['weights']
    }

# Load all viability models
print("\nLoading viability models...")
viability_models = {
    'new_bill': load_model_stage('viability', 'new_bill'),
    'early_stage': load_model_stage('viability', 'early_stage'),
    'progressive': load_model_stage('viability', 'progressive')
}
print("Models loaded successfully!")

# Feature engineering
print("\nPreparing features...")

def safe_get_column(df, col_name, default_value):
    """Safely get column with fallback to default"""
    if col_name in df.columns:
        return df[col_name].fillna(default_value)
    else:
        return default_value

# Encode only the missing categorical variables
df_unique['sponsor_party_encoded'] = df_unique['sponsor_party'].map(
    lambda x: label_encoders['party'].transform([x])[0] if x in label_encoders['party'].classes_ else 0
)

df_unique['policy_area_encoded'] = df_unique['policy_area'].map(
    lambda x: label_encoders['policy'].transform([x])[0] if x in label_encoders['policy'].classes_ else 0
)

# All other features already exist in temporal snapshots CSV
# Just ensure no NaNs or infinities
for col in df_unique.columns:
    if df_unique[col].dtype in ['float64', 'int64']:
        df_unique[col] = df_unique[col].fillna(0).replace([np.inf, -np.inf], 0)

print("Features prepared!")

# Calculate viability scores for each unique bill
print("\nCalculating viability scores for all unique bills...")
print("Using progressive model for all bills (most comprehensive)")
viability_scores = []

# Use progressive model for all bills (it has the most comprehensive features)
model = viability_models['progressive']

for idx, row in df_unique.iterrows():
    if idx % 5000 == 0:
        print(f"Processing bill {idx}/{len(df_unique)}...")

    try:
        # Prepare features for this bill
        X = pd.DataFrame([row[model['features']].fillna(0)])
        X = X.replace([np.inf, -np.inf], 0)

        # Scale and select
        X_scaled = model['scaler'].transform(X)
        X_selected = model['selector'].transform(X_scaled)

        # Get predictions from all models in ensemble
        rf_prob = model['rf_model'].predict_proba(X_selected)[0, 1]
        gb_prob = model['gb_model'].predict_proba(X_selected)[0, 1]
        lr_prob = model['lr_model'].predict_proba(X_selected)[0, 1]

        # Ensemble prediction (weighted average)
        ensemble_prob = (model['weights'][0] * rf_prob +
                        model['weights'][1] * gb_prob +
                        model['weights'][2] * lr_prob)

        # Apply calibration (average across CV folds)
        if model['calibrators']:
            calibrated_probs = []
            for calibrator in model['calibrators']:
                try:
                    cal_prob = calibrator.transform([ensemble_prob])[0]
                    calibrated_probs.append(cal_prob)
                except:
                    calibrated_probs.append(ensemble_prob)
            calibrated_prob = np.mean(calibrated_probs) if calibrated_probs else ensemble_prob
        else:
            calibrated_prob = ensemble_prob

        viability_scores.append({
            'bill_id': row['bill_id'],
            'congress': row['congress'],
            'viability_score': calibrated_prob,
            'passed': row['passed'],
            'days_active': row['days_active']
        })

    except Exception as e:
        print(f"Error processing bill {idx}: {str(e)}")
        # Use average viability as fallback
        viability_scores.append({
            'bill_id': row['bill_id'],
            'congress': row['congress'],
            'viability_score': 0.049,  # Average viability from v5.0
            'passed': row['passed'],
            'days_active': row['days_active']
        })

# Convert to DataFrame
viability_df = pd.DataFrame(viability_scores)
print(f"\nCalculated viability scores for {len(viability_df)} unique bills")

# Analyze pass rates by viability score ranges
print("\n" + "="*70)
print("ANALYZING PASS RATES BY VIABILITY SCORE")
print("="*70)

# Define viability score bins (same as v4.0)
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0]
bin_labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%',
              '60-70%', '70-80%', '80-90%', '90-95%', '95-98%', '98-99%', '99-100%']

# Add viability bins
viability_df['viability_bin'] = pd.cut(viability_df['viability_score'],
                                        bins=bins,
                                        labels=bin_labels,
                                        include_lowest=True)

# Calculate pass rates for each bin
pass_rate_data = []

print("\nPass Rates by Viability Range:")
print("-" * 70)
for bin_label in bin_labels:
    bin_data = viability_df[viability_df['viability_bin'] == bin_label]

    if len(bin_data) > 0:
        pass_rate = bin_data['passed'].mean() * 100
        count = len(bin_data)
        passed_count = bin_data['passed'].sum()

        pass_rate_data.append({
            'viability_range': bin_label,
            'bill_count': count,
            'passed_count': int(passed_count),
            'pass_rate': pass_rate,
            'min_viability': bins[bin_labels.index(bin_label)],
            'max_viability': bins[bin_labels.index(bin_label) + 1]
        })

        print(f"{bin_label:>12}: {count:>6,} bills, {passed_count:>5} passed ({pass_rate:>5.1f}%)")

# Create pass rate lookup table
pass_rate_df = pd.DataFrame(pass_rate_data)

# Also calculate more granular data for smooth interpolation
print("\nCalculating fine-grained pass rates (100 bins)...")
fine_bins = np.linspace(0, 1, 101)
fine_bin_centers = (fine_bins[:-1] + fine_bins[1:]) / 2

fine_pass_rates = []
for i in range(len(fine_bins) - 1):
    mask = ((viability_df['viability_score'] >= fine_bins[i]) &
            (viability_df['viability_score'] < fine_bins[i+1]))
    bin_data = viability_df[mask]

    if len(bin_data) >= 10:  # Only include bins with at least 10 bills
        pass_rate = bin_data['passed'].mean() * 100
        fine_pass_rates.append({
            'viability_score': fine_bin_centers[i],
            'pass_rate': pass_rate,
            'bill_count': len(bin_data)
        })

fine_pass_rate_df = pd.DataFrame(fine_pass_rates)
print(f"Created {len(fine_pass_rate_df)} fine-grained bins")

# Save the results
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save the detailed pass rate table
output_path = 'data/viability_pass_rates.csv'
pass_rate_df.to_csv(output_path, index=False)
print(f"✅ Saved {output_path}")

# Save the fine-grained pass rate data
output_path_fine = 'data/viability_pass_rates_fine.csv'
fine_pass_rate_df.to_csv(output_path_fine, index=False)
print(f"✅ Saved {output_path_fine}")

# Save a summary statistics file
summary_stats = {
    'version': '5.0',
    'total_bills': len(viability_df),
    'overall_pass_rate': viability_df['passed'].mean() * 100,
    'average_viability': viability_df['viability_score'].mean(),
    'viability_std': viability_df['viability_score'].std(),
    'analysis_date': datetime.now().isoformat(),
    'model_used': 'viability_progressive',
    'congress_breakdown': viability_df.groupby('congress').agg({
        'passed': ['count', 'sum', 'mean'],
        'viability_score': 'mean'
    }).to_dict()
}

summary_path = 'data/viability_analysis_summary_v5.pkl'
joblib.dump(summary_stats, summary_path)
print(f"✅ Saved {summary_path}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print(f"Analyzed {len(viability_df)} unique bills")
print(f"Overall pass rate: {viability_df['passed'].mean() * 100:.2f}%")
print(f"Average viability score: {viability_df['viability_score'].mean():.4f} ({viability_df['viability_score'].mean()*100:.2f}%)")
print(f"Viability std dev: {viability_df['viability_score'].std():.4f}")
print("\nFiles saved:")
print("  - data/viability_pass_rates.csv (pass rates by bin)")
print("  - data/viability_pass_rates_fine.csv (fine-grained data)")
print("  - data/viability_analysis_summary_v5.pkl (summary statistics)")
print("="*70)

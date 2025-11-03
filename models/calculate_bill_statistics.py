"""
Calculate unique bill statistics from temporal snapshots dataset
Used to update v5.0 model documentation and website statistics
"""
import pandas as pd
import numpy as np

print("="*70)
print("CALCULATING UNIQUE BILL STATISTICS - v5.0")
print("="*70)

# Load temporal snapshots
df = pd.read_csv('../data/bills_temporal_snapshots.csv', low_memory=False)
print(f"\nLoaded {len(df):,} temporal snapshots")

# Get unique bills only (using the final snapshot for each bill)
# Final snapshot = max days_active for each bill_id
bill_final_states = df.loc[df.groupby('bill_id')['days_active'].idxmax()]
print(f"Unique bills: {len(bill_final_states):,}")

# Calculate unique bill statistics
total_bills = len(bill_final_states)
viable_bills = bill_final_states['viable'].sum()
passed_bills = bill_final_states['passed'].sum()

viable_rate = viable_bills / total_bills
passage_rate = passed_bills / total_bills
viable_pass_rate = passed_bills / viable_bills if viable_bills > 0 else 0

print("\n" + "="*70)
print("UNIQUE BILL STATISTICS")
print("="*70)
print(f"Total unique bills: {total_bills:,}")
print(f"Viable bills: {viable_bills:,} ({viable_rate:.1%})")
print(f"Passed bills: {passed_bills:,} ({passage_rate:.1%})")
print(f"Pass rate among viable bills: {viable_pass_rate:.1%}")
print(f"Pass rate among non-viable bills: {(bill_final_states[bill_final_states['viable']==0]['passed'].sum() / (total_bills - viable_bills)):.1%}")

# Snapshot statistics for comparison
print("\n" + "="*70)
print("SNAPSHOT STATISTICS (for comparison)")
print("="*70)
snapshot_viable_rate = df['viable'].mean()
snapshot_passage_rate = df['passed'].mean()
snapshot_viable_pass_rate = df[df['viable']==1]['passed'].mean()

print(f"Total snapshots: {len(df):,}")
print(f"Viable snapshots: {df['viable'].sum():,} ({snapshot_viable_rate:.1%})")
print(f"Passed snapshots: {df['passed'].sum():,} ({snapshot_passage_rate:.1%})")
print(f"Pass rate among viable snapshots: {snapshot_viable_pass_rate:.1%}")

# Calculate viability bins for "what proportion of bills with X viability actually passed"
print("\n" + "="*70)
print("VIABILITY ANALYSIS - BY UNIQUE BILLS")
print("="*70)

# Since we don't have predicted viability in the snapshots,
# let's calculate by actual viability status
print("\nActual outcomes by viability status:")
print(f"  Non-viable bills that passed: {bill_final_states[bill_final_states['viable']==0]['passed'].sum()}")
print(f"  Viable bills that passed: {bill_final_states[bill_final_states['viable']==1]['passed'].sum()}")

# Analyze by congress to see variation
print("\n" + "="*70)
print("STATISTICS BY CONGRESS")
print("="*70)

for congress in sorted(bill_final_states['congress'].unique()):
    congress_bills = bill_final_states[bill_final_states['congress'] == congress]
    c_total = len(congress_bills)
    c_viable = congress_bills['viable'].sum()
    c_passed = congress_bills['passed'].sum()
    c_viable_rate = c_viable / c_total
    c_passage_rate = c_passed / c_total
    c_viable_pass = congress_bills[congress_bills['viable']==1]['passed'].sum()
    c_viable_pass_rate = c_viable_pass / c_viable if c_viable > 0 else 0

    print(f"\nCongress {int(congress)}:")
    print(f"  Bills: {c_total:,}")
    print(f"  Viable: {c_viable:,} ({c_viable_rate:.1%})")
    print(f"  Passed: {c_passed:,} ({c_passage_rate:.1%})")
    print(f"  Viable → Passed: {c_viable_pass_rate:.1%}")

# Save by-congress statistics
congress_stats = []
for congress in sorted(bill_final_states['congress'].unique()):
    congress_bills = bill_final_states[bill_final_states['congress'] == congress]
    c_total = len(congress_bills)
    c_viable = congress_bills['viable'].sum()
    c_passed = congress_bills['passed'].sum()
    c_viable_pass = congress_bills[congress_bills['viable']==1]['passed'].sum()

    congress_stats.append({
        'congress': int(congress),
        'total_bills': int(c_total),
        'viable_bills': int(c_viable),
        'passed_bills': int(c_passed),
        'viable_rate': float(c_viable / c_total),
        'passage_rate': float(c_passed / c_total),
        'viable_pass_rate': float(c_viable_pass / c_viable if c_viable > 0 else 0)
    })

# Save summary statistics
summary = {
    'unique_bill_stats': {
        'total_bills': int(total_bills),
        'viable_bills': int(viable_bills),
        'passed_bills': int(passed_bills),
        'viable_rate': float(viable_rate),
        'passage_rate': float(passage_rate),
        'viable_pass_rate': float(viable_pass_rate)
    },
    'snapshot_stats': {
        'total_snapshots': int(len(df)),
        'viable_snapshots': int(df['viable'].sum()),
        'passed_snapshots': int(df['passed'].sum()),
        'viable_rate': float(snapshot_viable_rate),
        'passage_rate': float(snapshot_passage_rate),
        'viable_pass_rate': float(snapshot_viable_pass_rate)
    },
    'by_congress': congress_stats
}

import json
with open('../models/bill_statistics_v5.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Also save a CSV for easy viewing
stats_df = pd.DataFrame([
    {
        'metric': 'Total Bills (Unique)',
        'count': total_bills,
        'percentage': '100%'
    },
    {
        'metric': 'Viable Bills (Unique)',
        'count': viable_bills,
        'percentage': f'{viable_rate:.1%}'
    },
    {
        'metric': 'Passed Bills (Unique)',
        'count': passed_bills,
        'percentage': f'{passage_rate:.1%}'
    },
    {
        'metric': 'Viable → Passed (Unique)',
        'count': passed_bills,
        'percentage': f'{viable_pass_rate:.1%}'
    },
    {
        'metric': '---',
        'count': '---',
        'percentage': '---'
    },
    {
        'metric': 'Total Snapshots',
        'count': len(df),
        'percentage': '100%'
    },
    {
        'metric': 'Viable Snapshots',
        'count': df['viable'].sum(),
        'percentage': f'{snapshot_viable_rate:.1%}'
    },
    {
        'metric': 'Passed Snapshots',
        'count': df['passed'].sum(),
        'percentage': f'{snapshot_passage_rate:.1%}'
    },
    {
        'metric': 'Viable → Passed (Snapshots)',
        'count': df[df['viable']==1]['passed'].sum(),
        'percentage': f'{snapshot_viable_pass_rate:.1%}'
    }
])

stats_df.to_csv('../models/bill_statistics_v5.csv', index=False)

print("\n" + "="*70)
print("Statistics saved to:")
print("  - models/bill_statistics_v5.json (detailed)")
print("  - models/bill_statistics_v5.csv (summary)")
print("="*70)

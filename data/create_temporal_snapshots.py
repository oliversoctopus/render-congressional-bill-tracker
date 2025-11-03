"""
Create Temporal Snapshots for Model Training

This script generates training examples at multiple time points for each bill.
Instead of one row per bill (at final state), we create snapshots at:
- Day 1 (introduction)
- Day 7, 14, 30, 60, 90, 120, 180 (if bill survived that long)
- Final state

This eliminates survivorship bias and allows models to learn from bills
at various stages of development.

Input: data/temporal/*.csv (from parse_bulk_data.py)
Output: data/bills_temporal_snapshots.csv

Usage:
    python create_temporal_snapshots.py

Expected Runtime: 2-4 hours (processing ~77K bills × multiple snapshots each)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import time

# Configuration
TEMPORAL_DIR = Path("temporal")
OUTPUT_FILE = Path("bills_temporal_snapshots.csv")
PROGRESS_FILE = Path("snapshot_progress.json")

# Snapshot intervals (in days after introduction)
SNAPSHOT_DAYS = [1, 7, 14, 30, 60, 90, 120, 180, 365, 730]

# Congress end dates (for calculating outcomes)
CONGRESS_END_DATES = {
    113: datetime(2015, 1, 3),
    114: datetime(2017, 1, 3),
    115: datetime(2019, 1, 3),
    116: datetime(2021, 1, 3),
    117: datetime(2023, 1, 3),
    118: datetime(2025, 1, 3),
}


def load_progress():
    """Load snapshot creation progress"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'processed_bills': [], 'last_save_time': None}


def save_progress(progress):
    """Save snapshot creation progress"""
    progress['last_save_time'] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def load_data():
    """Load the temporal CSV files"""
    print("Loading temporal data...")

    bills_df = pd.read_csv(TEMPORAL_DIR / 'bills_metadata.csv')
    # low_memory=False prevents dtype warnings for large files
    actions_df = pd.read_csv(TEMPORAL_DIR / 'actions_timeline.csv', low_memory=False)
    cosponsors_df = pd.read_csv(TEMPORAL_DIR / 'cosponsors_timeline.csv', low_memory=False)

    # Convert dates
    bills_df['introduced_date'] = pd.to_datetime(bills_df['introduced_date'])
    bills_df['latest_action_date'] = pd.to_datetime(bills_df['latest_action_date'])

    actions_df['date'] = pd.to_datetime(actions_df['date'])
    cosponsors_df['sponsored_date'] = pd.to_datetime(cosponsors_df['sponsored_date'])

    print(f"✅ Loaded {len(bills_df):,} bills")
    print(f"✅ Loaded {len(actions_df):,} actions")
    print(f"✅ Loaded {len(cosponsors_df):,} cosponsors")

    return bills_df, actions_df, cosponsors_df


def determine_outcome(bill_row, congress_end_date):
    """
    Determine if bill passed or failed based on latest action.

    Returns: (passed, viable)
    """
    latest_action = str(bill_row.get('latest_action_text', '')).lower()

    # Check if passed
    passed_keywords = ['became public law', 'became private law', 'presented to president']
    passed = any(keyword in latest_action for keyword in passed_keywords)

    # Determine viability (bills that gained traction)
    viable = passed  # Passed bills are always viable

    # Bills with significant activity are viable even if they didn't pass
    if not viable:
        action_keywords = ['reported', 'committee discharged', 'passed house', 'passed senate',
                          'ordered reported', 'markup', 'hearing held']
        viable = any(keyword in latest_action for keyword in action_keywords)

    return (int(passed), int(viable))


def create_snapshot(bill_row, snapshot_date, actions_df, cosponsors_df, bill_actions, bill_cosponsors):
    """
    Create a snapshot of bill state at a specific date.

    Returns: dict with features as they existed on snapshot_date
    """
    bill_id = bill_row['bill_id']
    introduced_date = bill_row['introduced_date']

    # Filter to data up to snapshot_date
    snapshot_actions = bill_actions[bill_actions['date'] <= snapshot_date]
    snapshot_cosponsors = bill_cosponsors[bill_cosponsors['sponsored_date'] <= snapshot_date]

    # Calculate temporal metrics
    days_active = (snapshot_date - introduced_date).days
    days_active = max(1, min(days_active, 730))  # Clip to match training

    # Calculate days_since_last_action
    if len(snapshot_actions) > 0:
        last_action_date = snapshot_actions['date'].max()
        days_since_last_action = (snapshot_date - last_action_date).days
    else:
        days_since_last_action = 0

    # Action metrics
    action_count = len(snapshot_actions)

    # Committee info from actions
    committee_actions = snapshot_actions[snapshot_actions['committees'].notna() & (snapshot_actions['committees'] != '')]
    unique_committees = set()
    for committees_str in committee_actions['committees']:
        if pd.notna(committees_str) and committees_str:
            unique_committees.update([c.strip() for c in str(committees_str).split(',') if c.strip()])
    committee_count = len(unique_committees)

    # Cosponsor metrics
    total_cosponsors = len(snapshot_cosponsors)
    original_cosponsors = len(snapshot_cosponsors[snapshot_cosponsors['is_original'] == 'true'])

    # Party breakdown of cosponsors
    dem_cosponsors = len(snapshot_cosponsors[snapshot_cosponsors['party'] == 'D'])
    rep_cosponsors = len(snapshot_cosponsors[snapshot_cosponsors['party'] == 'R'])
    ind_cosponsors = len(snapshot_cosponsors[snapshot_cosponsors['party'] == 'I'])

    # Sponsor party (from bill metadata)
    sponsor_party = bill_row.get('sponsor_party', 'Unknown')
    dem_sponsors = 1 if sponsor_party == 'D' else 0
    rep_sponsors = 1 if sponsor_party == 'R' else 0
    ind_sponsors = 1 if sponsor_party == 'I' else 0

    # Calculate derived features
    total_sponsors = 1 + total_cosponsors  # Primary sponsor + cosponsors
    dem_total = dem_sponsors + dem_cosponsors
    rep_total = rep_sponsors + rep_cosponsors

    party_balance = (dem_total - rep_total) / (total_sponsors + 1)
    party_dominance = abs(party_balance)
    bipartisan_score = 1 - party_dominance
    has_bipartisan_support = int((dem_total > 0) and (rep_total > 0))

    # Activity metrics
    activity_rate = action_count / max(days_active, 1)
    normalized_activity = action_count / np.log1p(days_active)
    early_activity = action_count / (min(days_active, 30) + 1)
    sustained_activity = action_count / (min(days_active, 180) + 1)

    # Temporal flags
    is_fresh = int(days_active <= 30)
    is_active = int(days_active <= 90)
    is_stale = int(days_active > 180)

    # Committee metrics
    has_committee = int(committee_count > 0)
    multi_committee = int(committee_count >= 2)
    committee_density = committee_count / max(days_active / 30, 1)

    # Support growth
    cosponsor_growth = (total_cosponsors - original_cosponsors) / max(days_active / 30, 1)
    support_velocity = total_sponsors / np.sqrt(max(days_active, 1))

    # Interaction features
    bipartisan_momentum = bipartisan_score * normalized_activity
    committee_activity = committee_count * activity_rate

    # Text features
    title = str(bill_row.get('title', ''))
    title_length = len(title)
    title_word_count = len(title.split())
    title_complexity = title_length / max(title_word_count, 1)

    # Subject features
    subject_count = bill_row.get('subject_count', 0)
    if pd.isna(subject_count):
        subject_count = 0

    # Temporal features based on introduction date
    month_introduced = introduced_date.month
    quarter_introduced = (introduced_date.month - 1) // 3 + 1
    is_election_year = int(introduced_date.year % 2 == 0)

    # Congress features
    congress = int(bill_row['congress'])

    # Create feature dict
    snapshot = {
        # Identifiers
        'bill_id': bill_id,
        'snapshot_date': snapshot_date.strftime('%Y-%m-%d'),
        'congress': congress,
        'bill_type': bill_row['bill_type'],
        'bill_number': bill_row['bill_number'],

        # Temporal metrics
        'days_active': days_active,
        'days_since_last_action': days_since_last_action,
        'log_days_active': np.log1p(days_active),
        'sqrt_days_active': np.sqrt(days_active),
        'log_days_since_last_action': np.log1p(days_since_last_action),

        # Action metrics
        'action_count': action_count,
        'activity_rate': activity_rate,
        'normalized_activity': normalized_activity,
        'early_activity': early_activity,
        'sustained_activity': sustained_activity,

        # Committee metrics
        'committee_count': committee_count,
        'has_committee': has_committee,
        'multi_committee': multi_committee,
        'committee_density': committee_density,
        'committee_activity': committee_activity,

        # Sponsor/cosponsor metrics
        'cosponsor_count': total_cosponsors,
        'original_cosponsor_count': original_cosponsors,
        'sponsor_count': 1,
        'total_sponsors': total_sponsors,
        'cosponsor_growth': cosponsor_growth,
        'support_velocity': support_velocity,

        # Party metrics
        'sponsor_party': sponsor_party,
        'dem_sponsors': dem_sponsors,
        'rep_sponsors': rep_sponsors,
        'ind_sponsors': ind_sponsors,
        'dem_cosponsors': dem_cosponsors,
        'rep_cosponsors': rep_cosponsors,
        'ind_cosponsors': ind_cosponsors,
        'dem_total': dem_total,
        'rep_total': rep_total,
        'party_balance': party_balance,
        'party_dominance': party_dominance,
        'bipartisan_score': bipartisan_score,
        'has_bipartisan_support': has_bipartisan_support,
        'bipartisan_momentum': bipartisan_momentum,

        # Temporal flags
        'is_fresh': is_fresh,
        'is_active': is_active,
        'is_stale': is_stale,

        # Text features
        'title': title[:200],  # Truncate for storage
        'title_length': title_length,
        'title_word_count': title_word_count,
        'title_complexity': title_complexity,

        # Subject features
        'subject_count': int(subject_count),
        'policy_area': bill_row.get('policy_area', 'Unknown'),

        # Introduction temporal features
        'month_introduced': month_introduced,
        'quarter_introduced': quarter_introduced,
        'is_election_year': is_election_year,

        # Congress features
        'congress_numeric': congress,
    }

    return snapshot


def generate_snapshots_for_bill(bill_row, actions_df, cosponsors_df):
    """
    Generate all snapshots for a single bill.

    Returns: list of snapshot dicts
    """
    bill_id = bill_row['bill_id']
    introduced_date = bill_row['introduced_date']
    latest_action_date = bill_row['latest_action_date']
    congress = int(bill_row['congress'])

    # Get all actions and cosponsors for this bill
    bill_actions = actions_df[actions_df['bill_id'] == bill_id].copy()
    bill_cosponsors = cosponsors_df[cosponsors_df['bill_id'] == bill_id].copy()

    # Determine final outcome
    congress_end_date = CONGRESS_END_DATES.get(congress, datetime(2099, 1, 1))
    passed, viable = determine_outcome(bill_row, congress_end_date)

    snapshots = []

    # Calculate bill's actual lifespan (introduction to last action)
    bill_lifespan_days = (latest_action_date - introduced_date).days

    # Create snapshots at predefined intervals
    for days in SNAPSHOT_DAYS:
        snapshot_date = introduced_date + timedelta(days=days)

        # Only create snapshot if it's within the bill's actual lifespan
        # or if it's the first snapshot (day 1)
        if days == 1 or (snapshot_date <= latest_action_date and snapshot_date <= congress_end_date):
            snapshot = create_snapshot(
                bill_row,
                snapshot_date,
                actions_df,
                cosponsors_df,
                bill_actions,
                bill_cosponsors
            )

            # Add outcome labels (from final state)
            snapshot['passed'] = passed
            snapshot['viable'] = viable

            snapshots.append(snapshot)

    # Always include final snapshot (at latest action date)
    if latest_action_date > introduced_date + timedelta(days=1):
        final_snapshot = create_snapshot(
            bill_row,
            latest_action_date,
            actions_df,
            cosponsors_df,
            bill_actions,
            bill_cosponsors
        )
        final_snapshot['passed'] = passed
        final_snapshot['viable'] = viable
        snapshots.append(final_snapshot)

    return snapshots


def main():
    """Main snapshot generation orchestration"""
    print("="*70)
    print("CREATING TEMPORAL SNAPSHOTS FOR MODEL TRAINING")
    print("="*70)
    print(f"Snapshot intervals: {SNAPSHOT_DAYS} days")
    print(f"Output: {OUTPUT_FILE}")
    print("="*70)
    print()

    # Load data
    bills_df, actions_df, cosponsors_df = load_data()

    # Load progress
    progress = load_progress()
    processed_bills = set(progress.get('processed_bills', []))

    if processed_bills:
        print(f"\nResuming: {len(processed_bills):,} bills already processed")

    # Filter to unprocessed bills
    unprocessed_bills = bills_df[~bills_df['bill_id'].isin(processed_bills)]
    print(f"\nProcessing {len(unprocessed_bills):,} bills...")
    print()

    # Generate snapshots
    all_snapshots = []
    start_time = time.time()

    for idx, (_, bill_row) in enumerate(unprocessed_bills.iterrows(), 1):
        bill_id = bill_row['bill_id']

        # Generate snapshots for this bill
        snapshots = generate_snapshots_for_bill(bill_row, actions_df, cosponsors_df)
        all_snapshots.extend(snapshots)

        # Track progress
        processed_bills.add(bill_id)

        # Progress update every 100 bills
        if idx % 100 == 0 or idx == len(unprocessed_bills):
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (len(unprocessed_bills) - idx) / rate if rate > 0 else 0

            print(f"  {idx:,}/{len(unprocessed_bills):,} bills ({idx/len(unprocessed_bills)*100:.1f}%) | "
                  f"Rate: {rate:.1f} bills/sec | "
                  f"ETA: {remaining/60:.1f} min | "
                  f"Snapshots: {len(all_snapshots):,}", end='\r')

        # Save progress every 1000 bills
        if idx % 1000 == 0:
            progress['processed_bills'] = list(processed_bills)
            save_progress(progress)

            # Also save snapshots incrementally
            if all_snapshots:
                temp_df = pd.DataFrame(all_snapshots)
                temp_df.to_csv(f"{OUTPUT_FILE}.temp", index=False)

    print()  # Final newline

    # Save final results
    print(f"\n{'='*70}")
    print(f"SAVING SNAPSHOTS")
    print(f"{'='*70}")

    if all_snapshots:
        snapshots_df = pd.DataFrame(all_snapshots)

        # Sort by bill_id and snapshot_date
        snapshots_df = snapshots_df.sort_values(['bill_id', 'snapshot_date'])

        # Save
        snapshots_df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Saved {len(snapshots_df):,} snapshots to {OUTPUT_FILE}")

        # Statistics
        print(f"\nSnapshot Statistics:")
        print(f"  - Total snapshots: {len(snapshots_df):,}")
        print(f"  - Unique bills: {snapshots_df['bill_id'].nunique():,}")
        print(f"  - Avg snapshots per bill: {len(snapshots_df) / snapshots_df['bill_id'].nunique():.1f}")
        print(f"  - Passed bills: {snapshots_df['passed'].sum():,} ({snapshots_df['passed'].mean()*100:.1f}%)")
        print(f"  - Viable bills: {snapshots_df['viable'].sum():,} ({snapshots_df['viable'].mean()*100:.1f}%)")

        # Distribution by days_active
        print(f"\nSnapshot Distribution by Bill Age:")
        for threshold in [1, 7, 14, 30, 60, 90, 180, 365]:
            count = len(snapshots_df[snapshots_df['days_active'] <= threshold])
            pct = count / len(snapshots_df) * 100
            print(f"  - ≤{threshold:3d} days: {count:6,} ({pct:5.1f}%)")

        # Clean up temp file
        temp_file = Path(f"{OUTPUT_FILE}.temp")
        if temp_file.exists():
            temp_file.unlink()
    else:
        print("⚠️ No snapshots generated")

    # Save final progress
    progress['processed_bills'] = list(processed_bills)
    progress['completed'] = True
    save_progress(progress)

    elapsed = (time.time() - start_time) / 60  # minutes

    print(f"\n{'='*70}")
    print(f"SNAPSHOT GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {elapsed:.1f} minutes")
    print(f"Output: {OUTPUT_FILE.absolute()}")
    print(f"\nNext step: Update and run models/model.ipynb to train models on temporal snapshots")
    print(f"{'='*70}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSnapshot generation interrupted by user.")
        print("Progress has been saved. Re-run this script to resume.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("\nProgress has been saved. Re-run this script to resume.")

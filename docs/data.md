# Data Documentation

## Data Sources
- **Primary Source**: Congress.gov API (v3) – Official U.S. Congress data.
  - Endpoints: `/bill/{congress}/{bill_type}`, `/bill/{congress}/{bill_type}/{bill_id}` (details like sponsors, actions, committees).
  - API Key: Required (stored in .env); rate-limited.
- **Scope**: **6 Congresses (113th-118th, 2013-2025)**; all HR, S, HJRES, SJRES bills (~77,000 bills total).
- **Ethical Note**: Public domain data; no scraping—API compliant. No personal data beyond public sponsor info.

## Data Acquisition
- Notebook: `data/extract_data.ipynb`
  - Fetches bills in batches (limit=250) for each congress.
  - Handles retries, caching in congress-specific cache directories.
  - Extracts: Bill ID, title, sponsors/cosponsors, actions, committees, subjects, timelines.
  - **Multi-congress support**: Iterates through congresses 113-118.
- Checkpointing: Resumes if interrupted, maintains separate caches per congress.

## Preprocessing
- Notebook: `data/preprocess_6_congress.ipynb`
  - **Combines data from all 6 congresses** into unified dataset.
  - Identifies outcomes based on final status in each congress.
  - Features engineering:
    - Party balances and bipartisan scores
    - Activity rates and momentum indicators
    - Congress-specific features (congress_numeric, is_recent_congress)
    - Temporal features (early_activity, sustained_activity)
  - Outcomes: Passed (1), Failed (0); 'Viable' target for traction (16% of bills).
- Saved Files:
  - `bills_6congress_training.csv`: All bills with outcomes (~77K rows, 40+ columns).
  - `viability_pass_rates.csv`: Pass rates by viability score ranges.
  - `viability_pass_rates_fine.csv`: Fine-grained viability-to-passage correlation.
- Preprocessing Steps: 
  - Date parsing and normalization across congresses
  - Missing value handling with congress-aware defaults
  - Label encoding for party/policy areas
  - Derived features including days_active, activity_rate, support_velocity

## Data Structure
- **Key columns**: 
  - Identifiers: bill_id, congress, bill_type
  - Outcomes: passed, viable, failure_reason
  - Sponsor info: sponsor_party, cosponsor_count, bipartisan_score
  - Activity metrics: action_count, committee_count, days_active
  - Temporal: introduced_date, latest_action_date, is_stale
- **Congress distribution**:
  - 113th: 9,091 bills (3.2% passed)
  - 114th: 10,233 bills (3.3% passed)
  - 115th: 11,421 bills (3.8% passed)
  - 116th: 14,345 bills (2.4% passed)
  - 117th: 15,242 bills (2.4% passed)
  - 118th: 16,565 bills (1.7% passed)
- **Size**: Main dataset ~50MB; supporting files ~5MB each.
- **Privacy**: No sensitive data; aggregated sponsor stats only.

## Viability Analysis
- **Definition**: Bills gaining traction through committee activity, cosponsorship, or legislative progress.
- **Criteria**: 
  - Passed bills automatically viable
  - High activity (6+ actions with committee involvement)
  - Strong support (30+ cosponsors with activity)
  - Significant milestones reached
- **Distribution**: 16% of bills classified as viable, with 16.7% of viable bills passing.

For usage: Load via pandas in app/model scripts with appropriate preprocessing.
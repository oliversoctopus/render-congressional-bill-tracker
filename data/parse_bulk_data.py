"""
Parse Bulk Bill Status XML Data

This script parses the downloaded BILLSTATUS XML files and extracts:
- Actions with dates (for temporal timeline)
- Cosponsors with join dates (for temporal timeline)
- Bill metadata

Input: data/bulk_downloads/{congress}/{bill_type}/*.xml
Output: data/temporal/*.csv

Usage:
    python parse_bulk_data.py

Expected Runtime: 1-2 hours (parsing ~77,000 XML files)
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import time

# Configuration
CONGRESSES = [113, 114, 115, 116, 117, 118]
BILL_TYPES = ['hr', 's', 'hjres', 'sjres']
BULK_DOWNLOADS_DIR = Path("bulk_downloads")
OUTPUT_DIR = Path("temporal")
PROGRESS_FILE = OUTPUT_DIR / "parsing_progress.json"


def load_progress():
    """Load parsing progress from checkpoint file"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_progress(progress):
    """Save parsing progress to checkpoint file"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def parse_bill_xml(xml_path):
    """
    Parse a single BILLSTATUS XML file.

    Returns: (bill_metadata, actions_list, cosponsors_list)
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Find the bill element
        bill = root.find('.//bill')
        if bill is None:
            return (None, [], [])

        # Extract bill metadata
        bill_id = None
        congress = None
        bill_type = None
        bill_number = None

        # Get identifiers
        congress_elem = bill.find('congress')
        if congress_elem is not None:
            congress = congress_elem.text

        type_elem = bill.find('type')
        if type_elem is not None:
            bill_type = type_elem.text.lower()

        number_elem = bill.find('number')
        if number_elem is not None:
            bill_number = number_elem.text

        if congress and bill_type and bill_number:
            bill_id = f"{congress}-{bill_type.upper()}-{bill_number}"
        else:
            return (None, [], [])

        # Extract basic metadata
        metadata = {
            'bill_id': bill_id,
            'congress': congress,
            'bill_type': bill_type,
            'bill_number': bill_number,
        }

        # Title
        title_elem = bill.find('.//title')
        if title_elem is not None:
            metadata['title'] = title_elem.text

        # Introduced date
        introduced_date_elem = bill.find('.//introducedDate')
        if introduced_date_elem is not None:
            metadata['introduced_date'] = introduced_date_elem.text

        # Latest action
        latest_action = bill.find('.//latestAction')
        if latest_action is not None:
            action_date_elem = latest_action.find('actionDate')
            action_text_elem = latest_action.find('text')
            if action_date_elem is not None:
                metadata['latest_action_date'] = action_date_elem.text
            if action_text_elem is not None:
                metadata['latest_action_text'] = action_text_elem.text

        # Sponsors
        sponsors = bill.find('.//sponsors')
        if sponsors is not None:
            sponsor_list = []
            for item in sponsors.findall('.//item'):
                full_name = item.find('fullName')
                party = item.find('party')
                state = item.find('state')

                if full_name is not None:
                    sponsor_list.append(full_name.text)

                if party is not None and 'sponsor_party' not in metadata:
                    metadata['sponsor_party'] = party.text

                if state is not None and 'sponsor_state' not in metadata:
                    metadata['sponsor_state'] = state.text

            metadata['sponsors'] = ', '.join(sponsor_list) if sponsor_list else ''

        # Policy area
        policy_area = bill.find('.//policyArea/name')
        if policy_area is not None:
            metadata['policy_area'] = policy_area.text

        # Subjects count
        legislative_subjects = bill.find('.//legislativeSubjects')
        if legislative_subjects is not None:
            subject_count = len(legislative_subjects.findall('.//item'))
            metadata['subject_count'] = subject_count

        # Parse actions
        actions_list = []
        actions = bill.find('.//actions')
        if actions is not None:
            for item in actions.findall('.//item'):
                action_date_elem = item.find('actionDate')
                action_text_elem = item.find('text')
                action_type_elem = item.find('type')
                action_code_elem = item.find('actionCode')
                source_system_elem = item.find('.//sourceSystem/name')

                action = {
                    'bill_id': bill_id,
                    'date': action_date_elem.text if action_date_elem is not None else '',
                    'text': action_text_elem.text if action_text_elem is not None else '',
                    'type': action_type_elem.text if action_type_elem is not None else '',
                    'action_code': action_code_elem.text if action_code_elem is not None else '',
                    'source_system': source_system_elem.text if source_system_elem is not None else '',
                }

                # Extract committees from this action
                committees = item.find('committees')
                committee_names = []
                if committees is not None:
                    for comm_item in committees.findall('.//item'):
                        comm_name = comm_item.find('name')
                        if comm_name is not None:
                            committee_names.append(comm_name.text)

                action['committees'] = ', '.join(committee_names) if committee_names else ''

                actions_list.append(action)

        # Parse cosponsors
        cosponsors_list = []
        cosponsors = bill.find('.//cosponsors')
        if cosponsors is not None:
            for item in cosponsors.findall('.//item'):
                full_name = item.find('fullName')
                party = item.find('party')
                state = item.find('state')
                district = item.find('district')
                sponsorship_date = item.find('sponsorshipDate')
                is_original = item.find('isOriginalCosponsor')

                cosponsor = {
                    'bill_id': bill_id,
                    'name': full_name.text if full_name is not None else '',
                    'party': party.text if party is not None else '',
                    'state': state.text if state is not None else '',
                    'district': district.text if district is not None else '',
                    'sponsored_date': sponsorship_date.text if sponsorship_date is not None else '',
                    'is_original': is_original.text if is_original is not None else 'false',
                }

                cosponsors_list.append(cosponsor)

        return (metadata, actions_list, cosponsors_list)

    except ET.ParseError as e:
        print(f"  ⚠️ XML parse error in {xml_path.name}: {e}")
        return (None, [], [])
    except Exception as e:
        print(f"  ⚠️ Error parsing {xml_path.name}: {e}")
        return (None, [], [])


def parse_congress_data(congress, bill_type, progress):
    """
    Parse all XML files for one congress and bill type.

    Returns: (bills_metadata, all_actions, all_cosponsors)
    """
    # Check if already parsed
    key = f"{congress}_{bill_type}"
    if progress.get(key, {}).get('parsed', False):
        print(f"  ✓ {bill_type.upper():6s} - Already parsed (cached)")
        return ([], [], [])

    # Find XML directory
    xml_dir = BULK_DOWNLOADS_DIR / str(congress) / bill_type

    if not xml_dir.exists():
        print(f"  ⚠️ {bill_type.upper():6s} - Directory not found: {xml_dir}")
        return ([], [], [])

    # Get all XML files
    xml_files = list(xml_dir.glob("BILLSTATUS-*.xml"))

    if not xml_files:
        print(f"  ⚠️ {bill_type.upper():6s} - No XML files found in {xml_dir}")
        return ([], [], [])

    print(f"  → {bill_type.upper():6s} - Parsing {len(xml_files)} XML files...")

    all_metadata = []
    all_actions = []
    all_cosponsors = []

    start_time = time.time()

    for i, xml_file in enumerate(xml_files, 1):
        # Parse file
        metadata, actions, cosponsors = parse_bill_xml(xml_file)

        if metadata:
            all_metadata.append(metadata)
            all_actions.extend(actions)
            all_cosponsors.extend(cosponsors)

        # Progress update every 500 files
        if i % 500 == 0 or i == len(xml_files):
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(xml_files) - i) / rate if rate > 0 else 0

            print(f"    {i}/{len(xml_files)} files ({i/len(xml_files)*100:.1f}%) | "
                  f"Rate: {rate:.1f} files/sec | "
                  f"ETA: {remaining:.0f}s | "
                  f"Actions: {len(all_actions):,} | "
                  f"Cosponsors: {len(all_cosponsors):,}", end='\r')

    # Final newline
    print()

    elapsed = time.time() - start_time
    print(f"  ✓ {bill_type.upper():6s} - Parsed {len(all_metadata)} bills, "
          f"{len(all_actions):,} actions, {len(all_cosponsors):,} cosponsors "
          f"in {elapsed:.1f}s")

    # Mark as parsed
    if key not in progress:
        progress[key] = {}
    progress[key]['parsed'] = True
    progress[key]['num_bills'] = len(all_metadata)
    progress[key]['num_actions'] = len(all_actions)
    progress[key]['num_cosponsors'] = len(all_cosponsors)
    progress[key]['parse_time'] = datetime.now().isoformat()
    save_progress(progress)

    return (all_metadata, all_actions, all_cosponsors)


def main():
    """Main parsing orchestration"""
    print("="*70)
    print("PARSING BULK BILL STATUS XML DATA")
    print("="*70)
    print(f"Input: {BULK_DOWNLOADS_DIR.absolute()}")
    print(f"Output: {OUTPUT_DIR.absolute()}")
    print(f"Congresses: {', '.join(map(str, CONGRESSES))}")
    print(f"Bill Types: {', '.join([t.upper() for t in BILL_TYPES])}")
    print("="*70)
    print()

    # Load progress
    progress = load_progress()
    if progress:
        completed = sum(1 for v in progress.values() if v.get('parsed', False))
        print(f"Resuming from previous run: {completed} already parsed")
        print()

    # Accumulators for all data
    all_bills = []
    all_actions = []
    all_cosponsors = []

    # Statistics
    total_groups = len(CONGRESSES) * len(BILL_TYPES)
    processed_groups = 0
    start_time = datetime.now()

    # Parse each congress
    for congress in CONGRESSES:
        print(f"\n{'='*70}")
        print(f"{congress}th Congress")
        print(f"{'='*70}")

        for bill_type in BILL_TYPES:
            bills, actions, cosponsors = parse_congress_data(congress, bill_type, progress)

            all_bills.extend(bills)
            all_actions.extend(actions)
            all_cosponsors.extend(cosponsors)

            processed_groups += 1

    # Save combined datasets
    print(f"\n{'='*70}")
    print(f"SAVING COMBINED DATASETS")
    print(f"{'='*70}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Save bills metadata
    if all_bills:
        bills_df = pd.DataFrame(all_bills)
        bills_df.to_csv(OUTPUT_DIR / 'bills_metadata.csv', index=False)
        print(f"✅ Saved bills_metadata.csv ({len(bills_df):,} bills)")
    else:
        print(f"⚠️ No bill metadata to save")

    # Save actions timeline
    if all_actions:
        actions_df = pd.DataFrame(all_actions)
        # Sort by date
        actions_df['date'] = pd.to_datetime(actions_df['date'], errors='coerce')
        actions_df = actions_df.sort_values('date')
        actions_df.to_csv(OUTPUT_DIR / 'actions_timeline.csv', index=False)
        print(f"✅ Saved actions_timeline.csv ({len(actions_df):,} actions)")
    else:
        print(f"⚠️ No actions to save")

    # Save cosponsors timeline
    if all_cosponsors:
        cosponsors_df = pd.DataFrame(all_cosponsors)
        # Sort by sponsored date
        cosponsors_df['sponsored_date'] = pd.to_datetime(cosponsors_df['sponsored_date'], errors='coerce')
        cosponsors_df = cosponsors_df.sort_values('sponsored_date')
        cosponsors_df.to_csv(OUTPUT_DIR / 'cosponsors_timeline.csv', index=False)
        print(f"✅ Saved cosponsors_timeline.csv ({len(cosponsors_df):,} cosponsors)")
    else:
        print(f"⚠️ No cosponsors to save")

    # Final summary
    elapsed = (datetime.now() - start_time).total_seconds() / 60  # minutes

    print(f"\n{'='*70}")
    print(f"PARSING COMPLETE")
    print(f"{'='*70}")
    print(f"Bills: {len(all_bills):,}")
    print(f"Actions: {len(all_actions):,}")
    print(f"Cosponsors: {len(all_cosponsors):,}")
    print(f"Total time: {elapsed:.1f} minutes")
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_DIR / 'bills_metadata.csv'}")
    print(f"  - {OUTPUT_DIR / 'actions_timeline.csv'}")
    print(f"  - {OUTPUT_DIR / 'cosponsors_timeline.csv'}")
    print(f"\nNext step: Run create_temporal_snapshots.py to generate training data")
    print(f"{'='*70}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nParsing interrupted by user.")
        print("Progress has been saved. Re-run this script to resume.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("\nProgress has been saved. Re-run this script to resume.")

import requests
import pandas as pd
import os
import streamlit as st

# Try to get API key from Streamlit secrets first (for deployed app)
# If not available, try environment variable (for local development)
try:
    # For Streamlit Cloud deployment
    CONGRESS_API_KEY = st.secrets["CONGRESS_API_KEY"]
except:
    # For local development with .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        CONGRESS_API_KEY = os.getenv('CONGRESS_API_KEY')
    except:
        # If no .env file, just try environment variable
        CONGRESS_API_KEY = os.getenv('CONGRESS_API_KEY')

# Optional: Add a fallback or warning if no API key is found
if not CONGRESS_API_KEY:
    st.warning("⚠️ No Congress.gov API key found. The app may have limited functionality.")
    CONGRESS_API_KEY = ""  # Set empty string to avoid None errors

def fetch_bill_titles(bill_id, congress=118, bill_type='hr'):
    """
    Fetch all titles for a bill
    """
    if not CONGRESS_API_KEY:
        return {'short_title': '', 'official_title': '', 'display_title': ''}
    
    url = f'https://api.congress.gov/v3/bill/{congress}/{bill_type}/{bill_id}/titles?api_key={CONGRESS_API_KEY}'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        titles_data = data.get('titles', [])
        
        # Find the short title
        short_title = ''
        official_title = ''
        display_title = ''
        
        for title_item in titles_data:
            if isinstance(title_item, dict):
                title_type = title_item.get('titleType', '').lower()
                title_text = title_item.get('title', '')
                
                if 'short' in title_type and title_text and not short_title:
                    short_title = title_text
                elif 'official' in title_type and title_text:
                    official_title = title_text
                elif 'display' in title_type and title_text:
                    display_title = title_text
        
        return {
            'short_title': short_title,
            'official_title': official_title,
            'display_title': display_title
        }
    else:
        return {'short_title': '', 'official_title': '', 'display_title': ''}

def fetch_bill(bill_id, congress=118, bill_type='hr'):
    """
    Fetch comprehensive bill information including sponsors, committees, and subjects
    """
    if not CONGRESS_API_KEY:
        return pd.DataFrame()
    
    url = f'https://api.congress.gov/v3/bill/{congress}/{bill_type}/{bill_id}?api_key={CONGRESS_API_KEY}'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        bill_data = data.get('bill', {})
        
        # Extract sponsor information
        sponsors_data = bill_data.get('sponsors', [])
        if isinstance(sponsors_data, dict):
            sponsors_data = [sponsors_data]
        elif not isinstance(sponsors_data, list):
            sponsors_data = []
        
        sponsor_info = {}
        if sponsors_data and isinstance(sponsors_data[0], dict):
            sponsor_info = {
                'sponsor_name': sponsors_data[0].get('fullName', ''),
                'sponsor_party': sponsors_data[0].get('party', ''),
                'sponsor_state': sponsors_data[0].get('state', ''),
                'sponsor_district': sponsors_data[0].get('district', '')
            }
        else:
            sponsor_info = {
                'sponsor_name': '',
                'sponsor_party': '',
                'sponsor_state': '',
                'sponsor_district': ''
            }
        
        # Extract committees
        committees_data = bill_data.get('committees', {})
        committee_list = []
        
        if isinstance(committees_data, dict) and 'item' in committees_data:
            items = committees_data['item']
            if isinstance(items, list):
                committee_list = items
            elif isinstance(items, dict):
                committee_list = [items]
        elif isinstance(committees_data, list):
            committee_list = committees_data
        
        # Get latest action
        latest_action = bill_data.get('latestAction', {})
        latest_action_text = latest_action.get('text', '') if isinstance(latest_action, dict) else ''
        latest_action_date = latest_action.get('actionDate', '') if isinstance(latest_action, dict) else ''
        
        # Create DataFrame with all information
        df = pd.DataFrame([{
            'bill_id': f"{congress}-{bill_type.upper()}-{bill_id}",
            'bill_number': bill_data.get('number', ''),
            'bill_type': bill_data.get('type', ''),
            'title': bill_data.get('title', ''),
            'short_title': '',  # Will be updated with title fetch
            'introduced_date': bill_data.get('introducedDate', ''),
            'latest_action': latest_action_text,
            'latest_action_date': latest_action_date,
            'policy_area': bill_data.get('policyArea', {}).get('name', '') if isinstance(bill_data.get('policyArea'), dict) else '',
            'committees': ', '.join([c.get('name', '') for c in committee_list if isinstance(c, dict)]),
            'committee_count': len(committee_list),
            'sponsor_count': len(sponsors_data),
            'congress': bill_data.get('congress', congress),
            'origin_chamber': bill_data.get('originChamber', ''),
            'origin_chamber_code': bill_data.get('originChamberCode', ''),
            'is_law': bill_data.get('isLaw', False),
            'laws': ', '.join([str(law.get('number', '')) for law in bill_data.get('laws', []) if isinstance(law, dict)]),
            **sponsor_info
        }])
        
        # Get detailed title information
        titles_info = fetch_bill_titles(bill_id, congress, bill_type)
        df.loc[0, 'short_title'] = titles_info['short_title']
        
        # If no short title, use display title
        if not df.loc[0, 'short_title'] and titles_info['display_title']:
            df.loc[0, 'short_title'] = titles_info['display_title']
        
        return df
    else:
        print(f"Error for bill {bill_id}: {response.status_code}")
        return pd.DataFrame()

def fetch_bill_actions(bill_id, congress=118, bill_type='hr'):
    """
    Fetch all bill actions with pagination support
    """
    if not CONGRESS_API_KEY:
        return pd.DataFrame()
    
    all_actions = []
    offset = 0
    limit = 250  # Maximum allowed by the API
    
    while True:
        # Add limit and offset parameters for pagination
        url = f'https://api.congress.gov/v3/bill/{congress}/{bill_type}/{bill_id}/actions?api_key={CONGRESS_API_KEY}&limit={limit}&offset={offset}'
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            actions_data = data.get('actions', [])
            
            # If actions_data is empty, we've reached the end
            if not actions_data:
                break
            
            for a in actions_data:
                # Handle committees field which can be a dict or list
                committees_data = a.get('committees', {})
                
                # Extract committee names based on data structure
                if isinstance(committees_data, dict):
                    # If it's a dict, look for 'item' field
                    items = committees_data.get('item', [])
                    if isinstance(items, list):
                        committee_names = [c.get('name', '') for c in items if isinstance(c, dict)]
                    elif isinstance(items, dict):
                        committee_names = [items.get('name', '')]
                    else:
                        committee_names = []
                elif isinstance(committees_data, list):
                    # If it's already a list of committees
                    committee_names = [c.get('name', '') for c in committees_data if isinstance(c, dict)]
                else:
                    committee_names = []
                
                action = {
                    'bill_id': f"{congress}-{bill_type.upper()}-{bill_id}",
                    'action_date': a.get('actionDate', ''),
                    'text': a.get('text', ''),
                    'type': a.get('type', ''),
                    'action_code': a.get('actionCode', ''),
                    'source_system': a.get('sourceSystem', {}).get('name', '') if isinstance(a.get('sourceSystem'), dict) else '',
                    'committees': ', '.join(committee_names)
                }
                all_actions.append(action)
            
            # If we got fewer than the limit, we've reached the end
            if len(actions_data) < limit:
                break
            
            # Otherwise, continue to the next page
            offset += limit
        else:
            print(f"Error for actions {bill_id}: {response.status_code}")
            break
    
    # Convert to DataFrame
    if all_actions:
        df_actions = pd.DataFrame(all_actions)
        print(f"Fetched {len(df_actions)} actions for {bill_id}")
        return df_actions
    else:
        print(f"No actions found for {bill_id}")
        return pd.DataFrame()

def fetch_cosponsors(bill_id, congress=118, bill_type='hr'):
    """
    Fetch detailed cosponsor information
    """
    if not CONGRESS_API_KEY:
        return pd.DataFrame(), {}
    
    url = f'https://api.congress.gov/v3/bill/{congress}/{bill_type}/{bill_id}/cosponsors?api_key={CONGRESS_API_KEY}'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        cosponsors_data = data.get('cosponsors', [])
        
        # Handle case where cosponsors is a dict with items
        if isinstance(cosponsors_data, dict):
            items = cosponsors_data.get('item', [])
            if isinstance(items, list):
                cosponsors_data = items
            elif isinstance(items, dict):
                cosponsors_data = [items]
            else:
                cosponsors_data = []
        elif not isinstance(cosponsors_data, list):
            cosponsors_data = []
        
        cosponsors = []
        for c in cosponsors_data:
            if isinstance(c, dict):
                cosponsor = {
                    'bill_id': f"{congress}-{bill_type.upper()}-{bill_id}",
                    'name': c.get('fullName', ''),
                    'party': c.get('party', ''),
                    'state': c.get('state', ''),
                    'district': c.get('district', ''),
                    'sponsored_date': c.get('sponsorshipDate', ''),
                    'is_original': c.get('isOriginalCosponsor', False)
                }
                cosponsors.append(cosponsor)
        
        df_cosponsors = pd.DataFrame(cosponsors)
        
        # Calculate party breakdown
        if not df_cosponsors.empty:
            party_counts = df_cosponsors['party'].value_counts().to_dict()
            return df_cosponsors, party_counts
        else:
            return df_cosponsors, {}
    else:
        print(f"Error for cosponsors {bill_id}: {response.status_code}")
        return pd.DataFrame(), {}

def fetch_subjects(bill_id, congress=118, bill_type='hr'):
    """
    Fetch bill subjects
    """
    if not CONGRESS_API_KEY:
        return {'subjects': [], 'policy_area': 'Unknown', 'subject_count': 0}
    
    url = f'https://api.congress.gov/v3/bill/{congress}/{bill_type}/{bill_id}/subjects?api_key={CONGRESS_API_KEY}'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        subjects = data.get('subjects', {})
        
        # Extract legislative subjects - handle different structures
        legislative_subjects = subjects.get('legislativeSubjects', [])
        
        # Handle case where legislativeSubjects is a list directly
        if isinstance(legislative_subjects, list):
            subject_names = [s.get('name', '') for s in legislative_subjects if isinstance(s, dict) and s.get('name')]
        # Handle case where it's a dict with 'item' field
        elif isinstance(legislative_subjects, dict):
            items = legislative_subjects.get('item', [])
            if isinstance(items, list):
                subject_names = [s.get('name', '') for s in items if isinstance(s, dict) and s.get('name')]
            elif isinstance(items, dict) and items.get('name'):
                subject_names = [items.get('name', '')]
            else:
                subject_names = []
        else:
            subject_names = []
        
        # Extract policy area - handle different structures
        policy_area = subjects.get('policyArea', {})
        if isinstance(policy_area, dict):
            policy_area_name = policy_area.get('name', 'Unknown')
        elif isinstance(policy_area, str):
            policy_area_name = policy_area
        else:
            policy_area_name = 'Unknown'
        
        return {
            'subjects': subject_names,
            'policy_area': policy_area_name,
            'subject_count': len(subject_names)
        }
    else:
        print(f"Error for subjects {bill_id}: {response.status_code}")
        return {'subjects': [], 'policy_area': 'Unknown', 'subject_count': 0}

def fetch_text_versions(bill_id, congress=118, bill_type='hr'):
    """
    Fetch available text versions of the bill
    """
    if not CONGRESS_API_KEY:
        return pd.DataFrame()
    
    url = f'https://api.congress.gov/v3/bill/{congress}/{bill_type}/{bill_id}/text?api_key={CONGRESS_API_KEY}'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        text_versions_data = data.get('textVersions', [])
        
        # Handle case where textVersions is a dict with items
        if isinstance(text_versions_data, dict):
            items = text_versions_data.get('item', [])
            if isinstance(items, list):
                text_versions_data = items
            elif isinstance(items, dict):
                text_versions_data = [items]
            else:
                text_versions_data = []
        elif not isinstance(text_versions_data, list):
            text_versions_data = []
        
        versions = []
        for v in text_versions_data:
            if isinstance(v, dict):
                formats_data = v.get('formats', [])
                format_types = []
                
                if isinstance(formats_data, list):
                    format_types = [f.get('type', '') for f in formats_data if isinstance(f, dict)]
                elif isinstance(formats_data, dict):
                    format_types = [formats_data.get('type', '')]
                
                version = {
                    'type': v.get('type', ''),
                    'date': v.get('date', ''),
                    'formats': format_types
                }
                versions.append(version)
        
        return pd.DataFrame(versions)
    else:
        print(f"Error for text versions {bill_id}: {response.status_code}")
        return pd.DataFrame()

def fetch_comprehensive_bill_data(bill_id, congress=118, bill_type='hr'):
    """
    Fetch all available data for a bill
    """
    if not CONGRESS_API_KEY:
        return None
    
    # Get basic bill info
    bill_df = fetch_bill(bill_id, congress, bill_type)
    
    if bill_df.empty:
        return None
    
    # Get additional data
    actions_df = fetch_bill_actions(bill_id, congress, bill_type)
    cosponsors_df, party_breakdown = fetch_cosponsors(bill_id, congress, bill_type)
    subjects_data = fetch_subjects(bill_id, congress, bill_type)
    text_versions_df = fetch_text_versions(bill_id, congress, bill_type)
    
    # Combine all data
    result = {
        'bill': bill_df,
        'actions': actions_df,
        'cosponsors': cosponsors_df,
        'party_breakdown': party_breakdown,
        'subjects': subjects_data,
        'text_versions': text_versions_df
    }
    
    return result

# Keep the public comments function unchanged
def fetch_public_comments(docket_id='CMS-2024-0001'):
    """
    Fetch public comments (unchanged from original)
    """
    url = f'https://api.regulations.gov/v4/comments?filter[docketId]={docket_id}&api_key=DEMO_KEY'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        comments = [item['attributes']['comment'] for item in data.get('data', []) if 'comment' in item['attributes']]
        df_comments = pd.DataFrame({'comment': comments})
        return df_comments
    else:
        print(f"Error for docket {docket_id}: {response.status_code}")
        return pd.DataFrame()
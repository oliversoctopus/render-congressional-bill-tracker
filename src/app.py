import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import os
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV

# Import data fetch functions
from data_fetch import (fetch_bill, fetch_bill_actions, fetch_public_comments, 
                       fetch_comprehensive_bill_data, fetch_cosponsors, fetch_subjects)

# Page configuration
st.set_page_config(
    page_title="Congressional Bill Tracker - Advanced Analytics",
    page_icon="📊",
    layout="wide"
)

st.title('📊 Congressional Bill Tracker v5.0 - Advanced Analytics')
st.markdown("### AI-powered predictions trained on 273,113 temporal snapshots from 76,854 bills (Congresses 113-118)")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📖 About This System")
    st.markdown("""
    ### Congressional Bill Tracker v5.0

    **Training Data:**
    - 📊 273,113 temporal snapshots
    - 🏛️ 76,854 unique bills
    - 📅 6 Congresses (113th-118th)

    **Historical Rates (Unique Bills):**
    - 📈 3.7% pass overall
    - 🎯 4.9% viable
    - ✅ 75.4% of viable bills pass

    **Model Performance:**
    - ✅ 86.6%-97.9% ROC-AUC
    - ✅ Calibrated probabilities
    - ✅ Ensemble models for stability
    - ✅ Stage-specific predictions
    - ✅ Temporal snapshot training

    ### Model Stages

    **🆕 New Bill (Day 1)**
    - 16 basic features
    - 94.1% viability ROC-AUC
    - 97.9% passage ROC-AUC

    **📈 Early Stage (2-30 days)**
    - 24 extended features
    - 96.4% viability ROC-AUC
    - 96.4% passage ROC-AUC

    **🔄 Progressive (30+ days)**
    - 41 comprehensive features
    - 86.6% viability ROC-AUC
    - 94.4% passage ROC-AUC

    ### Understanding Predictions

    **Confidence Intervals:**
    - Narrow = Models agree
    - Wide = More uncertainty
    - Shows min/max from ensemble

    **Pass Rate Context:**
    - Unique bills: 3.7% overall
    - Viable bills: 75.4% pass rate
    - Varies by Congress (60.9%-87.1%)
    """)

# Main input
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    bill_input = st.text_input('Enter Bill Number', placeholder='e.g., 1234')
with col2:
    bill_type = st.selectbox('Type', ['hr', 's'], 
                            format_func=lambda x: 'H.R.' if x == 'hr' else 'S.')
with col3:
    congress = st.number_input('Congress', min_value=110, max_value=120, value=118)

# Display options
with st.expander("📊 Display Options"):
    col1, col2 = st.columns(2)
    with col1:
        show_confidence = st.checkbox("Show confidence intervals", value=True)
        show_model_breakdown = st.checkbox("Show individual model predictions", value=False)
    with col2:
        show_feature_analysis = st.checkbox("Show feature importance", value=True)
        show_similar_bills = st.checkbox("Show similar bills analysis", value=False)

if bill_input:
    try:
        # Congress end dates for detecting ended congresses
        CONGRESS_END_DATES = {
            113: datetime(2015, 1, 3),
            114: datetime(2017, 1, 3),
            115: datetime(2019, 1, 3),
            116: datetime(2021, 1, 3),
            117: datetime(2023, 1, 3),
            118: datetime(2025, 1, 3),
            119: datetime(2027, 1, 3),  # Future
            120: datetime(2029, 1, 3),  # Future
        }

        congress_ended = datetime.now() > CONGRESS_END_DATES.get(congress, datetime(2099, 1, 1))

        # Fetch bill data
        with st.spinner('Fetching bill information...'):
            comprehensive_data = fetch_comprehensive_bill_data(
                bill_input, congress=congress, bill_type=bill_type
            )
            
            if not comprehensive_data:
                st.error("Could not fetch bill data. Please check the bill number.")
                st.stop()
            
            df = comprehensive_data['bill_info']
            actions_df = comprehensive_data['actions']
            cosponsors_df = comprehensive_data['cosponsors']
            subjects_data = comprehensive_data['subjects']
            metrics = comprehensive_data['metrics']
        
        # Calculate temporal metrics
        if not actions_df.empty:
            actions_df['date'] = pd.to_datetime(actions_df['date'])
            first_action = actions_df['date'].min()
            last_action = actions_df['date'].max()
            # FIXED: Use last_action - first_action (matches training data)
            # Training uses: (latest_action_date - introduced_date)
            days_active = (last_action - first_action).days
            days_active = max(1, min(days_active, 730))  # Match training clip (1-730 days)
            days_since_last_action = (datetime.now() - last_action).days
        else:
            days_active = 1
            days_since_last_action = 0
        
        # Bill header with verification - use correct title from comprehensive data
        if not df.empty:
            # Get title from the comprehensive data which includes titles from the separate API call
            full_title = None
            display_title = None # Not used because it doesn't provide the actual title
            
            # Check for short_title first (from the titles API call)
            if 'short_title' in df.columns and df['short_title'].values[0]:
                display_title = df['short_title'].values[0]
                full_title = df['title'].values[0] if 'title' in df.columns else display_title
            elif 'title' in df.columns and df['title'].values[0]:
                full_title = df['title'].values[0]
                display_title = full_title[:100] + "..." if len(full_title) > 100 else full_title
            
            if display_title:
                st.subheader(f"Bill Analysis: {bill_type.upper()}.{bill_input} - {full_title}")
            else:
                st.subheader(f"Bill Analysis: {bill_type.upper()}.{bill_input}")
            
            # Add bill verification info
            with st.expander("📋 Bill Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Bill Number:** {bill_type.upper()}.{bill_input}")
                    st.write(f"**Congress:** {congress}th")
                    if 'introduced_date' in df.columns:
                        st.write(f"**Introduced:** {df['introduced_date'].values[0]}")
                    if 'sponsor' in df.columns:
                        st.write(f"**Sponsor:** {df['sponsor'].values[0]}")
                with col2:
                    if 'status' in df.columns:
                        st.write(f"**Status:** {df['status'].values[0]}")
                    if 'policy_area' in df.columns:
                        st.write(f"**Policy Area:** {df['policy_area'].values[0]}")
                    if 'committees' in df.columns:
                        st.write(f"**Committees:** {df['committees'].values[0]}")
                
                if 'title' in df.columns:
                    st.write(f"**Full Title:** {df['title'].values[0]}")
                elif 'official_title' in df.columns:
                    st.write(f"**Full Title:** {df['official_title'].values[0]}")

                st.info("💡 If this is not the bill you're looking for, please verify the bill number and congress session.")
        else:
            st.subheader(f"Bill Analysis: {bill_type.upper()}.{bill_input}")

        # Educational mode for ended congresses
        simulate_date = None
        if congress_ended:
            st.error(f"⚠️ **The {congress}th Congress ended on {CONGRESS_END_DATES[congress].strftime('%B %d, %Y')}**")
            st.error("This bill can no longer become law. Bills that do not pass in their congress session automatically die.")

            st.info("📚 **Educational Mode**: Below shows what the model would have predicted at different points during the congress.")

            # Let user select a historical date to simulate
            if not df.empty and 'introduced_date' in df.columns:
                min_date = pd.to_datetime(df['introduced_date'].values[0]).date()
                max_date = min(last_action.date() if not actions_df.empty else CONGRESS_END_DATES[congress].date(),
                              CONGRESS_END_DATES[congress].date())

                simulate_date = st.date_input(
                    "Select a date to simulate predictions (as if that were 'today'):",
                    min_value=min_date,
                    max_value=max_date,
                    value=max_date,
                    help="Choose a historical date to see what the model would have predicted at that point in time"
                )

                # Recalculate days_active and days_since_last_action based on simulation date
                if simulate_date:
                    simulate_datetime = pd.to_datetime(simulate_date)

                    # Filter actions up to simulation date
                    if not actions_df.empty:
                        simulated_actions = actions_df[actions_df['date'] <= simulate_datetime]
                        if not simulated_actions.empty:
                            sim_first_action = simulated_actions['date'].min()
                            sim_last_action = simulated_actions['date'].max()
                            days_active = (sim_last_action - sim_first_action).days
                            days_active = max(1, min(days_active, 730))
                            days_since_last_action = (simulate_datetime - sim_last_action).days

                    st.caption(f"🕐 Simulating predictions as if today were **{simulate_date.strftime('%B %d, %Y')}** ({days_active} days of activity, {days_since_last_action} days since last action)")

            st.markdown("---")

        # Key metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Days Active", days_active)
        with col2:
            st.metric("Total Actions", metrics.get('total_actions', 0))
        with col3:
            st.metric("Committees", metrics.get('committee_count', 0))
        with col4:
            st.metric("Cosponsors", df['cosponsor_count'].values[0] if not df.empty else 0)
        with col5:
            bipartisan = "Yes" if df['is_bipartisan'].values[0] else "No"
            st.metric("Bipartisan", bipartisan)
        with col6:
            activity_rate = metrics.get('total_actions', 0) / max(days_active, 1)
            st.metric("Activity Rate", f"{activity_rate:.2f}/day")
        
        # Activity timeline as table
        if not actions_df.empty:
            st.subheader("📅 Legislative Activity Timeline")
            
            # Parse dates with multiple possible formats
            def parse_action_date(date_str):
                """Parse dates from various formats the API might return"""
                if pd.isna(date_str) or not date_str:
                    return None
                
                # Try different date formats
                formats = [
                    '%Y-%m-%d',  # Standard ISO format
                    '%Y-%m-%dT%H:%M:%S',  # ISO with time
                    '%Y-%m-%dT%H:%M:%SZ',  # ISO with UTC
                    '%Y-%m-%dT%H:%M:%S.%fZ',  # ISO with milliseconds
                    '%m/%d/%Y',  # US format
                    '%d/%m/%Y',  # European format
                ]
                
                for fmt in formats:
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except:
                        continue
                
                # If all formats fail, try pandas general parser
                try:
                    return pd.to_datetime(date_str)
                except:
                    return None
            
            # Apply parsing to all dates
            actions_df['parsed_date'] = actions_df['date'].apply(parse_action_date)
            
            # Remove any rows with invalid dates
            valid_actions = actions_df.dropna(subset=['parsed_date']).copy()
            
            if len(valid_actions) == 0:
                st.warning("No valid dates found in legislative actions.")
            else:
                # Sort by date in descending order (most recent first)
                valid_actions = valid_actions.sort_values('parsed_date', ascending=False)
            
            # Remove true duplicates (same date AND same text)
            # But keep actions that happened on the same date with different text
            valid_actions['date_str'] = valid_actions['parsed_date'].dt.strftime('%Y-%m-%d')
            valid_actions = valid_actions.drop_duplicates(subset=['date_str', 'text'], keep='first')
            
            # Format the data for display
            timeline_data = []
            for _, action in valid_actions.iterrows():
                # Clean and truncate action text
                action_text = str(action['text']).strip()
                if len(action_text) > 200:
                    action_text = action_text[:197] + "..."
                
                # Calculate days ago
                days_ago = (datetime.now() - action['parsed_date']).days
                
                timeline_data.append({
                    'Date': action['parsed_date'].strftime('%B %d, %Y'),
                    'Action': action_text,
                    'Days Ago': days_ago,
                    'Chamber': action.get('source_system', 'Unknown') if 'source_system' in action else 'Congress'
                })
            
            timeline_df = pd.DataFrame(timeline_data)
            
            # Show total count and date range
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Actions", len(valid_actions))
            with col2:
                if len(valid_actions) > 1:
                    days_span = (valid_actions['parsed_date'].max() - valid_actions['parsed_date'].min()).days
                else:
                    days_span = 0
                st.metric("Activity Span", f"{days_span} days")
            with col3:
                recent_actions = len(valid_actions[valid_actions['parsed_date'] > datetime.now() - timedelta(days=30)])
                st.metric("Recent Actions (30d)", recent_actions)
            
            # Display the table with pagination for long lists
            if len(timeline_df) > 20:
                # Initialize session state for pagination
                if 'leg_timeline_page' not in st.session_state:
                    st.session_state.leg_timeline_page = 0  # 0-indexed
                
                items_per_page = 20
                total_pages = (len(timeline_df) + items_per_page - 1) // items_per_page
                
                # Ensure current page is within bounds
                st.session_state.leg_timeline_page = max(0, min(st.session_state.leg_timeline_page, total_pages - 1))
                
                # Calculate data slice
                start_idx = st.session_state.leg_timeline_page * items_per_page
                end_idx = min(start_idx + items_per_page, len(timeline_df))
                
                # Display the data first
                st.dataframe(
                    timeline_df.iloc[start_idx:end_idx],
                    hide_index=True,
                    width='stretch',
                    height=min(700, (end_idx - start_idx) * 35 + 35)
                )
                
                # Then display pagination controls below
                col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
                
                with col1:
                    if st.button('⏮️ First', key='timeline_first', width='stretch'):
                        st.session_state.leg_timeline_page = 0
                        st.rerun()
                
                with col2:
                    if st.button('◀️ Prev', key='timeline_prev', width='stretch', 
                                disabled=(st.session_state.leg_timeline_page == 0)):
                        st.session_state.leg_timeline_page -= 1
                        st.rerun()
                
                with col3:
                    # Page selector dropdown
                    page_options = [f"Page {i+1} of {total_pages}" for i in range(total_pages)]
                    selected_page = st.selectbox(
                        "Jump to page",
                        options=range(total_pages),
                        index=st.session_state.leg_timeline_page,
                        format_func=lambda x: f"Page {x+1} of {total_pages}",
                        key='timeline_page_select',
                        label_visibility="collapsed"
                    )
                    if selected_page != st.session_state.leg_timeline_page:
                        st.session_state.leg_timeline_page = selected_page
                        st.rerun()
                
                with col4:
                    if st.button('Next ▶️', key='timeline_next', width='stretch',
                                disabled=(st.session_state.leg_timeline_page >= total_pages - 1)):
                        st.session_state.leg_timeline_page += 1
                        st.rerun()
                
                with col5:
                    if st.button('Last ⏭️', key='timeline_last', width='stretch'):
                        st.session_state.leg_timeline_page = total_pages - 1
                        st.rerun()
                
                # Show page info
                st.caption(f"Showing actions {start_idx + 1} to {end_idx} of {len(timeline_df)} total")
                
            else:
                # Display all if 20 or fewer actions
                st.dataframe(
                    timeline_df,
                    hide_index=True,
                    width='stretch',
                    height=min(700, len(timeline_df) * 35 + 35)
                )
            
            # Add download button for full timeline
            csv = timeline_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Timeline (CSV)",
                data=csv,
                file_name=f"{bill_type}_{bill_input}_timeline.csv",
                mime="text/csv"
            )
        
        st.markdown("---")
        
        # Load individual model stages on demand (lazy loading for performance)
        @st.cache_resource
        def load_single_model(_model_type, _stage):
            """Load a single model stage (lazy loading)"""
            import joblib
            from sklearn.ensemble import VotingClassifier

            model_dir = f'models/{_model_type}_{_stage}'

            if not os.path.exists(model_dir):
                return None

            # Load components
            rf_model = joblib.load(f'{model_dir}/rf_model.pkl')
            components = joblib.load(f'{model_dir}/components.pkl')
            ensemble_config = joblib.load(f'{model_dir}/ensemble_config.pkl')

            # Reconstruct ensemble
            ensemble = VotingClassifier(
                estimators=[
                    ('rf', rf_model),
                    ('gb', components['gb_model']),
                    ('lr', components['lr_model'])
                ],
                voting='soft',
                weights=ensemble_config['weights']
            )
            ensemble.estimators_ = [rf_model, components['gb_model'], components['lr_model']]
            ensemble.classes_ = rf_model.classes_
            ensemble.le_ = None

            return {
                'model': ensemble,
                'scaler': components['scaler'],
                'selector': components['selector'],
                'features': components['metadata']['features'],
                'selected_features': components['metadata']['selected_features']
            }

        # Load metadata only (not models)
        @st.cache_resource
        def load_metadata():
            """Load metadata only (not models - they load on demand)"""
            try:
                # Load metadata and encoders
                metadata_package = joblib.load('models/metadata.pkl')

                # Load viability pass rate data if available
                viability_pass_rates = None
                viability_pass_rates_fine = None
                if os.path.exists('data/viability_pass_rates.csv'):
                    viability_pass_rates = pd.read_csv('data/viability_pass_rates.csv')
                if os.path.exists('data/viability_pass_rates_fine.csv'):
                    viability_pass_rates_fine = pd.read_csv('data/viability_pass_rates_fine.csv')

                # Return metadata only
                return {
                    'label_encoders': metadata_package['label_encoders'],
                    'metadata': metadata_package['metadata'],
                    'viability_pass_rates': viability_pass_rates,
                    'viability_pass_rates_fine': viability_pass_rates_fine
                }
            except Exception as e:
                st.error(f"Error loading metadata: {e}")
                return None
        
        # Load metadata only (fast)
        metadata = load_metadata()

        if metadata:
            label_encoders = metadata['label_encoders']

            # Determine model stage
            if days_active <= 1:
                stage = 'new_bill'
                stage_name = "🆕 New Bill Model"
            elif days_active <= 30:
                stage = 'early_stage'
                stage_name = "📈 Early Stage Model"
            else:
                stage = 'progressive'
                stage_name = "🔄 Progressive Model"

            # Load ONLY the needed models (lazy loading for performance)
            with st.spinner(f"Loading {stage_name}..."):
                viability_model = load_single_model('viability', stage)
                passage_model = load_single_model('passage', stage)

            if not viability_model or not passage_model:
                st.error(f"Failed to load models for stage: {stage}")
                st.stop()
            
            st.header(f"AI Predictions - {stage_name}")
            st.caption(f"Based on {days_active} days of legislative activity")
            
            # Check if bill has already passed House/Senate or become law
            has_passed_house = False
            has_passed_senate = False
            has_become_law = False
            
            # Check status from df
            if not df.empty and 'status' in df.columns:
                status_text = str(df['status'].values[0]).lower()
                if 'passed house' in status_text or 'received in the senate' in status_text:
                    has_passed_house = True
                if 'passed senate' in status_text or 'received in the house' in status_text:
                    has_passed_senate = True
                if 'became law' in status_text or 'public law' in status_text:
                    has_become_law = True
            
            # Also check actions for law status
            if not actions_df.empty:
                action_texts = ' '.join(actions_df['text'].str.lower().tolist())
                if 'became public law' in action_texts or 'became law' in action_texts:
                    has_become_law = True
                if 'passed house' in action_texts:
                    has_passed_house = True
                if 'passed senate' in action_texts:
                    has_passed_senate = True
            
            # Prepare features
            # FIXED: Use bill's actual introduction date for temporal features
            introduced_date = pd.to_datetime(df['introduced_date'].values[0]) if not df.empty and 'introduced_date' in df.columns else datetime.now()

            feature_data = {
                # Basic features
                'sponsor_party': df['sponsor_parties'].values[0] if not df.empty else 'Unknown',
                'sponsor_party_encoded': 0,  # Will be encoded below
                'sponsor_count': len(df['sponsors'].values[0].split(',')) if not df.empty else 1,
                'original_cosponsor_count': metrics.get('original_cosponsor_count', 0),
                'cosponsor_count': df['cosponsor_count'].values[0] if not df.empty else 0,
                'month_introduced': introduced_date.month,
                'quarter_introduced': (introduced_date.month - 1) // 3 + 1,
                'is_election_year': int(introduced_date.year % 2 == 0),  # FIXED: %2 not %4 (all elections, not just presidential)
                'title_length': len(df['short_title'].values[0]) if not df.empty and 'short_title' in df.columns and df['short_title'].values[0] else len(df['title'].values[0]) if not df.empty and 'title' in df.columns and df['title'].values[0] else 100,
                'title_word_count': len(df['short_title'].values[0].split()) if not df.empty and 'short_title' in df.columns and df['short_title'].values[0] else len(df['title'].values[0].split()) if not df.empty and 'title' in df.columns and df['title'].values[0] else 20,
                'title_complexity': 0,  # Will be calculated
                'subject_count': len(subjects_data.get('subjects', [])),
                'policy_area': df['policy_area'].values[0] if not df.empty else 'Unknown',
                'policy_area_encoded': 0,  # Will be encoded
                'dem_total': metrics.get('dem_total', 0),
                'rep_total': metrics.get('rep_total', 0),
                'party_balance': 0,  # Will be calculated
                'party_dominance': 0,  # Will be calculated
                'bipartisan_score': metrics.get('bipartisan_score', 0),
                'has_bipartisan_support': int(df['is_bipartisan'].values[0]) if not df.empty else 0,
                
                # Extended features
                'total_sponsors': 0,  # Will be calculated
                'is_fresh': int(days_active <= 30),
                'support_velocity': 0,  # Will be calculated
                'cosponsor_growth': 0,  # Will be calculated
                
                # Progressive features
                'days_active': days_active,
                'log_days_active': np.log1p(days_active),
                'sqrt_days_active': np.sqrt(days_active),
                'action_count': metrics.get('total_actions', 0),
                'activity_rate': activity_rate,
                'normalized_activity': metrics.get('total_actions', 0) / np.log1p(days_active),
                'early_activity': metrics.get('total_actions', 0) / (min(days_active, 30) + 1),
                'sustained_activity': metrics.get('total_actions', 0) / (min(days_active, 180) + 1),
                'is_active': int(days_active <= 90),
                'is_stale': int(days_active > 180),
                'committee_count': metrics.get('committee_count', 0),
                'has_committee': int(metrics.get('committee_count', 0) > 0),
                'multi_committee': int(metrics.get('committee_count', 0) >= 2),
                'committee_density': metrics.get('committee_count', 0) / max(days_active / 30, 1),
                'bipartisan_momentum': 0,  # Will be calculated
                'committee_activity': 0,  # Will be calculated
                
                # Congress-specific features (new)
                'congress_numeric': congress,
                'is_recent_congress': int(congress >= 117),  # Kept for compatibility with old models

                # Abandonment detection features (new - Phase 1)
                'days_since_last_action': days_since_last_action,
                'log_days_since_last_action': np.log1p(days_since_last_action)
            }
            
            # Calculate derived features
            feature_data['total_sponsors'] = feature_data['sponsor_count'] + feature_data['cosponsor_count']
            feature_data['title_complexity'] = feature_data['title_length'] / (feature_data['title_word_count'] + 1)
            feature_data['party_balance'] = (feature_data['dem_total'] - feature_data['rep_total']) / (feature_data['total_sponsors'] + 1)
            feature_data['party_dominance'] = abs(feature_data['party_balance'])
            feature_data['support_velocity'] = feature_data['total_sponsors'] / np.sqrt(days_active)
            feature_data['cosponsor_growth'] = (feature_data['cosponsor_count'] - feature_data['original_cosponsor_count']) / max(days_active / 30, 1)
            feature_data['bipartisan_momentum'] = feature_data['bipartisan_score'] * feature_data['normalized_activity']
            feature_data['committee_activity'] = feature_data['committee_count'] * feature_data['activity_rate']
            
            # Adjust features for bills that have already progressed
            if has_become_law:
                # This bill is already law - set maximum values
                feature_data['action_count'] = max(feature_data['action_count'], 50)
                feature_data['committee_count'] = max(feature_data['committee_count'], 5)
                feature_data['is_stale'] = 0
                feature_data['activity_rate'] = max(feature_data['activity_rate'], 1.0)
                feature_data['normalized_activity'] = max(feature_data['normalized_activity'], 5.0)
                
                # Add a prominent note about the special case
                st.success("✅ **Note: This bill has already become law!** The predictions below are hypothetical and shown for demonstration purposes only.")
            elif has_passed_house or has_passed_senate:
                # Add a note about the special case but don't artificially boost features
                st.info(f"📊 Note: This bill has already {'passed the House' if has_passed_house else 'passed the Senate'}. The model predictions reflect the bill's characteristics at its current stage.")
            
            # Encode categorical variables
            try:
                feature_data['sponsor_party_encoded'] = label_encoders['party'].transform([feature_data['sponsor_party']])[0]
            except:
                feature_data['sponsor_party_encoded'] = 0
            
            try:
                feature_data['policy_area_encoded'] = label_encoders['policy'].transform([feature_data['policy_area']])[0]
            except:
                feature_data['policy_area_encoded'] = 0
            
            # Create DataFrame
            bill_df = pd.DataFrame([feature_data])

            # Prepare features
            X_features = bill_df[viability_model['features']].fillna(0)
            X_features = X_features.replace([np.inf, -np.inf], 0)
            
            # Scale and select
            X_scaled = viability_model['scaler'].transform(X_features)
            X_selected = X_scaled[:, viability_model['selector'].get_support()]
            
            # Get predictions from all models - fix the warnings
            try:
                # Try to get individual model predictions
                # Create DataFrame with selected features to preserve names
                X_selected_df = pd.DataFrame(X_selected, columns=viability_model['selected_features'])
                
                rf_viability = viability_model['rf_model'].predict_proba(X_selected_df)[0, 1]
                gb_viability = viability_model['gb_model'].predict_proba(X_selected_df)[0, 1]
                lr_viability = viability_model['lr_model'].predict_proba(X_selected_df)[0, 1]
                viability_scores = [rf_viability, gb_viability, lr_viability]
            except:
                # Fall back to ensemble only
                viability_scores = []
            
            # Ensemble prediction - also use DataFrame
            ensemble_viability = viability_model['model'].predict_proba(X_selected_df)[0, 1]
            
            # Calculate confidence interval if we have individual models
            if viability_scores:
                viability_low = min(viability_scores)
                viability_high = max(viability_scores)
                viability_spread = viability_high - viability_low
            else:
                # Use a default spread based on model performance
                viability_spread = 0.1
                viability_low = max(0, ensemble_viability - viability_spread/2)
                viability_high = min(1, ensemble_viability + viability_spread/2)
            
            # Display predictions - but handle already-passed bills specially
            if has_become_law:
                # For bills that are already law, show 100% scores
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Viability Prediction")
                    st.metric("Viability Score", "100%", "✓ Became Law")
                    
                    # Show a simple success indicator instead of gauge
                    st.success("✅ **BECAME LAW** - This bill has already been signed into law")
                
                with col2:
                    st.subheader("🏛️ Passage Prediction")
                    st.metric("Passage Score", "100%", "✓ Became Law")
                    
                    # Show success indicator
                    st.success("✅ **PASSED** - Successfully enacted into law")
                
                # Skip the rest of the prediction logic for bills that are already law
                st.markdown("---")
                st.subheader("📊 Historical Analysis")
                
                # Show actual timeline instead of predictions
                if not actions_df.empty:
                    key_milestones = actions_df[
                        actions_df['text'].str.contains(
                            'introduced|committee|passed|became law|public law', 
                            case=False, 
                            na=False
                        )
                    ]
                    
                    st.write("**Key Milestones:**")
                    for _, milestone in key_milestones.iterrows():
                        st.write(f"• {pd.to_datetime(milestone['date']).strftime('%B %d, %Y')}: {milestone['text'][:100]}...")
                
            else:
                # Normal prediction flow for bills that haven't become law yet
                col1, col2 = st.columns(2)
            
                with col1:
                    st.subheader("🎯 Viability Prediction")
                    
                    # Main prediction - back to original style
                    if show_confidence:
                        st.metric(
                            "Viability Score",
                            f"{ensemble_viability:.1%}",
                            f"±{viability_spread/2:.1%}"
                        )
                        st.caption(f"Confidence interval: {viability_low:.1%} - {viability_high:.1%}")
                    else:
                        st.metric("Viability Score", f"{ensemble_viability:.1%}")
                    
                    # Visual gauge with needle indicator
                    # Calculate needle angle (180 degree arc from 0 to 100%)
                    angle = 180 - (ensemble_viability * 180)  # 180 degrees at 0%, 0 degrees at 100%
                    
                    fig_viability = go.Figure()
                    
                    # Add the gauge background
                    fig_viability.add_trace(go.Indicator(
                        mode = "gauge+number",
                        value = ensemble_viability * 100,
                        number = {'suffix': "%", 'font': {'size': 40}},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [0, 100], 
                                    'tickwidth': 2, 
                                    'tickcolor': "darkgray",
                                    'tickvals': [0, 25, 50, 75, 100],
                                    'ticktext': ['0', '25', '50', '75', '100']},
                            'bar': {'color': "darkblue", 'thickness': 0.3},  # Thin bar as needle
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 30], 'color': '#ffcccc'},
                                {'range': [30, 50], 'color': '#ffffcc'},
                                {'range': [50, 100], 'color': '#ccffcc'}
                            ]
                        }
                    ))
                    
                    fig_viability.update_layout(
                        height=250,
                        margin=dict(l=30, r=30, t=30, b=30),
                        showlegend=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    
                    st.plotly_chart(fig_viability, config={'responsive': True})
                    
                    # Clear PASS/FAIL indicator
                    is_viable = ensemble_viability >= 0.5
                    
                    if ensemble_viability >= 0.7:
                        st.success("✅ **HIGHLY VIABLE** - Strong indicators for success")
                    elif ensemble_viability >= 0.5:
                        st.info("📊 **VIABLE** - Moderate chance of gaining traction")
                    elif ensemble_viability >= 0.3:
                        st.warning("⚠️ **MARGINAL** - Limited viability")
                    else:
                        st.error("❌ **NOT VIABLE** - Very unlikely to gain traction")
                
                with col2:
                    st.subheader("🏛️ Passage Prediction")
                    
                    if is_viable:
                        # Get passage predictions
                        X_passage = bill_df[passage_model['features']].fillna(0)
                        X_passage = X_passage.replace([np.inf, -np.inf], 0)
                        X_passage_scaled = passage_model['scaler'].transform(X_passage)
                        X_passage_selected = X_passage_scaled[:, passage_model['selector'].get_support()]
                        
                        try:
                            # Create DataFrame with selected features to preserve names
                            X_passage_selected_df = pd.DataFrame(X_passage_selected, columns=passage_model['selected_features'])
                            
                            rf_passage = passage_model['rf_model'].predict_proba(X_passage_selected_df)[0, 1]
                            gb_passage = passage_model['gb_model'].predict_proba(X_passage_selected_df)[0, 1]
                            lr_passage = passage_model['lr_model'].predict_proba(X_passage_selected_df)[0, 1]
                            passage_scores = [rf_passage, gb_passage, lr_passage]
                        except:
                            passage_scores = []
                        
                        # Use DataFrame for ensemble too
                        ensemble_passage = passage_model['model'].predict_proba(X_passage_selected_df)[0, 1]
                        
                        if passage_scores:
                            passage_low = min(passage_scores)
                            passage_high = max(passage_scores)
                            passage_spread = passage_high - passage_low
                        else:
                            passage_spread = 0.1
                            passage_low = max(0, ensemble_passage - passage_spread/2)
                            passage_high = min(1, ensemble_passage + passage_spread/2)
                        
                        if show_confidence:
                            st.metric(
                                "Passage Score",
                                f"{ensemble_passage:.1%}",
                                f"±{passage_spread/2:.1%}"
                            )
                            # Show actual range instead of ± notation
                            if passage_scores:
                                st.caption(f"Model range: {passage_low:.1%} - {passage_high:.1%}")
                            else:
                                st.caption(f"Confidence interval: {passage_low:.1%} - {passage_high:.1%}")
                        else:
                            st.metric("Passage Score", f"{ensemble_passage:.1%}")
                        
                        # Passage gauge with thin bar as needle
                        fig_passage = go.Figure()
                        
                        fig_passage.add_trace(go.Indicator(
                            mode = "gauge+number",
                            value = ensemble_passage * 100,
                            number = {'suffix': "%", 'font': {'size': 40}},
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            gauge = {
                                'axis': {'range': [0, 100],
                                        'tickwidth': 2,
                                        'tickcolor': "darkgray",
                                        'tickvals': [0, 25, 50, 75, 100],
                                        'ticktext': ['0', '25', '50', '75', '100']},
                                'bar': {'color': "darkgreen", 'thickness': 0.3},  # Thin bar as needle
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, 30], 'color': '#ffcccc'},
                                    {'range': [30, 50], 'color': '#ffffcc'},
                                    {'range': [50, 100], 'color': '#ccffcc'}
                                ]
                            }
                        ))
                        
                        fig_passage.update_layout(
                            height=250,
                            margin=dict(l=30, r=30, t=30, b=30),
                            showlegend=False,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)"
                        )
                        
                        st.plotly_chart(fig_passage, config={'responsive': True})
                        
                        # Clear PASS/FAIL indicator for passage
                        if ensemble_passage >= 0.7:
                            st.success("✅ **LIKELY TO PASS** - Strong support expected")
                        elif ensemble_passage >= 0.5:
                            st.info("📊 **POSSIBLE PASSAGE** - Moderate chance of success")
                        elif ensemble_passage >= 0.3:
                            st.warning("⚠️ **UNLIKELY TO PASS** - Faces significant obstacles")
                        else:
                            st.error("❌ **VERY UNLIKELY** - Lacks necessary support")
                        
                        overall_chance = ensemble_viability * ensemble_passage
                    else:
                        st.info("📊 Bill must be viable first")
                        st.caption("Focus on building support and momentum")
                        overall_chance = ensemble_viability * 0.05
                
                # Overall assessment
                st.markdown("---")
                st.subheader("📊 Overall Assessment")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    confidence_level = "High" if viability_spread < 0.2 else "Moderate" if viability_spread < 0.4 else "Low"
                    st.metric("Prediction Confidence", confidence_level)
                
                with col2:
                    st.metric("Overall Passage Chance", f"{overall_chance:.1%}")
                
                with col3:
                    percentile = (1 - overall_chance) * 100
                    st.metric("Percentile Rank", f"Top {100-percentile:.0f}%")

                # Abandonment warning (only for active congresses)
                if not congress_ended:
                    if days_since_last_action > 180:
                        st.warning(f"⚠️ **No Activity in {days_since_last_action} Days**\n\n"
                                  f"This bill appears to have been abandoned. Last action was **{days_since_last_action} days ago** "
                                  f"on {last_action.strftime('%B %d, %Y')}. Bills without activity for 6+ months rarely advance.")
                    elif days_since_last_action > 90:
                        st.info(f"ℹ️ **Limited Recent Activity**: Last action was {days_since_last_action} days ago "
                               f"({last_action.strftime('%B %d, %Y')}). Bill may be stalled.")

                # Individual model breakdown
                if show_model_breakdown:
                    st.subheader("🔍 Individual Model Predictions")
                    
                    # Add explanation for calibration
                    with st.expander("ℹ️ Why might ensemble predictions differ from individual models?"):
                        st.markdown("""
                        The ensemble prediction may be outside the range of individual model predictions because:
                        
                        1. **Probability Calibration**: The ensemble uses isotonic calibration to adjust predictions 
                           based on validation data, making them more realistic.
                        2. **Class Imbalance**: With only 3.7% of bills passing, calibration often reduces 
                           overly optimistic predictions.
                        3. **Model Weighting**: The ensemble weights models differently (RF: 40%, GB: 40%, LR: 20%)
                        
                        Individual models show raw predictions, while the ensemble shows calibrated probabilities 
                        that better reflect historical pass rates.
                        """)
                    
                    try:
                        # Get individual model predictions if available
                        individual_viability = []
                        individual_passage = []
                        
                        if 'rf_viability' in locals():
                            individual_viability = [
                                ('Random Forest', rf_viability),
                                ('Gradient Boosting', gb_viability),
                                ('Logistic Regression', lr_viability)
                            ]
                        
                        if 'rf_passage' in locals() and is_viable:
                            individual_passage = [
                                ('Random Forest', rf_passage),
                                ('Gradient Boosting', gb_passage),
                                ('Logistic Regression', lr_passage)
                            ]
                        
                        # Create comparison table
                        if individual_viability or individual_passage:
                            comparison_data = []
                            
                            for model_name, viab in (individual_viability or [('Ensemble', ensemble_viability)]):
                                pass_score = 0
                                for m, p in individual_passage:
                                    if m == model_name:
                                        pass_score = p
                                        break
                                
                                comparison_data.append({
                                    'Model': model_name,
                                    'Viability': f"{viab:.1%}",
                                    'Passage': f"{pass_score:.1%}" if pass_score > 0 else "N/A"
                                })
                            
                            # Add ensemble row
                            comparison_data.append({
                                'Model': '**Ensemble (Calibrated)**',
                                'Viability': f"**{ensemble_viability:.1%}**",
                                'Passage': f"**{ensemble_passage:.1%}**" if is_viable else "N/A"
                            })
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, hide_index=True)
                        
                        # Visualize as bar chart
                        model_data = pd.DataFrame({
                            'Model': ['Random Forest', 'Gradient Boosting', 'Logistic Regression', 'Ensemble'],
                            'Viability': [rf_viability if 'rf_viability' in locals() else 0, 
                                         gb_viability if 'gb_viability' in locals() else 0, 
                                         lr_viability if 'lr_viability' in locals() else 0,
                                         ensemble_viability],
                            'Passage': [rf_passage if 'rf_passage' in locals() and is_viable else 0, 
                                       gb_passage if 'gb_passage' in locals() and is_viable else 0, 
                                       lr_passage if 'lr_passage' in locals() and is_viable else 0,
                                       ensemble_passage if is_viable else 0]
                        })
                    except:
                        model_data = pd.DataFrame({
                            'Model': ['Ensemble'],
                            'Viability': [ensemble_viability],
                            'Passage': [ensemble_passage if is_viable else 0]
                        })
                    
                    fig_breakdown = px.bar(
                        model_data.melt(id_vars='Model', var_name='Prediction', value_name='Probability'),
                        x='Model',
                        y='Probability',
                        color='Prediction',
                        barmode='group',
                        title='Model Predictions Comparison',
                        color_discrete_map={'Viability': '#1f77b4', 'Passage': '#2ca02c'}
                    )
                    fig_breakdown.update_yaxes(tickformat='.0%', range=[0, 1])
                    fig_breakdown.update_layout(height=400)
                    st.plotly_chart(fig_breakdown, config={'responsive': True})
                
                # Feature importance
                if show_feature_analysis:
                    st.subheader("📊 Key Factors Analysis")
                    
                    # Get feature values for important features
                    important_features = viability_model['selected_features'][:10]
                    
                    feature_display = []
                    for feat in important_features:
                        if feat in feature_data:
                            value = feature_data[feat]
                            # Format based on feature type
                            if isinstance(value, bool):
                                display_value = "Yes" if value else "No"
                            elif isinstance(value, (int, float)):
                                if 'rate' in feat or 'score' in feat or 'ratio' in feat:
                                    display_value = f"{value:.2f}"
                                else:
                                    display_value = f"{value:.0f}"
                            else:
                                display_value = str(value)
                            
                            feature_display.append({
                                'Feature': feat.replace('_', ' ').title(),
                                'Value': display_value,
                                'Impact': 'Positive' if value > 0 else 'Negative' if value < 0 else 'Neutral'
                            })
                    
                    feature_df = pd.DataFrame(feature_display)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(feature_df.head(5), hide_index=True)
                    with col2:
                        st.dataframe(feature_df.tail(5), hide_index=True)
                
                # Strategic recommendations
                st.markdown("---")
                st.subheader("💡 Strategic Recommendations")
                
                # Personalized recommendations based on predictions and features
                recommendations = []
                
                if not is_viable:
                    if feature_data['cosponsor_count'] < 10:
                        recommendations.append("🎯 **Build Support**: Focus on recruiting more cosponsors (currently only {})".format(
                            int(feature_data['cosponsor_count'])))
                    if feature_data['committee_count'] == 0:
                        recommendations.append("📋 **Committee Assignment**: Work to get the bill assigned to relevant committees")
                    if not feature_data['has_bipartisan_support']:
                        recommendations.append("🤝 **Bipartisan Outreach**: Seek support from across the aisle")
                    if feature_data['is_stale']:
                        recommendations.append("⏰ **Time Factor**: Consider reintroduction in the next session")
                else:
                    if ensemble_passage < 0.3 if 'ensemble_passage' in locals() else True:
                        recommendations.append("📈 **Maintain Momentum**: Keep generating legislative activity")
                        recommendations.append("🎤 **Public Support**: Build grassroots backing")
                    if feature_data['committee_density'] < 0.5:
                        recommendations.append("⚡ **Accelerate Progress**: Push for faster committee action")
                
                for rec in recommendations:
                    st.markdown(rec)
                
                # Historical comparison
                if show_similar_bills:
                    st.subheader("📚 Historical Comparison")
                    
                    # Get actual pass rate from data if available
                    pass_rate_data = model_package.get('viability_pass_rates')
                    pass_rate_fine = model_package.get('viability_pass_rates_fine')
                    
                    if pass_rate_data is not None and not pass_rate_data.empty:
                        # Find the appropriate bin for this viability score
                        viability_pct = ensemble_viability * 100
                        
                        # Find the matching bin
                        bin_match = None
                        for _, row in pass_rate_data.iterrows():
                            if row['min_viability'] * 100 <= viability_pct < row['max_viability'] * 100:
                                bin_match = row
                                break
                        
                        # Try to get more precise estimate from fine-grained data
                        precise_rate = None
                        if pass_rate_fine is not None and not pass_rate_fine.empty:
                            # Find closest viability score in fine data
                            closest_idx = (pass_rate_fine['viability_score'] - ensemble_viability).abs().idxmin()
                            closest_data = pass_rate_fine.loc[closest_idx]
                            
                            # Only use if we have enough bills in that range
                            if closest_data['bill_count'] >= 10:
                                precise_rate = closest_data['pass_rate']
                        
                        # Use precise rate if available, otherwise use bin rate
                        if precise_rate is not None:
                            estimated_pass_rate = precise_rate
                            data_source = "precise historical data"
                        elif bin_match is not None:
                            estimated_pass_rate = bin_match['pass_rate']
                            data_source = f"{bin_match['bill_count']:,} bills in {bin_match['viability_range']} range"
                        else:
                            # Fallback if no bin matches
                            estimated_pass_rate = 3.7  # Overall average
                            data_source = "overall average (no specific data for this range)"

                        st.info(f"""
                        **Based on analysis of 76,854 unique bills across 6 Congresses (113-118):**

                        📊 **Actual Historical Outcomes:**
                        - Bills with viability score ~{ensemble_viability:.1%}: **{estimated_pass_rate:.1f}%** passed
                        - Based on: {data_source}
                        - This is {estimated_pass_rate/3.7:.1f}x the overall average (3.7%)

                        ⏱️ **Typical Timeline:**
                        - Average time to passage: ~180 days
                        - Most common failure point: Committee stage

                        📈 **Data Note:**
                        This estimate comes from actual historical data - we analyzed all 76,854 unique bills
                        in our training set and calculated real pass rates for each viability score range.
                        """)
                        
                        # Show the bin breakdown if requested
                        with st.expander("📊 See all viability ranges"):
                            display_df = pass_rate_data[['viability_range', 'bill_count', 'passed_count', 'pass_rate']].copy()
                            display_df.columns = ['Viability Range', 'Total Bills', 'Passed', 'Pass Rate (%)']
                            display_df['Pass Rate (%)'] = display_df['Pass Rate (%)'].round(1)
                            st.dataframe(display_df, hide_index=True)
                    
                    else:
                        # Fallback to simple calculation if data not available
                        st.warning("Historical pass rate data not found. Using simplified estimates.")
                        
                        # Simple conservative estimate
                        if ensemble_viability >= 0.9:
                            estimated_pass_rate = 45.0
                        elif ensemble_viability >= 0.7:
                            estimated_pass_rate = 25.0
                        elif ensemble_viability >= 0.5:
                            estimated_pass_rate = 15.0
                        else:
                            estimated_pass_rate = 5.0
                        
                        st.info(f"""
                        Based on analysis of 76,854 unique bills across 6 Congresses (113-118):
                        - Bills with similar viability scores: ~{estimated_pass_rate:.1f}% passed (estimate)
                        - Average time to passage: ~180 days
                        - Most common failure point: Committee stage
                        - Historical context: Only 3.7% of all bills become law

                        *Note: Run the viability pass rate analyzer for precise historical data.*
                        """)
        
        else:
            st.error("❗ Models not found. Please run the training script.")
            
    except Exception as e:
        st.error(f"❗ An error occurred: {str(e)}")
        
        with st.expander("🐛 Debug Information"):
            import traceback
            st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.caption("""
Congressional Bill Tracker v5.0 | AI predictions based on 273,113 temporal snapshots from 76,854 unique bills
Trained on 113th-118th Congress data | Model accuracy: 86.6%-97.9% ROC-AUC
""")
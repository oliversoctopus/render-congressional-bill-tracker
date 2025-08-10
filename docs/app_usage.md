# Dashboard Usage Guide

## Overview
The Congressional Bill Tracker provides real-time AI predictions for legislative success using models trained on **76,897 bills from 6 U.S. Congresses (113th-118th, 2013-2025)**.

- **Main Application**: `src/app.py` (Streamlit dashboard)
- **Data Source**: Congress.gov API via `src/data_fetch.py`
- **ML Models**: Pre-trained ensemble models with 88-99% ROC-AUC accuracy

## Access Options

### Option 1: Use the Deployed App (Recommended)
Visit the live dashboard at: https://predictivebilltracker.streamlit.app
- No installation required
- Always up-to-date with latest models
- Instant access with full functionality

### Option 2: Run Locally
1. **Install dependencies**: 
   ```bash
   pip install -r requirements.txt
   ```
2. **Set up API key** (optional for enhanced features):
   ```bash
   echo "CONGRESS_API_KEY=your_key_here" > .env
   ```
3. **Launch the app**:
   ```bash
   streamlit run src/app.py
   ```
4. **Open browser**: Navigate to `http://localhost:8501`

## How to Use

### Input Parameters
- **Bill Number**: Enter the numeric identifier (e.g., 1234)
- **Type**: Select HR (House), S (Senate), HJRES, or SJRES
- **Congress**: Select from 113th-118th (default is 118th)
  - Note: Model trained on all 6 congresses for maximum accuracy

### Understanding the Output

#### 1. Bill Verification Section
- Confirms bill title, sponsor, and current status
- Shows policy area and committee assignments
- Helps verify you're analyzing the correct bill

#### 2. Key Metrics Dashboard
- **Days Active**: Time since introduction
- **Total Actions**: Legislative activity count
- **Committees**: Number of committee referrals
- **Cosponsors**: Support level indicator
- **Bipartisan**: Cross-party support status
- **Activity Rate**: Actions per day metric

#### 3. Legislative Timeline
- Complete chronological action history
- Paginated display for bills with extensive histories (20 actions per page)
- Download option for CSV export
- Shows action text, date, and days ago

#### 4. AI Predictions (Enhanced Accuracy)
- **Viability Score**: Likelihood of gaining traction (0-100%)
  - New Bill Model: 88.6% accuracy
  - Early Stage Model: 99.4% accuracy
  - Progressive Model: 99.6% accuracy
- **Passage Score**: Chance of becoming law (for viable bills)
  - Focused prediction on the 16% of bills deemed viable
- **Confidence Intervals**: Bootstrap-based uncertainty ranges
- **Visual Gauges**: Color-coded prediction displays
  - Green (>70%): High likelihood
  - Yellow (30-70%): Moderate chance
  - Red (<30%): Low probability

#### 5. Advanced Features (Optional)
- **Model Breakdown**: Individual predictions from Random Forest, Gradient Boosting, and Logistic Regression
- **Feature Analysis**: Top factors influencing prediction with importance scores
- **Similar Bills**: Historical comparison from 6-congress dataset
- **Viability-to-Passage Correlation**: Visual chart showing historical pass rates by viability score

#### 6. Strategic Recommendations
- Data-driven action items based on current status
- Focus areas for improving bill success
- Timing considerations based on legislative calendar
- Historical context from similar bills across 6 congresses

## Display Options
Toggle these features via the expandable menu:
- Show/hide confidence intervals
- Display individual model predictions
- View feature importance analysis
- Compare with similar historical bills
- View viability-passage correlation charts

## Tips for Best Results
- **Optimal timing**: Bills with 30+ days of activity provide most accurate predictions (99.6% ROC-AUC)
- **Early predictions**: Even day-1 predictions achieve 88.6% accuracy
- **Historical context**: Models trained on 12+ years of data capture long-term patterns
- **Confidence check**: Always review confidence intervals for prediction reliability
- **Export options**: Use CSV export for detailed analysis and reporting

## Model Performance by Stage
| Bill Age | Model Used | ROC-AUC | Accuracy |
|----------|------------|---------|----------|
| Day 1 | New Bill | 0.886 | 88.8% |
| Days 2-30 | Early Stage | 0.994 | 98.5% |
| Day 30+ | Progressive | 0.996 | 99.1% |

## Troubleshooting
- **"Model not found"**: Ensure `models/` directory contains all optimized component files
- **Slow initial load**: First prediction loads ~132MB of models (10-15 seconds)
- **No data found**: Verify bill number and congress session (113th-118th supported)
- **API limits**: Consider using local deployment with API key for heavy usage
- **Memory issues**: Models are split into components for efficient loading

## Understanding Model Improvements
The enhanced 6-congress model offers:
- **4.5x more training data** (77K vs 16K bills)
- **Cross-congress validation** ensuring robust predictions
- **Historical patterns** from different political environments
- **Congress-aware features** accounting for legislative period differences

## Deployment
For your own instance:
1. Fork the repository
2. Deploy via [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Set secrets in Streamlit dashboard for API keys
4. Models auto-load from optimized component structure

## Support
- **Documentation**: See `/docs` folder for detailed technical documentation
- **Issues**: Report via GitHub Issues
- **Contact**: oliver.s.fan@gmail.com
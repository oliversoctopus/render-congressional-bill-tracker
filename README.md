# Predictive Bill Tracker Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://congressionalbilltracker.streamlit.app/)

## Overview
This project is an AI-powered legislative tracking tool submitted to the **AI4Legislation 2025** competition in the "Legislative Tracking" category. It provides a dashboard to predict the viability and passage probability of U.S. Congressional bills using machine learning models trained on comprehensive data from **6 congresses (113th-118th, 2013-2025)**.

Key Features:
- Fetches real-time bill data from the Congress.gov API.
- Predicts bill viability (likelihood of gaining traction) and passage probability with confidence intervals.
- Visualizes bill timelines, sponsor details, and key metrics.
- Time-aware models: Different predictions based on bill age (new, early-stage, progressive).
- **Enhanced model accuracy** through training on ~77,000 bills across 6 congresses.
- Built with Streamlit for an interactive UI.

The tool helps citizens, advocates, and policymakers track bills, understand their chances of success, and identify factors influencing outcomes. It addresses challenges in legislative transparency by simplifying complex data and providing actionable insights.

**Competition Alignment**:
- **Innovation**: Dual-stage prediction (viability then passage) with ensemble models and confidence ranges, trained on extensive multi-congress dataset.
- **Impact**: Empowers civic engagement by demystifying bill progress; enhanced reliability through historical pattern analysis across 12+ years.
- **Technical Excellence**: Uses NLP for bill analysis, ensemble ML for predictions, and API integration with cross-congress validation.
- **Usability**: Intuitive dashboard with metrics, timelines, and recommendations.
- **Ethics**: Bias mitigation in models, transparent predictions, and data privacy focus.

For full details, see [docs/project_report.md](docs/project_report.md).

## Demo Video
Watch the 7-minute demonstration: [YouTube Demonstration Video](https://www.youtube.com/watch?v=oLsZL_xDgDU&ab_channel=OliverFan)

This video covers: Project motivation, technical architecture, dashboard demo, societal impact, and ideas for the future.

## Installation & Setup
### Option 1
Open the deployed Streamlit website with preloaded code and dependencies: https://congressionalbilltracker.streamlit.app

*Note: If the website is closed due to inactivity, click "Yes, get this app back up!" and wait 1-2 minutes for the necessary dependencies to be installed.*

### Option 2
Or, if you would like to set up the code yourself, then you can do the following:
1. Clone the repo: <br>git clone https://github.com/oliversoctopus/predictive-bill-tracker-dashboard</br>
cd predictive-bill-tracker-dashboard
2. Install dependencies (Python 3.8+ required):
pip install -r requirements.txt  
3. Set up environment variables (in `.env` file):
CONGRESS_API_KEY=your_api_key_here  
4. Run the app:
streamlit run src/app.py

For data extraction and model training, see [docs/data.md](docs/data.md) and [docs/model_training.md](docs/model_training.md).

## Project Structure
- **data/**: Notebooks for data extraction and preprocessing from Congress API; includes 6-congress preprocessing notebook.
- **models/**: Notebook for training ML models; saved models in optimized PKL format.
- **src/**: Core application code (Streamlit dashboard and data fetching).
- **docs/**: Detailed documentation (report, data, models, ethics, etc.).

## Datasets
- Sourced from Congress.gov API (113th-118th Congress bills: HR, S, HJRES, SJRES).
- Processed datasets: `bills_6congress_training.csv` (~77,000 bills with outcomes).
- Viability analysis: `viability_pass_rates.csv` and `viability_pass_rates_fine.csv`.
- Details: See [docs/data.md](docs/data.md).

## AI Models
- Ensemble models (Random Forest, Gradient Boosting, Logistic Regression) for viability and passage prediction.
- **Trained on ~77,000 bills from 6 congresses**; handles class imbalance with calibration.
- Saved in `models/` as optimized split PKL components for efficient loading.
- **Improved performance**: ROC-AUC scores of 0.88-0.99 across different model stages.
- Details: See [docs/model_training.md](docs/model_training.md).

## Ethical Considerations
See [docs/ethics.md](docs/ethics.md) for bias mitigation, transparency, and societal impact.

## License & Attributions
- **License**: MIT + Commons Clause (see LICENSE file).
- **Attributions**: Congress.gov API, scikit-learn, Streamlit, Plotly. Full list in [docs/attributions.md](docs/attributions.md).

## Competition Submission
- Category: Legislative Tracking.

Questions? Contact oliver.s.fan@gmail.com
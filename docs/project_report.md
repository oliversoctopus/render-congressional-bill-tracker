# Project Report: Predictive Bill Tracker Dashboard

## Executive Summary
An AI-powered legislative tracking tool that predicts bill viability and passage probability using ensemble machine learning models trained on **76,897 bills from 6 U.S. Congresses (113th-118th, 2013-2025)**. The enhanced dataset provides unprecedented accuracy and historical validation for legislative predictions.

## 1. Problem Statement and Motivation

### The Challenge
- **Volume**: Each Congress introduces 10,000-16,000+ bills; only 2-4% become law
- **Complexity**: Bills navigate committees, amendments, and political dynamics
- **Opacity**: Citizens struggle to understand which bills matter and why
- **Historical patterns**: Single-congress models miss long-term legislative trends

### Our Solution
A comprehensive predictive system that:
- Analyzes bills across 12+ years of legislative history
- Provides dual-phase predictions (viability then passage)
- Offers time-aware models that improve as bills progress
- Delivers actionable insights for civic engagement

### Target Users
- **Citizens**: Track bills affecting their interests with historical context
- **Advocates/NGOs**: Prioritize efforts based on data-driven insights
- **Policymakers**: Understand success factors across multiple congresses
- **Researchers**: Access comprehensive legislative analytics

## 2. Technical Approach

### Data Pipeline
- **Source**: Congress.gov API for 6 congresses (113th-118th)
- **Scale**: ~77,000 bills with complete outcomes
- **Processing**: 
  - Unified preprocessing across congresses
  - Congress-aware feature engineering
  - Temporal normalization for cross-congress comparison

### Model Architecture

#### Two-Phase System
1. **Viability Prediction** (16% positive rate)
   - Identifies bills likely to gain traction
   - Filters noise for focused passage prediction

2. **Passage Prediction** (16.7% of viable bills)
   - Concentrated on viable candidates
   - Higher precision for rare positive class

#### Time-Aware Stages
- **New Bill** (Day 1): Initial characteristics, ROC-AUC 0.886
- **Early Stage** (Days 2-30): Momentum indicators, ROC-AUC 0.994
- **Progressive** (30+ days): Full history, ROC-AUC 0.996

#### Ensemble Methods
- Random Forest (40%): Captures non-linear patterns
- Gradient Boosting (40%): Sequential error correction
- Logistic Regression (20%): Baseline stability
- Isotonic Calibration: Realistic probability adjustment

### Key Features
- **Congress-specific**: congress_numeric, is_recent_congress
- **Temporal**: early_activity, sustained_activity, momentum
- **Political**: bipartisan_score, party_balance
- **Procedural**: committee_count, action_rate

### Dashboard Implementation
- **Framework**: Streamlit for rapid deployment
- **Visualizations**: Plotly for interactive charts
- **API Integration**: Real-time Congress.gov data
- **Model Loading**: Optimized component architecture

## 3. Results and Evaluation

### Model Performance

| Stage | Viability ROC-AUC | Passage ROC-AUC | CV Score |
|-------|------------------|-----------------|----------|
| New Bill | 0.886 | 0.812 | 0.888±0.008 |
| Early Stage | 0.994 | 0.976 | 0.994±0.002 |
| Progressive | 0.996 | 0.982 | 0.996±0.001 |

### Cross-Congress Validation
Consistent performance across different political environments:
- 113th Congress: 3.2% passage (Obama, divided)
- 114th Congress: 3.3% passage (Obama, GOP control)
- 115th Congress: 3.8% passage (Trump, GOP control)
- 116th Congress: 2.4% passage (Trump, divided)
- 117th Congress: 2.4% passage (Biden, Dem control)
- 118th Congress: 1.7% passage (Biden, divided)

### Impact Metrics
- **Early identification**: 85% of passed bills flagged as viable within 30 days
- **Precision at threshold**: 81% precision for high-confidence predictions
- **Historical validation**: Patterns hold across 12+ years

### Usability Features
- **Visual predictions**: Color-coded probability gauges
- **Confidence intervals**: Bootstrap-based uncertainty
- **Historical context**: Similar bills from 6-congress dataset
- **Actionable recommendations**: Data-driven improvement suggestions

## 4. Ethical Considerations

### Bias Mitigation
- **Multi-congress training**: Reduces temporal and political biases
- **Party-neutral features**: Equal performance across party lines
- **Transparent methodology**: Open-source code and documentation

### Privacy & Security
- **Public data only**: No personal information collected
- **No user tracking**: Privacy-preserving design
- **API compliance**: Respects rate limits and terms

### Societal Impact
- **Democratization**: Makes legislative analysis accessible
- **Empowerment**: Enables data-driven advocacy
- **Education**: Explains legislative process alongside predictions

## 5. Innovation & Technical Excellence

### Key Innovations
1. **6-Congress Dataset**: Unprecedented scope for legislative ML
2. **Dual-Phase Architecture**: Separates viability from passage
3. **Time-Aware Models**: Adaptive predictions as bills evolve
4. **Optimized Components**: Efficient model storage and loading

### Technical Achievements
- **Scalability**: Handles 77K+ bills efficiently
- **Accuracy**: 99%+ ROC-AUC for mature bill predictions
- **Robustness**: Validated across diverse political periods
- **Interpretability**: Feature importance and confidence metrics

## 6. Future Enhancements

### Short-term (3-6 months)
- Real-time webhook integration for instant updates
- Mobile application development
- State legislature expansion (starting with California, Texas)

### Medium-term (6-12 months)
- NLP analysis of bill text for semantic understanding
- Network analysis of sponsor relationships
- Amendment tracking and impact prediction

### Long-term (1+ years)
- International expansion (parliamentary systems)
- Predictive policy impact assessment
- AI-assisted bill drafting recommendations

## 7. Competition Alignment

### Legislative Tracking Excellence
- **Comprehensive coverage**: 6 congresses, 77K bills
- **Proven accuracy**: 99%+ ROC-AUC for developed bills
- **User-friendly**: Intuitive dashboard with actionable insights
- **Ethical design**: Privacy-preserving, bias-mitigated

### Measurable Impact
- Potential to influence civic engagement for millions
- Cost savings for advocacy organizations through prioritization
- Educational value for understanding democracy

## Conclusion
The Predictive Bill Tracker represents a significant advancement in legislative analytics, leveraging comprehensive historical data to provide accurate, actionable predictions. By training on 6 congresses, the system offers unprecedented reliability and insights into the legislative process, empowering citizens, advocates, and policymakers to engage more effectively with democracy.

## Appendices

### A. Technical Specifications
- **Languages**: Python 3.8+
- **Frameworks**: Streamlit, scikit-learn, Plotly
- **Data Size**: ~77K bills, 50MB processed dataset
- **Model Size**: ~132MB total (optimized components)
- **Response Time**: <2 seconds for predictions

### B. Repository Structure
```
predictive-bill-tracker/
├── data/           # Data processing notebooks
├── models/         # Trained models and training code
├── src/            # Application source code
├── docs/           # Documentation
└── requirements.txt
```

### C. Deployment
- **Platform**: Streamlit Community Cloud
- **URL**: https://congressionalbilltracker.streamlit.app
- **Availability**: 24/7 with auto-scaling
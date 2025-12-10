# Warehouse Flow Optimizer

A decision-support system that detects operational bottlenecks in warehouse workflows, predicts future constraints, and recommends actionable improvements.

## Problem Statement

Warehouses operate as flow systems where orders move through stages: **Receiving → Picking → Packing → Sorting → Shipping**. When one stage becomes a bottleneck, throughput suffers across the entire operation. This project provides:

1. **Bottleneck Detection** - Identifies which stage is currently constraining throughput
2. **Predictive Analytics** - Forecasts which stage will become the bottleneck in the next shift
3. **What-If Analysis** - Simulates operational changes and quantifies their impact
4. **Optimization Recommendations** - Ranks actions by expected throughput improvement

## Key Features

### Bottleneck Detection Engine
- Multi-signal scoring system using queue time, utilization, cycle time variability
- Confidence scoring for detection reliability
- Cost impact estimation (delay costs, lost throughput)

### ML Prediction Model
- Random Forest classifier trained on operational features
- Predicts next-hour bottleneck with feature importance analysis
- Risk factor identification for proactive intervention

### What-If Scenario Analyzer
- Simulate adding workers to a stage
- Model cycle time improvements (process optimization)
- Evaluate worker reallocation strategies
- Quantified throughput and lead time impact

### Interactive Dashboard
- Real-time bottleneck visualization
- Utilization heatmaps and queue time analysis
- Scenario simulation interface
- Ranked recommendation engine

## Tech Stack

- **Python 3.10+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine learning models
- **Streamlit** - Interactive dashboard
- **Plotly** - Data visualization

## Project Structure

```
warehouse-flow/
├── app.py                    # Streamlit dashboard
├── src/
│   ├── __init__.py
│   ├── data_generator.py     # Synthetic data generation
│   ├── metrics.py            # Operational metrics calculation
│   ├── bottleneck_detector.py # Detection algorithm
│   ├── predictor.py          # ML prediction model
│   └── optimizer.py          # What-if scenario analyzer
├── data/
│   ├── raw/                  # Raw event logs
│   └── processed/            # Processed datasets
├── models/                   # Saved ML models
├── tests/                    # Unit tests
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/kravipa1/warehouse-flow.git
cd warehouse-flow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the Dashboard
```bash
streamlit run app.py
```

### Use as a Library
```python
from src.data_generator import generate_dataset
from src.bottleneck_detector import detect_bottleneck
from src.optimizer import WhatIfAnalyzer

# Generate or load event log data
event_log = generate_dataset(num_days=14, bottleneck_stage="packing")

# Detect current bottleneck
analysis = detect_bottleneck(event_log)
print(f"Bottleneck: {analysis.bottleneck_stage}")
print(f"Reason: {analysis.primary_reason}")

# Simulate adding workers
analyzer = WhatIfAnalyzer(event_log)
result = analyzer.simulate_add_workers("packing", 2)
print(f"Throughput improvement: {result.throughput_change_pct:+.1f}%")
```

## Methodology

### Bottleneck Scoring
The detection algorithm scores each stage on four dimensions:
- **Queue Time Score (40%)** - Average wait time before processing
- **Utilization Score (30%)** - Worker capacity usage
- **Variability Score (15%)** - Cycle time consistency
- **Tail Score (15%)** - P95/median ratio for long-tail delays

### Prediction Features
The ML model uses:
- Order volume by stage (current and rolling averages)
- Cycle time trends
- Queue time trends
- Time-based features (hour of day, day of week, peak hours)

### What-If Simulation
Lightweight extrapolation model:
- Worker additions reduce utilization proportionally
- Queue time scales with utilization above 85%
- Bottleneck resolution triggers throughput capacity increase

## Sample Output

```
=== Bottleneck Detection ===
Bottleneck Stage: PACKING
Confidence: 87%
Primary Reason: High queue buildup (8.3 min wait)

Contributing Factors:
  • Utilization at 92.1%
  • Average queue time of 8.3 minutes

Recommendations:
  1. Add 2 worker(s) to packing to reduce utilization below 80%
  2. Pre-stage packing materials and optimize workstation layout
  3. Cross-train workers to shift to packing during peak hours
```

## License

MIT License

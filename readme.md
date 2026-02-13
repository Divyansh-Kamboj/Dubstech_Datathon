# Project Aesclepius: 2026 Healthcare Simulator
### 🏆 Built for the DubsTech Datathon

## 🏥 Overview
Project Aesclepius is a predictive simulation engine designed to model healthcare resource allocation in a post-crisis economy. By 2026, healthcare demand is projected to outpace supply. This tool allows policymakers and hospital administrators to test "Safety Net" strategies, balancing the competing goals of economic efficiency and human compassion.

## 🚀 Quick Start
1.  **Read the docs*: [Technical Explainer Notebook](tech_explainer.ipynb)
2.  **Run the Simulation**: [Launch Streamlit App](https://project-aesclepius.streamlit.app)

## 🧠 The Technology
* **Triage Engine**: XGBoost Classifier trained to predict patient mortality risk.
* **Empathy Engine**: Survival Analysis (Cox Proportional Hazards) to visualize wait times.
* **Optimization**: Linear Programming (PuLP) to solve the budget Knapsack problem.

## 📊 Data Sources
* **Clinical Training Data**: NY SPARCS 2022 (Statewide Planning and Research Cooperative System), containing 100,000+ real patient records.
* **Population Forecasts**: Based on trends from the **National Health Interview Survey (NHIS)** and CBO population projections.

## ⚠️ Key Definitions
* **Risk Factor**: A prediction of how likely a patient is to face a critical outcome if care is delayed.

## 📉 Model Accuracy & Performance
*Metrics calculated on a 20% hold-out test set from the NY SPARCS 2022 dataset.*

| Model | Metric | Value | Margin of Error (95% CI) |
| :--- | :--- | :--- | :--- |
| **XGBoost Triage** | Classification Accuracy | **49.1%** | +/- 0.3% |
| **Cox Survival (Avg)** | Concordance Index | **0.662** | N/A |
| *Mental Health Sub-Model* | *Concordance Index* | *0.752* | *High Predictive Power* |
| *Oncology Sub-Model* | *Concordance Index* | *0.705* | *High Predictive Power* |

*Note: The XGBoost accuracy reflects the difficulty of predicting exact mortality levels (4 classes) in a highly complex, noisy real-world dataset.*

## 💻 Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```


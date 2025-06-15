# 🕵️ Spot the Scam – Fraud Job Listing Detection

An interactive machine learning dashboard that detects fraudulent job postings using XGBoost and SHAP. Built for **Anveshan Hackathon 2025 (DS-1: Spot the Scam)**.

---

## 🚀 Features

- 📂 Upload raw `test.csv` and get fraud probability predictions
- 📊 Dashboard with metrics, fraud score distribution, and suspicious job listings
- 📌 Insights like:
  - Top locations with fake jobs
  - Description length vs. fraud
  - Remote+Logo pattern analysis
- 🔬 SHAP explainability with force plots for each row
- 📥 Download full prediction CSV
- 🌙 Dark-mode ready UI (custom `.streamlit/config.toml`)

---

## 🧠 Model Details

- **Algorithm**: XGBoost (binary classification)
- **Evaluation Metric**: F1-Score
- **Input**: Preprocessed features from `X_test_processed.csv`
- **Explainability**: SHAP force plot for individual predictions

---

## 📁 Repository Structure

spot-the-scam/
├── app.py
├── xgboost_model.pkl
├── X_test_processed.csv
├── requirements.txt
├── README.md
└── .streamlit/
└── config.toml
---

---

## ⚙️ Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/spot-the-scam.git
   cd spot-the-scam
2. Install dependencies:
   pip install -r requirements.txt

3. Run the app:
   streamlit run app.py

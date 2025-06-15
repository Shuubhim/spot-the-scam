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
---

## ⚙️ Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/spot-the-scam.git
   cd spot-the-scam
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the app:
   ```bash
   streamlit run app.py

## 📁 External Files

Since `X_test_processed.csv` exceeds GitHub's file size limit, it's hosted on Google Drive.

- 🔗 [Download X_test_processed.csv](https://drive.google.com/uc?export=download&id=1012u7YCKd9cm7Rcp52TKnMbrjxgII5fK)

> ⚠️ Make sure to download this file manually and place it in your root project folder before running `app.py`.


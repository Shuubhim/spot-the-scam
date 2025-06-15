# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import streamlit.components.v1 as components
from shap.plots import force

# Page setup
st.set_page_config(page_title="Spot the Scam", layout="wide")
plt.style.use("seaborn-v0_8-darkgrid")

# =============================
# 🎨 Custom CSS
# =============================
st.markdown("""
<style>
h1, h2, h3 {
    color: #FF4B4B;
}
hr {
    margin: 1rem 0;
    border: none;
    height: 1px;
    background-color: #e6e6e6;
}
</style>
""", unsafe_allow_html=True)

# =============================
# 🔬 Load Model
# =============================
@st.cache_resource

def load_model():
    return joblib.load("xgboost_model.pkl")

model = load_model()

# =============================
# 🔍 Title + Upload
# =============================
st.markdown("""
<h1 style='text-align: center;'>🕵️ Spot the Scam</h1>
<h4 style='text-align: center; color: gray;'>Fraud Job Listing Detection Dashboard with Explainability</h4>
<hr>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload your raw `test.csv` file to begin:", type=["csv"])

if uploaded_file:
    try:
        raw_test = pd.read_csv(uploaded_file)
        processed_test = pd.read_csv("X_test_processed.csv")

        preds_proba = model.predict_proba(processed_test)[:, 1]
        preds = (preds_proba >= 0.5).astype(int)

        raw_test = raw_test.loc[processed_test.index].reset_index(drop=True)
        result_df = raw_test.copy()
        result_df["fraud_probability"] = preds_proba
        result_df["predicted_label"] = preds

        # ================
        # 📊 Metrics
        # ================
        total = len(result_df)
        fraud_count = result_df["predicted_label"].sum()
        real_count = total - fraud_count
        fraud_pct = (fraud_count / total) * 100

        st.subheader("📊 Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Listings", total)
        col2.metric("Predicted Fraud", f"{fraud_count} ({fraud_pct:.1f}%)")
        col3.metric("Predicted Real", real_count)

        st.divider()

        # ================
        # 🚨 Top 10 Suspicious
        # ================
        st.subheader("🚨 Top 10 Most Suspicious Jobs")
        top_fraud = result_df.sort_values("fraud_probability", ascending=False).head(10)
        st.dataframe(top_fraud[["title", "location", "fraud_probability"]], use_container_width=True)

        st.divider()

        # ================
        # 📊 Visuals
        # ================
        colA, colB = st.columns(2)

        with colA:
            st.subheader("🧐 Real vs Fraud Ratio")
            pie_data = result_df["predicted_label"].value_counts()
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            ax1.pie(pie_data, labels=["Real", "Fraud"], autopct="%1.1f%%", colors=["green", "red"])
            ax1.set_title("Prediction Split")
            st.pyplot(fig1)
            plt.clf()

        with colB:
            st.subheader("📊 Fraud Probability Distribution")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.histplot(result_df["fraud_probability"], bins=30, kde=True, color="purple", ax=ax2)
            ax2.set_xlabel("Fraud Probability")
            ax2.set_title("Distribution of Fraud Scores")
            st.pyplot(fig2)
            plt.clf()

        st.divider()
        st.subheader("🔍 Deeper Insights")

        # Location Chart
        st.markdown("##### 📽️ Top Locations for Predicted Fraud")
        top_locations = result_df[result_df["predicted_label"] == 1]["location"].value_counts().head(10)
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.barplot(y=top_locations.index, x=top_locations.values, palette="Reds", ax=ax3)
        ax3.set_title("Top 10 Locations with Most Predicted Fraud Jobs")
        st.pyplot(fig3)
        plt.clf()

        # Description Length
        st.markdown("##### ✍️ Description Length vs Fraud Probability")
        result_df["desc_length"] = result_df["description"].fillna("").apply(len)
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        sns.scatterplot(data=result_df, x="desc_length", y="fraud_probability", hue="predicted_label",
                        palette={0: "green", 1: "red"}, alpha=0.6, ax=ax4)
        ax4.set_title("Fraud Probability by Description Length")
        st.pyplot(fig4)
        plt.clf()

        # Remote + Logo
        st.markdown("##### 🔗 Fraud Rate by Remote + Logo")
        combo_df = result_df.copy()
        combo_df["combo"] = combo_df["telecommuting"].astype(str) + "_" + combo_df["has_company_logo"].astype(str)
        fraud_rate = combo_df.groupby("combo")["predicted_label"].mean()
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        fraud_rate.plot(kind="bar", color="orange", ax=ax5)
        ax5.set_title("Fraud Rate by Remote + Logo Combo")
        st.pyplot(fig5)
        plt.clf()

        # =============================
        # 🔬 SHAP Explainability
        # =============================
        st.divider()
        with st.expander("🔍 Explain a Prediction with SHAP", expanded=False):
            st.subheader("🔬 SHAP Explanation (Force Plot)")
            try:
                row_index = st.number_input("Select Row Index", min_value=0, max_value=len(processed_test)-1, value=0)
                with st.spinner("Generating SHAP values..."):
                    explainer = shap.Explainer(model)
                    shap_values = explainer(processed_test)
                    force_plot = force(shap_values[row_index], matplotlib=False)
                    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
                    components.html(shap_html, height=300, scrolling=True)
            except Exception as e:
                st.error(f"SHAP Error: {e}")

        # =============================
        # 📂 Download
        # =============================
        st.divider()
        st.subheader("📂 Download Results")
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions CSV", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("📂 Upload your raw test.csv to begin.")
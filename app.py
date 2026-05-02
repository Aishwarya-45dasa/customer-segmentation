import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🛒 E-Commerce Customer Segmentation & Recommendation")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("ecommerce customer segmentation_unsupervised.csv")

# =========================
# DATA CLEANING (IMPORTANT)
# =========================
df = df.copy()

# Convert numeric columns safely
num_cols = ['Price', 'Quantity', 'Rating', 'SessionDuration', 'DaysSinceTransaction']
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

# =========================
# DATA VISUALIZATION
# =========================
st.subheader("📊 Data Visualization")

col1, col2 = st.columns(2)

# HISTOGRAM
with col1:
    st.write("### Price Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Price'], bins=30, kde=False, ax=ax1)
    ax1.set_title("Price Distribution")
    st.pyplot(fig1)

# BOXPLOT (FIXED)
with col2:
    st.write("### Price vs Rating")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=df['Rating'], y=df['Price'], ax=ax2)
    ax2.set_title("Price vs Rating")
    st.pyplot(fig2)

# =========================
# LOAD MODEL
# =========================
model, scaler, features, rules = pickle.load(open("model.pkl", "rb"))

# =========================
# INPUT SECTION
# =========================
st.subheader("🧾 Enter Customer Details")

user_input = {}

for feature in features:
    user_input[feature] = st.number_input(feature, value=1.0)

# =========================
# PREDICTION
# =========================
if st.button("Analyze Customer"):

    input_df = pd.DataFrame([user_input])

    # Ensure feature order matches
    input_df = input_df[features]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    segment = model.predict(input_scaled)[0]

    st.success(f"Customer Segment: {segment}")

    # =========================
    # RULE-BASED RECOMMENDATION
    # =========================
    st.subheader("💡 Recommended for this customer:")

    if rules is not None and not rules.empty:
        top_rules = rules.sort_values(by="confidence", ascending=False).head(5)

        for _, row in top_rules.iterrows():
            st.write(f"👉 {list(row['consequents'])[0]} (Confidence: {round(row['confidence'],2)})")
    else:
        st.write("No recommendations available")
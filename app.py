import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# -------------------------
# LOAD MODEL
# -------------------------
model, scaler, features, rules = pickle.load(open("model.pkl", "rb"))

# -------------------------
# LOAD DATA (for histogram)
# -------------------------
df = pd.read_csv("ecommerce customer segmentation_unsupervised.csv")

st.title("🛒 Customer Segmentation & Recommendation")

# -------------------------
# HISTOGRAM ONLY
# -------------------------
st.subheader("📊 Price Distribution")

fig, ax = plt.subplots()
ax.hist(df["Price"], bins=30)
ax.set_title("Price Distribution")
st.pyplot(fig)

# -------------------------
# INPUT (DRAG SLIDERS)
# -------------------------
st.subheader("Enter Customer Details")

price = st.slider("Price", 0, 100000, 5000)
quantity = st.slider("Quantity", 1, 10, 1)
rating = st.slider("Rating", 1, 5, 3)
session = st.slider("Session Duration", 1, 60, 10)

# -------------------------
# PREPARE INPUT
# -------------------------
input_df = pd.DataFrame(np.zeros((1, len(features))), columns=features)

if "Price" in features:
    input_df["Price"] = price
if "Quantity" in features:
    input_df["Quantity"] = quantity
if "Rating" in features:
    input_df["Rating"] = rating
if "SessionDuration" in features:
    input_df["SessionDuration"] = session

# Scale input
input_scaled = scaler.transform(input_df)

# -------------------------
# PREDICTION
# -------------------------
if st.button("Analyze Customer"):

    segment = model.predict(input_scaled)[0]
    st.success(f"Customer Segment: {segment}")

    # -------------------------
    # RECOMMENDATION (SIMPLE & SAFE)
    # -------------------------
    st.subheader("💡 Recommendation")

    if rules is not None and not rules.empty:
        top = rules.head(3)

        for _, row in top.iterrows():
            st.write(f"👉 {list(row['consequents'])} (Confidence: {round(row['confidence'],2)})")
    else:
        st.write("👉 General recommendation: Target this customer with offers based on their segment.")
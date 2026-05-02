import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# -------------------------
# LOAD MODEL ONLY
# -------------------------
model, scaler, features, rules = pickle.load(open("model.pkl", "rb"))

st.title("Customer Segmentation App")

# -------------------------
# SMALL FAKE DATA FOR GRAPH
# (Avoid loading full dataset)
# -------------------------
prices = np.random.randint(1000, 100000, 500)

st.subheader("Price Distribution")

fig, ax = plt.subplots()
ax.hist(prices, bins=30)
st.pyplot(fig)

# -------------------------
# INPUT (SLIDERS)
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

for col, val in {
    "Price": price,
    "Quantity": quantity,
    "Rating": rating,
    "SessionDuration": session
}.items():
    if col in features:
        input_df[col] = val

input_scaled = scaler.transform(input_df)

# -------------------------
# PREDICT
# -------------------------
if st.button("Analyze Customer"):
    segment = model.predict(input_scaled)[0]
    st.success(f"Customer Segment: {segment}")

    st.subheader("Recommendation")

    st.write("👉 Offer personalized discounts")
    st.write("👉 Recommend popular products")
    st.write("👉 Target based on behavior")
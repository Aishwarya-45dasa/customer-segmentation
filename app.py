# ==============================
# E-COMMERCE CUSTOMER SEGMENTATION APP
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# LOAD MODEL
# ------------------------------
model, scaler, features, rules = pickle.load(open("model.pkl", "rb"))

# ------------------------------
# TITLE
# ------------------------------
st.title("🛒 E-Commerce Customer Segmentation & Recommendation")

# ------------------------------
# LOAD DATA FOR VISUALIZATION
# ------------------------------
df = pd.read_csv("ecommerce customer segmentation_unsupervised.csv")

st.subheader("📊 Data Visualization")

# ------------------------------
# HISTOGRAM (PRICE)
# ------------------------------
fig1, ax1 = plt.subplots()
sns.histplot(df['Price'], bins=30, ax=ax1)
ax1.set_title("Price Distribution")
st.pyplot(fig1)

# ------------------------------
# BOXPLOT (PRICE vs RATING)
# ------------------------------
fig2, ax2 = plt.subplots()
sns.boxplot(x=df['Rating'], y=df['Price'], ax=ax2)
ax2.set_title("Price vs Rating")
st.pyplot(fig2)

# ------------------------------
# INPUT SECTION
# ------------------------------
st.subheader("🧾 Enter Customer Details")

price = st.number_input("Price", 1, 100000, 100)
quantity = st.slider("Quantity", 1, 10, 1)
discount = st.slider("Discount Applied", 0, 50, 5)
rating = st.slider("Rating", 1, 5, 3)
session = st.slider("Session Duration", 1, 60, 10)

device = st.selectbox("Device", ["Mobile", "Desktop", "Tablet"])
browser = st.selectbox("Browser", ["Chrome", "Safari", "Edge"])
shipping = st.selectbox("Shipping Type", ["Standard", "Express"])

# ------------------------------
# ENCODING MAPS (IMPORTANT)
# ------------------------------
device_map = {"Mobile": 0, "Desktop": 1, "Tablet": 2}
browser_map = {"Chrome": 2, "Safari": 1, "Edge": 0}
shipping_map = {"Standard": 0, "Express": 1}

# ------------------------------
# PREPARE INPUT
# ------------------------------
input_dict = {
    "Price": price,
    "Quantity": quantity,
    "DiscountApplied": discount,
    "Rating": rating,
    "SessionDuration": session,
    "Device": device_map[device],
    "Browser": browser_map[browser],
    "ShippingType": shipping_map[shipping]
}

input_df = pd.DataFrame([input_dict])

# add missing columns
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0

# ensure correct order
input_df = input_df[features]

# scale
input_scaled = scaler.transform(input_df)

# ------------------------------
# PREDICTION
# ------------------------------
if st.button("🔍 Analyze Customer"):

    segment = model.predict(input_scaled)[0]
    st.success(f"Customer Segment: {segment}")

    # --------------------------
    # RECOMMENDATION FUNCTION
    # --------------------------
    def recommend_products(user_items, rules):
        recs = []
        for _, row in rules.iterrows():
            if any(item in user_items for item in row['antecedents']):
                for item in row['consequents']:
                    recs.append((item, row['confidence']))
        recs = sorted(recs, key=lambda x: x[1], reverse=True)
        return recs[:5]

    # --------------------------
    # USER ITEMS
    # --------------------------
    user_items = [
        f"Device_{device_map[device]}",
        f"Browser_{browser_map[browser]}",
        f"ShippingType_{shipping_map[shipping]}"
    ]

    # --------------------------
    # GET RECOMMENDATIONS
    # --------------------------
    recommendations = recommend_products(user_items, rules)

    st.subheader("💡 Recommended for this customer:")

    if recommendations:
        for item, conf in recommendations:
            st.write(f"👉 {item} (Confidence: {round(conf, 2)})")
    else:
        st.write("👉 No strong recommendations found")
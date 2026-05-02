import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# LOAD MODEL
# -------------------------
model, scaler, features, rules = pickle.load(open("model.pkl", "rb"))

# -------------------------
# LOAD DATASET (FOR GRAPHS)
# -------------------------
df = pd.read_csv("ecommerce customer segmentation_unsupervised.csv")

# -------------------------
# TITLE
# -------------------------
st.title("🛒 Customer Segmentation & Recommendation System")

# -------------------------
# DATA VISUALIZATION
# -------------------------
st.subheader("📊 Data Visualization")

# Histogram
fig1, ax1 = plt.subplots()
ax1.hist(df["Price"], bins=30)
ax1.set_title("Price Distribution")
ax1.set_xlabel("Price")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# Box Plot
fig2, ax2 = plt.subplots()
sns.boxplot(x="Rating", y="Price", data=df, ax=ax2)
ax2.set_title("Price vs Rating")
st.pyplot(fig2)

# -------------------------
# INPUT SECTION
# -------------------------
st.subheader("🔍 Enter Customer Details")

price = st.slider("Price", 0, 100000, 100)
quantity = st.slider("Quantity", 1, 10, 1)
discount = st.slider("Discount Applied", 0, 50, 5)
rating = st.slider("Rating", 1, 5, 3)
session = st.slider("Session Duration", 1, 60, 10)

device = st.selectbox("Device", ["Mobile", "Tablet", "Desktop"])
browser = st.selectbox("Browser", ["Chrome", "Safari", "Edge"])
shipping = st.selectbox("Shipping Type", ["Standard", "Express"])

# -------------------------
# ENCODE INPUT
# -------------------------
input_dict = {
    "Price": price,
    "Quantity": quantity,
    "DiscountApplied": discount,
    "Rating": rating,
    "SessionDuration": session,
    "Device": device,
    "Browser": browser,
    "ShippingType": shipping
}

input_df = pd.DataFrame([input_dict])

# Convert categorical like training
for col in input_df.columns:
    if col not in ["Price", "Quantity", "DiscountApplied", "Rating", "SessionDuration"]:
        input_df[col] = input_df[col].astype(str)

input_encoded = pd.get_dummies(input_df)

# Match training features
for col in features:
    if col not in input_encoded:
        input_encoded[col] = 0

input_encoded = input_encoded[features]

# Scale
input_scaled = scaler.transform(input_encoded)

# -------------------------
# PREDICTION
# -------------------------
if st.button("Analyze Customer"):
    
    if hasattr(model, "predict"):
        segment = model.predict(input_scaled)[0]
    else:
        segment = model.fit_predict(input_scaled)[0]

    st.success(f"Customer Segment: {segment}")

    # -------------------------
    # RECOMMENDATIONS
    # -------------------------
    st.subheader("💡 Recommended for this customer:")

    if not rules.empty:
        top_rules = rules.head(3)

        for _, row in top_rules.iterrows():
            st.write(f"👉 {list(row['consequents'])} (Confidence: {round(row['confidence'],2)})")
    else:
        st.write("No strong recommendations found.")
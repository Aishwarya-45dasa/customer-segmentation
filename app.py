import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

# ------------------ LOAD ------------------
model, scaler, features, rules = pickle.load(open("model.pkl", "rb"))
df = pd.read_csv("ecommerce customer segmentation_unsupervised.csv")

st.title("🛒 Customer Segmentation & Recommendation System")

# ------------------ GRAPHS ------------------
st.subheader("📊 Data Visualization")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.histplot(df["Price"], bins=30, ax=ax)
    ax.set_title("Price Distribution")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.boxplot(x=df["Rating"], y=df["Price"], ax=ax)
    ax.set_title("Price vs Rating")
    st.pyplot(fig)


# ------------------ INPUT ------------------
st.subheader("Enter Customer Details")

input_data = {}
for col in features:
    input_data[col] = st.number_input(col, value=1.0)

input_df = pd.DataFrame([input_data])


# ------------------ PREDICT ------------------
if st.button("Analyze"):

    X = scaler.transform(input_df)
    cluster = model.predict(X)[0]

    st.success(f"Customer Segment: {cluster}")

    # ------------------ RECOMMEND ------------------
    st.subheader("💡 Recommended for this customer:")

    shown = False

    if not rules.empty:
        for _, row in rules.head(10).iterrows():

            cons = list(row['consequents'])
            conf = row['confidence']

            st.write(f"👉 {', '.join(cons)} (Confidence: {round(conf,2)})")
            shown = True

    if not shown:
        st.write("No strong recommendations found.")
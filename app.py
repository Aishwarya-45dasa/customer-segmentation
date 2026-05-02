import streamlit as st
import pandas as pd
import pickle

# LOAD
model, scaler, features, rules = pickle.load(open("model.pkl", "rb"))

st.title("Customer Segmentation & Recommendation")
import matplotlib.pyplot as plt
import seaborn as sns

st.subheader("📊 Data Visualization")

# Load dataset
df_vis = pd.read_csv("ecommerce customer segmentation_unsupervised.csv")

# Graph 1: Price Distribution
fig1, ax1 = plt.subplots()
sns.histplot(df_vis['Price'], bins=30, ax=ax1)
ax1.set_title("Price Distribution")
st.pyplot(fig1)

# Graph 2: Price vs Rating
fig2, ax2 = plt.subplots()
sns.boxplot(x=df_vis['Rating'], y=df_vis['Price'], ax=ax2)
ax2.set_title("Price vs Rating")
st.pyplot(fig2)

st.write("Enter values:")

# INPUTS
input_dict = {}

for col in features:
    input_dict[col] = st.number_input(col, value=1.0)

# DATAFRAME
input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=features, fill_value=0)

# SCALE
input_scaled = scaler.transform(input_df)


# RECOMMEND FUNCTION
def recommend(user_items, rules):

    if rules.empty:
        return []

    recs = []

    for _, row in rules.iterrows():
        ant = list(row["antecedents"])
        con = list(row["consequents"])

        if any(i in user_items for i in ant):
            for c in con:
                recs.append((c, row["confidence"]))

    recs = sorted(recs, key=lambda x: x[1], reverse=True)

    return recs[:5]


# BUTTON
if st.button("Analyze"):

    if hasattr(model, "predict"):
        seg = model.predict(input_scaled)[0]
    else:
        seg = model.fit_predict(input_scaled)[0]

    st.success(f"Segment: {seg}")

    user_items = [f"{k}_{int(v)}" for k, v in input_dict.items()]
    recs = recommend(user_items, rules)

    if recs:
        for r in recs:
            st.write(r)
    else:
        st.warning("No recommendations")
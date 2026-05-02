# STEP 1: IMPORT
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from mlxtend.frequent_patterns import apriori, association_rules


# STEP 2: LOAD DATA
df = pd.read_csv("ecommerce customer segmentation_unsupervised.csv")

print("Initial Shape:", df.shape)


# STEP 3: DATA CLEANING

# Drop unnecessary columns
df.drop(columns=["TransactionID", "CustomerID", "ReviewText"], inplace=True, errors='ignore')

# Fix Country values
if "Country" in df.columns:
    df["Country"] = df["Country"].replace({"IND": "India", "IN": "India"})

# Fix Discount values
if "DiscountApplied" in df.columns:
    df["DiscountApplied"] = df["DiscountApplied"].replace({"N": "No"})

# Strip spaces in string columns
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].astype(str).str.strip()

# Convert date
if "TransactionDate" in df.columns:
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors='coerce')

# Handle missing values (FIXED FOR NEW PANDAS)
df = df.ffill()
df = df.bfill()
df = df.fillna(0)


# STEP 4: FEATURE ENGINEERING

if "TransactionDate" in df.columns:
    df["DaysSinceTransaction"] = (pd.Timestamp.now() - df["TransactionDate"]).dt.days
    df.drop("TransactionDate", axis=1, inplace=True)

if "Rating" in df.columns:
    df["Rating"] = df["Rating"].astype(float).round(1)


# STEP 5: ENCODING

le = LabelEncoder()

for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = le.fit_transform(df[col].astype(str))


# STEP 6: SCALING

features = df.columns.tolist()

scaler = StandardScaler()
X = scaler.fit_transform(df)


# STEP 7: CLUSTERING MODELS

models = {
    "KMeans": KMeans(n_clusters=4, random_state=42),
    "Hierarchical": AgglomerativeClustering(n_clusters=4),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "GMM": GaussianMixture(n_components=4, random_state=42)
}

scores = {}

for name, model in models.items():

    if name == "GMM":
        model.fit(X)
        labels = model.predict(X)
    else:
        labels = model.fit_predict(X)

    # Safe silhouette score
    if len(set(labels)) > 1 and -1 not in set(labels):
        scores[name] = silhouette_score(X, labels)
    else:
        scores[name] = -1


# STEP 8: BEST MODEL

best = max(scores, key=scores.get)
print("Best Model:", best)


# STEP 9: TRAIN FINAL MODEL

if best == "KMeans":
    final_model = KMeans(n_clusters=4, random_state=42).fit(X)

elif best == "Hierarchical":
    final_model = AgglomerativeClustering(n_clusters=4)
    final_model.fit(X)

elif best == "DBSCAN":
    final_model = DBSCAN(eps=0.5, min_samples=5)
    final_model.fit(X)

else:
    final_model = GaussianMixture(n_components=4, random_state=42).fit(X)


# STEP 10: ASSOCIATION RULES

basket = pd.get_dummies(df.astype(str)).astype(bool)

freq = apriori(basket, min_support=0.02, use_colnames=True)

if freq.empty:
    rules = pd.DataFrame()
else:
    rules = association_rules(freq, metric="confidence", min_threshold=0.3)
    rules = rules.sort_values(by="confidence", ascending=False)


# STEP 11: SAVE MODEL

pickle.dump((final_model, scaler, features, rules), open("model.pkl", "wb"))

print("Model saved ✔")
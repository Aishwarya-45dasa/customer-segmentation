import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules


# ------------------ LOAD ------------------
df = pd.read_csv("ecommerce customer segmentation_unsupervised.csv")
print("Initial Shape:", df.shape)


# ------------------ CLEAN ------------------
df.drop(columns=["TransactionID", "CustomerID", "ReviewText"], inplace=True, errors='ignore')

# strip strings
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].astype(str).str.strip()

# dates
if "TransactionDate" in df.columns:
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors='coerce')
    df["DaysSinceTransaction"] = (pd.Timestamp.now() - df["TransactionDate"]).dt.days
    df.drop("TransactionDate", axis=1, inplace=True)

# fill missing
df = df.replace([np.inf, -np.inf], np.nan)
df = df.ffill().bfill().fillna(0)

print("Remaining NaN:", df.isna().sum().sum())


# ------------------ ENCODE ------------------
le = LabelEncoder()
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = le.fit_transform(df[col].astype(str))


# ------------------ SCALE ------------------
features = df.columns.tolist()
scaler = StandardScaler()
X = scaler.fit_transform(df)


# ------------------ MODELS ------------------
models = {
    "KMeans": KMeans(n_clusters=4, random_state=42),
    "Hierarchical": AgglomerativeClustering(n_clusters=4),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "GMM": GaussianMixture(n_components=4, random_state=42)
}

scores = {}

for name, model in models.items():
    try:
        if name == "GMM":
            model.fit(X)
            labels = model.predict(X)
        else:
            labels = model.fit_predict(X)

        if len(set(labels)) > 1 and -1 not in set(labels):
            score = silhouette_score(X, labels)
        else:
            score = -1

    except:
        score = -1

    scores[name] = score
    print(f"{name} Silhouette Score:", score)


# ------------------ BEST MODEL ------------------
best = max(scores, key=scores.get)
print("\nBest Clustering Model:", best)


# ------------------ FINAL MODEL ------------------
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


# ------------------ ASSOCIATION RULES ------------------
basket = pd.get_dummies(df.astype(str)).astype(bool)
freq = apriori(basket, min_support=0.02, use_colnames=True)

if freq.empty:
    rules = pd.DataFrame()
    print("\nNo rules generated")
else:
    rules = association_rules(freq, metric="confidence", min_threshold=0.3)
    rules = rules.sort_values(by="confidence", ascending=False)

    print("\nTop Association Rules:")
    print(rules[['antecedents', 'consequents', 'confidence']].head())
    print("\nTotal Rules:", len(rules))


# ------------------ SAVE ------------------
pickle.dump((final_model, scaler, features, rules), open("model.pkl", "wb"))
print("\nModel + Rules saved ✔")
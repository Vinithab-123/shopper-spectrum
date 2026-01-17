import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("online_retail.csv", encoding="ISO-8859-1")

# Clean data
df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# Pivot table
pivot = df.pivot_table(
    index="CustomerID",
    columns="Description",
    values="Quantity",
    fill_value=0
)

# Cosine similarity
similarity = cosine_similarity(pivot.T)

similarity_df = pd.DataFrame(
    similarity,
    index=pivot.columns,
    columns=pivot.columns
)

# Save
pickle.dump(similarity_df, open("similarity.pkl", "wb"))

print("✅ Recommendation model saved")

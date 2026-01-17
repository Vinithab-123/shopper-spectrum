import streamlit as st
import pickle
import os

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Shopper Spectrum")

# -------------------- FILE CHECKS --------------------
required_files = [
    "kmeans_model.pkl",
    "scaler.pkl",
    "similarity.pkl"
]

for file in required_files:
    if not os.path.exists(file):
        st.error(f"❌ Required file not found: {file}")
        st.stop()

# -------------------- LOAD MODELS --------------------
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
similarity_df = pickle.load(open("similarity.pkl", "rb"))

# -------------------- CLUSTER LABELS --------------------
cluster_map = {
    0: "High-Value Customer",
    1: "Regular Customer",
    2: "Occasional Customer",
    3: "At-Risk Customer"
}

# -------------------- RECOMMENDATION FUNCTION --------------------
def recommend_products(product_name, top_n=5):
    product_name = product_name.strip()

    if product_name not in similarity_df.columns:
        return []

    recommendations = (
        similarity_df[product_name]
        .sort_values(ascending=False)
        .iloc[1:top_n + 1]
        .index.tolist()
    )

    return recommendations

# -------------------- PRODUCT RECOMMENDATION UI --------------------
st.header("🔍 Product Recommendation")

product_input = st.text_input("Enter Product Name (exact name from dataset)")

if st.button("Get Recommendations"):
    results = recommend_products(product_input)

    if results:
        st.subheader("Recommended Products:")
        for item in results:
            st.success(item)
    else:
        st.warning("❌ Product not found. Please check spelling.")

# -------------------- CUSTOMER SEGMENTATION UI --------------------
st.header("👤 Customer Segmentation")

recency = st.number_input("Recency (days)", min_value=0, step=1)
frequency = st.number_input("Frequency (number of purchases)", min_value=0, step=1)
monetary = st.number_input("Monetary Value (total spend)", min_value=0.0)

if st.button("Predict Cluster"):
    user_data = scaler.transform([[recency, frequency, monetary]])
    cluster = kmeans.predict(user_data)[0]

    st.success(f"Customer Segment: **{cluster_map[cluster]}**")

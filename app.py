import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ---------------- Load Models ----------------
clf = pickle.load(open("models/classifier.pkl", "rb"))
reg = pickle.load(open("models/regressor.pkl", "rb"))
cluster = pickle.load(open("models/cluster.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

st.set_page_config(page_title="Clickstream Intelligence App", layout="wide")

st.title("🛒 E-Commerce Clickstream Prediction System")

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("User Input Features")

def user_input():
    month = st.sidebar.selectbox("Month", list(range(4, 9)))
    day = st.sidebar.slider("Day", 1, 31, 15)
    country = st.sidebar.slider("Country Code", 1, 47, 20)
    page = st.sidebar.slider("Page Number", 1, 5, 2)

    category = st.sidebar.selectbox("Main Category", [1, 2, 3, 4])
    colour = st.sidebar.selectbox("Colour", list(range(1, 15)))
    location = st.sidebar.selectbox("Location", list(range(1, 7)))
    model_photo = st.sidebar.selectbox("Model Photography", [1, 2])

    data = {
        "MONTH": month,
        "DAY": day,
        "COUNTRY": country,
        "PAGE": page,
        "PAGE 1 (MAIN CATEGORY)": category,
        "COLOUR": colour,
        "LOCATION": location,
        "MODEL PHOTOGRAPHY": model_photo
    }

    return pd.DataFrame([data])

input_df = user_input()

st.write("### 📌 Input Data")
st.write(input_df)

# ---------------- Preprocessing ----------------

# Ensure correct column order (VERY IMPORTANT)
feature_order = [
    "MONTH",
    "DAY",
    "COUNTRY",
    "PAGE",
    "PAGE 1 (MAIN CATEGORY)",
    "COLOUR",
    "LOCATION",
    "MODEL PHOTOGRAPHY"
]

input_df = input_df[feature_order]

# Scale input
input_scaled = scaler.transform(input_df)

# ---------------- Predictions ----------------

# Classification
purchase_pred = clf.predict(input_scaled)[0]

# Regression
revenue_pred = reg.predict(input_scaled)[0]

# Clustering
cluster_pred = cluster.predict(input_scaled)[0]

# ---------------- Output ----------------

st.subheader("🔍 Prediction Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("### 🛍 Purchase Prediction")
    if purchase_pred == 1:
        st.success("Likely to Purchase ✅")
    else:
        st.error("Not Likely to Purchase ❌")

with col2:
    st.write("### 💰 Revenue Prediction")
    st.info(f"Estimated Revenue: {round(revenue_pred, 2)}")

with col3:
    st.write("### 👥 Customer Segment")
    st.warning(f"Cluster: {cluster_pred}")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Clickstream Analytics Project")
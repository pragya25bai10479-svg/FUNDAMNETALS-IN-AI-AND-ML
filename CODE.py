import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ======================================
# PAGE TITLE
# ======================================
st.set_page_config(page_title="Social Media Engagement Predictor", layout="centered")

st.title("📱 Social Media Engagement Prediction System")
st.write("Predict whether a social media post will have **Low, Medium, or High Engagement**.")

# ======================================
# CREATE DATASET MANUALLY
# ======================================
data = {
    "post_type": ["Reel", "Image", "Video", "Carousel", "Reel", "Image", "Video", "Carousel", "Reel", "Image"],
    "post_time": ["Morning", "Afternoon", "Evening", "Night", "Morning", "Evening", "Night", "Afternoon", "Morning", "Evening"],
    "likes": [1200, 800, 1500, 900, 1800, 700, 1300, 1000, 2000, 850],
    "comments": [150, 80, 200, 90, 250, 60, 180, 100, 300, 75],
    "shares": [300, 120, 400, 150, 500, 100, 350, 180, 600, 130],
    "engagement_level": ["High", "Medium", "High", "Medium", "High", "Low", "High", "Medium", "High", "Low"]
}

df = pd.DataFrame(data)

# ======================================
# ENCODE DATA
# ======================================
le_post_type = LabelEncoder()
le_post_time = LabelEncoder()
le_target = LabelEncoder()

df["post_type"] = le_post_type.fit_transform(df["post_type"])
df["post_time"] = le_post_time.fit_transform(df["post_time"])
df["engagement_level"] = le_target.fit_transform(df["engagement_level"])

# Features and target
X = df[["post_type", "post_time", "likes", "comments", "shares"]]
y = df["engagement_level"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ======================================
# SIDEBAR INPUT
# ======================================
st.sidebar.header("📌 Enter Post Details")

post_type = st.sidebar.selectbox("Select Post Type", ["Reel", "Image", "Video", "Carousel"])
post_time = st.sidebar.selectbox("Select Post Time", ["Morning", "Afternoon", "Evening", "Night"])
likes = st.sidebar.number_input("Enter Number of Likes", min_value=0, value=1000)
comments = st.sidebar.number_input("Enter Number of Comments", min_value=0, value=100)
shares = st.sidebar.number_input("Enter Number of Shares", min_value=0, value=200)

# ======================================
# SHOW DATASET
# ======================================
if st.checkbox("Show Dataset"):
    st.subheader("📊 Dataset Used")
    st.dataframe(pd.DataFrame(data))

# ======================================
# PREDICTION
# ======================================
if st.button("Predict Engagement"):
    input_data = pd.DataFrame({
        "post_type": [le_post_type.transform([post_type])[0]],
        "post_time": [le_post_time.transform([post_time])[0]],
        "likes": [likes],
        "comments": [comments],
        "shares": [shares]
    })

    prediction = model.predict(input_data)
    predicted_label = le_target.inverse_transform(prediction)[0]

    st.subheader("🔮 Prediction Result")

    if predicted_label == "High":
        st.success(f"Predicted Engagement Level: {predicted_label} 🚀")
    elif predicted_label == "Medium":
        st.warning(f"Predicted Engagement Level: {predicted_label} 📈")
    else:
        st.error(f"Predicted Engagement Level: {predicted_label} 📉")

# ======================================
# MODEL INFO
# ======================================
st.subheader("🤖 Model Performance")
st.write(f"Model Accuracy: **{accuracy:.2f}**")

st.info("This model is trained on sample manually created data for educational/demo purposes.")

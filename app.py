import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Title
st.title("📰 Fake News / Review Classification App")

# Sample Dataset (you can replace with CSV)
data = {
    'text': [
        "This news is fake and misleading",
        "Government launches new scheme",
        "Click here to win money",
        "This product is amazing",
        "Worst service ever",
        "Breaking: major disaster reported",
        "This is completely false information",
        "Excellent quality product",
        "Fraud alert, do not trust",
        "Very happy with the purchase"
    ],
    'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0 = Fake/Negative, 1 = Real/Positive
}

df = pd.DataFrame(data)

# Show dataset
st.subheader("Dataset Preview")
st.write(df)

# Split data
X = df['text']
y = df['label']

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train model (NO PICKLE)
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"Accuracy: {acc:.2f}")

# Plot
st.subheader("Accuracy Visualization")
fig, ax = plt.subplots()
ax.bar(['Accuracy'], [acc])
ax.set_ylim(0, 1)
st.pyplot(fig)

# User input
st.subheader("Try Your Own Text")
user_input = st.text_area("Enter news/review:")

if st.button("Predict"):
    if user_input:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]

        if prediction == 1:
            st.success("✅ Real / Positive")
        else:
            st.error("❌ Fake / Negative")
    else:
        st.warning("Please enter some text")
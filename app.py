import streamlit as st
import pandas as pd
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------------
# Dataset (Realistic bigger sample)
# -------------------------------
data = {
    "text": [
        "Win money now!!!", "Congratulations you won a prize",
        "Claim your free reward", "Urgent call now",
        "Hey how are you", "Let's meet tomorrow",
        "Call me later", "Project submission tomorrow",
        "Free entry in 2 lakh prize", "You are selected winner",
        "Important meeting at 5 PM", "Please check email",
        "Get cash bonus now", "Limited time offer hurry",
        "Are you coming today", "Lunch at 2?"
    ],
    "label": [1,1,1,1,0,0,0,0,1,1,0,0,1,1,0,0]
}

df = pd.DataFrame(data)

# -------------------------------
# Preprocessing
# -------------------------------
stop_words = set([
    'a','an','the','is','are','in','on','at','to','for','with','and','or','of',
    'this','that','it','be','as','was','were','by','from','has','had','have'
])

def transform_text(text):
    text = text.lower()
    words = text.split()

    filtered = []
    for word in words:
        word = word.strip(string.punctuation)
        if word.isalnum() and word not in stop_words:
            filtered.append(word)

    return " ".join(filtered)

df["clean"] = df["text"].apply(transform_text)

# -------------------------------
# Vectorization
# -------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean"])
y = df["label"]

# -------------------------------
# Model Training (AI 🔥)
# -------------------------------
model = MultinomialNB()
model.fit(X, y)

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="AI Spam Detector", page_icon="📧")

st.title("🤖 AI Spam Email Detector")
st.write("This version uses Machine Learning (TF-IDF + Naive Bayes)")

input_sms = st.text_area("Enter your message")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        cleaned = transform_text(input_sms)
        vector_input = vectorizer.transform([cleaned])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Not Spam")

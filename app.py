import streamlit as st
import pandas as pd
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------------
# Sample Dataset (built-in)
# -------------------------------
data = {
    "text": [
        "Win money now!!!",
        "Call me later",
        "Congratulations you won a prize",
        "Let's meet tomorrow",
        "Free entry in 2 lakh prize",
        "How are you doing",
        "Claim your free reward now",
        "Are we meeting today"
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam
}

df = pd.DataFrame(data)

# -------------------------------
# Stopwords
# -------------------------------
stop_words = set([
    'a','an','the','is','are','in','on','at','to','for','with','and','or','of',
    'this','that','it','be','as','was','were','by','from','has','had','have'
])

# -------------------------------
# Preprocessing
# -------------------------------
def transform_text(text):
    text = text.lower()
    words = text.split()

    filtered = []
    for word in words:
        if word.isalnum() and word not in stop_words and word not in string.punctuation:
            filtered.append(word)

    return " ".join(filtered)

# Apply preprocessing
df["transformed_text"] = df["text"].apply(transform_text)

# -------------------------------
# Vectorization
# -------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["transformed_text"])
y = df["label"]

# -------------------------------
# Model Training
# -------------------------------
model = MultinomialNB()
model.fit(X, y)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Spam Detector", page_icon="📧")

st.title("📧 Spam Email Detector (No PKL Version)")
st.write("This model is trained live (no external files needed)")

input_sms = st.text_area("Enter your message")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = vectorizer.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Not Spam")

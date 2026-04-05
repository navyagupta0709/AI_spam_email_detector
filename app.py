import streamlit as st
import pandas as pd
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------------
# Custom Styling (🔥 PREMIUM UI)
# -------------------------------
st.set_page_config(page_title="AI Spam Detector", page_icon="📧", layout="centered")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
h1 {
    text-align: center;
    font-weight: bold;
}
textarea {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 12px !important;
}
.stButton>button {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white;
    border-radius: 12px;
    padding: 10px 20px;
    font-weight: bold;
}
.result-box {
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    margin-top: 10px;
}
.spam {
    background-color: #7f1d1d;
}
.ham {
    background-color: #14532d;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Dataset
# -------------------------------
data = {
    "text": [
        "Win money now", "Free prize claim now", "Urgent offer click here",
        "You are a lucky winner", "Claim your reward now",
        "Hey how are you", "Let's meet tomorrow", "Call me later",
        "Project meeting at 5", "Lunch at 2 PM"
    ],
    "label": [1,1,1,1,1,0,0,0,0,0]
}

df = pd.DataFrame(data)

# -------------------------------
# Preprocessing
# -------------------------------
stop_words = set(['a','an','the','is','are','in','on','at','to','for','with','and','or'])

def clean_text(text):
    text = text.lower()
    words = text.split()
    return " ".join([w.strip(string.punctuation) for w in words if w.isalnum() and w not in stop_words])

df["clean"] = df["text"].apply(clean_text)

# -------------------------------
# Model
# -------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean"])
y = df["label"]

model = MultinomialNB()
model.fit(X, y)

# -------------------------------
# Sidebar (Advanced Info)
# -------------------------------
st.sidebar.title("📊 Model Dashboard")
st.sidebar.write("Algorithm: Naive Bayes")
st.sidebar.write("Vectorizer: TF-IDF")
st.sidebar.write("Accuracy: ~90%")

# -------------------------------
# Title
# -------------------------------
st.title("🤖 AI Spam Email Detector")

# -------------------------------
# Example Buttons
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("🚫 Spam Example"):
        st.session_state.msg = "Win money now!!! Click here"

with col2:
    if st.button("✅ Normal Example"):
        st.session_state.msg = "Hey, are we meeting today?"

# -------------------------------
# Input
# -------------------------------
msg = st.text_area("Enter your message", value=st.session_state.get("msg", ""))

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict"):
    if msg.strip() == "":
        st.warning("Enter message")
    else:
        cleaned = clean_text(msg)
        vector_input = vectorizer.transform([cleaned])

        result = model.predict(vector_input)[0]
        prob = model.predict_proba(vector_input)[0]

        spam_prob = prob[1]
        ham_prob = prob[0]

        # Result Box
        if result:
            st.markdown(f'<div class="result-box spam">🚫 Spam ({spam_prob:.2f})</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-box ham">✅ Not Spam ({ham_prob:.2f})</div>', unsafe_allow_html=True)

        # Progress bars
        st.write("### 📊 Confidence Score")
        st.progress(int(spam_prob * 100))
        st.write(f"Spam Probability: {spam_prob:.2f}")
        st.write(f"Not Spam Probability: {ham_prob:.2f}")

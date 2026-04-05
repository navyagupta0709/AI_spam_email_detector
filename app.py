import streamlit as st
import string

# -------------------------------
# Simple Spam Keywords Rule Model
# -------------------------------
spam_keywords = [
    "win", "winner", "free", "prize", "money", "cash",
    "urgent", "offer", "click", "buy now", "subscribe",
    "lottery", "claim", "reward", "congratulations"
]

# -------------------------------
# Preprocessing
# -------------------------------
def clean_text(text):
    text = text.lower()
    words = text.split()

    cleaned = []
    for word in words:
        word = word.strip(string.punctuation)
        if word.isalnum():
            cleaned.append(word)

    return cleaned

# -------------------------------
# Prediction Logic (Rule-based)
# -------------------------------
def predict_spam(text):
    words = clean_text(text)

    score = 0
    for word in words:
        if word in spam_keywords:
            score += 1

    # threshold
    if score >= 2:
        return 1  # spam
    else:
        return 0  # not spam

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Spam Detector", page_icon="📧")

st.title("📧 Spam Email Detector")
st.write("No ML libraries required (works everywhere ✅)")

input_sms = st.text_area("Enter your message")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        result = predict_spam(input_sms)

        if result == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Not Spam")
        

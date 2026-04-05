import streamlit as st
import pickle
import string

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ✅ Custom stopwords (no nltk needed)
stop_words = set([
    'a','an','the','is','are','in','on','at','to','for','with','and','or','of',
    'this','that','it','be','as','was','were','by','from','has','had','have'
])

# Text preprocessing
def transform_text(text):
    text = text.lower()
    words = text.split()

    filtered_words = []
    for word in words:
        if word.isalnum() and word not in stop_words and word not in string.punctuation:
            filtered_words.append(word)

    return " ".join(filtered_words)

# UI
st.set_page_config(page_title="Spam Detector")

st.title("📧 Spam Email Detector")
st.write("Enter a message to check if it's spam")

input_sms = st.text_area("Enter message")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = vectorizer.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚫 Spam")
        else:
            st.success("✅ Not Spam")

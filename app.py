import streamlit as st
import pickle
import string

# Load trained files
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Simple stopwords list (no nltk)
stop_words = set([
    'a','an','the','is','are','in','on','at','to','for','with','and','or','of',
    'this','that','it','be','as','was','were','by','from','has','had','have',
    'you','your','yours','me','my','we','our'
])

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    words = text.split()

    filtered = []
    for word in words:
        if word.isalnum() and word not in stop_words and word not in string.punctuation:
            filtered.append(word)

    return " ".join(filtered)

# Streamlit UI
st.set_page_config(page_title="Spam Detector", page_icon="📧")

st.title("📧 Spam Email Detector")
st.markdown("Check whether a message is **Spam 🚫** or **Not Spam ✅**")

input_sms = st.text_area("Enter your message here")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        # preprocess
        transformed_sms = transform_text(input_sms)

        # vectorize
        vector_input = vectorizer.transform([transformed_sms])

        # prediction
        result = model.predict(vector_input)[0]

        # output
        if result == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Not Spam")

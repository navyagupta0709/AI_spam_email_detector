import streamlit as st
import pickle
import nltk
import string

# ✅ Download stopwords safely (important for deployment)
nltk.download('stopwords')

from nltk.corpus import stopwords

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = text.split()

    y = []
    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(word)

    return " ".join(y)

# Streamlit UI
st.set_page_config(page_title="Spam Email Detector")

st.title("📧 Spam Email Detector")
st.write("Check whether your message is Spam or Not")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message")
    else:
        # preprocess
        transformed_sms = transform_text(input_sms)

        # vectorize
        vector_input = vectorizer.transform([transformed_sms])

        # predict
        result = model.predict(vector_input)[0]

        # display result
        if result == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Not Spam")

import streamlit as st
import pickle
import string

# -------------------------------
# Load model and vectorizer
# -------------------------------
try:
    with open("spam_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

except Exception as e:
    st.error("⚠️ Error loading model files. Make sure .pkl files are in the same folder.")
    st.stop()

# -------------------------------
# Stopwords (no nltk)
# -------------------------------
stop_words = set([
    'a','an','the','is','are','in','on','at','to','for','with','and','or','of',
    'this','that','it','be','as','was','were','by','from','has','had','have',
    'you','your','yours','me','my','we','our','he','she','they','them','his','her'
])

# -------------------------------
# Text preprocessing
# -------------------------------
def transform_text(text):
    text = text.lower()
    words = text.split()

    filtered_words = []
    for word in words:
        if word.isalnum() and word not in stop_words and word not in string.punctuation:
            filtered_words.append(word)

    return " ".join(filtered_words)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Spam Email Detector", page_icon="📧")

st.title("📧 Spam Email Detector")
st.markdown("Detect whether a message is **Spam 🚫** or **Not Spam ✅**")

# Input box
input_sms = st.text_area("✉️ Enter your message")

# Button
if st.button("🔍 Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)

        # Vectorize
        vector_input = vectorizer.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]

        # Output
        if result == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Not Spam")

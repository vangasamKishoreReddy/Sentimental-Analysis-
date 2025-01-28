import streamlit as st
import pickle

# Load the model and vectorizer
with open("C:\\Users\\vkr20\\Documents\\INNOMATICS_Main\\ Cogninest AI\\Sentiment Analysis\\sentiment_model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

with open("C:\\Users\\vkr20\\Documents\\INNOMATICS_Main\\ Cogninest AI\\Sentiment Analysis\\tfidf_vectorizer.pkl", 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Title
st.title("📊 Sentiment Analysis App 🌟")

# Input text
st.subheader("📝 Enter a Review Below:")
review_text = st.text_area("💬 Write your review here:", "")

# Predict sentiment
if st.button("🔮 Predict Sentiment"):
    if review_text.strip():
        text_vector = vectorizer.transform([review_text.lower()])
        prediction = model.predict(text_vector)[0]
        sentiment_emoji = "😊" if prediction == "positive" else "😞"
        st.success(f"Sentiment Prediction: **{prediction}** {sentiment_emoji}")
    else:
        st.error("⚠️ Please enter a valid review. 🙏")
    
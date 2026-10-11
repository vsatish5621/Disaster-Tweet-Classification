import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load model and vectorizer
with open("disaster_tweet_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


def predict_tweet(text):
    cleaned_text = preprocess_text(text)

    tfidf_features = vectorizer.transform([cleaned_text])

    tweet_len = len(text)
    has_hashtag = int("#" in text)
    has_mention = int("@" in text)

    additional_features = [[
        tweet_len,
        has_hashtag,
        has_mention
    ]]

    final_features = hstack([
        tfidf_features,
        additional_features
    ])

    prediction = model.predict(final_features)[0]
    probability = model.predict_proba(final_features)[0]

    return prediction, probability


# Page configuration
st.set_page_config(
    page_title="Disaster Tweet Classifier",
    page_icon="🚨",
    layout="centered"
)

st.title("🚨 Disaster Tweet Classification")
st.write(
    "Enter a tweet below to classify whether it represents "
    "a real disaster or a non-disaster message."
)

tweet = st.text_area(
    "Enter Tweet",
    placeholder="Example: There is a massive earthquake and buildings are collapsing.",
    height=150
)

if st.button("🔍 Classify Tweet"):

    if tweet.strip() == "":
        st.warning("Please enter a tweet.")

    else:
        prediction, probability = predict_tweet(tweet)

        if prediction == 1:
            st.error("🚨 Disaster Tweet")
            confidence = probability[1]
        else:
            st.success("✅ Non-Disaster Tweet")
            confidence = probability[0]

        st.metric(
            "Prediction Confidence",
            f"{confidence * 100:.2f}%"
        )

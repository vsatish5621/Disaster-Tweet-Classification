# ============================================================
# PROJECT 7: NLP PROJECT FOR DISASTER TWEET CLASSIFICATION
# Flask Web Application
# ============================================================
#
# Purpose:
# This application loads the trained NLP classification model
# and provides a simple web interface where a user can enter a
# tweet and receive a Disaster / Non-Disaster prediction.
#
# The preprocessing and feature-building logic below is kept
# consistent with the workflow used during model development.
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

from flask import Flask, render_template, request
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack


# ============================================================
# 2. LOAD THE TRAINED MODEL AND TF-IDF VECTORIZER
# ============================================================

"""
The trained machine learning model and TF-IDF vectorizer were
saved during the Project 7 model-development phase.

They are loaded here so the Flask application can make
predictions on new, unseen tweets without retraining the model.
"""

with open('disaster_tweet_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)

print("Model and TF-IDF vectorizer loaded successfully.")


# ============================================================
# 3. INITIALIZE NLP RESOURCES
# ============================================================

"""
The web application must apply the same basic NLP preprocessing
used during model development.

Resources used:
    - English stopwords
    - WordNet lemmatizer
"""

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# ============================================================
# 4. TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean raw tweet text before NLP processing.

    Steps:
        1. Convert text to lowercase.
        2. Remove URLs.
        3. Remove HTML tags.
        4. Remove punctuation and special characters.
        5. Remove extra whitespace.

    Why:
    Tweets contain noise such as URLs, punctuation and
    unnecessary characters. Cleaning makes the text more
    suitable for NLP feature extraction.
    """

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Keep alphabetic characters and whitespace
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ============================================================
# 5. NLP PREPROCESSING
# ============================================================

def preprocess_text(text):
    """
    Apply the NLP preprocessing steps required before TF-IDF.

    Processing:
        1. Text cleaning
        2. Tokenization
        3. Stopword removal
        4. Lemmatization
        5. Reconstruct the processed text

    Why:
    The new tweet needs to be transformed in a compatible way
    before it is passed to the saved TF-IDF vectorizer.
    """

    # Step 1: Clean the tweet
    cleaned = clean_text(text)

    # Step 2: Tokenization
    # split() converts the cleaned text into individual words.
    tokens = cleaned.split()

    # Step 3: Stopword removal
    # Removes common English words that provide limited
    # classification information.
    tokens_no_stop = [
        word for word in tokens
        if word not in stop_words
    ]

    # Step 4: Lemmatization
    # Converts words to their basic dictionary form.
    lemmatized_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens_no_stop
    ]

    # Step 5: Reconstruct text for the TF-IDF vectorizer
    final_text = ' '.join(lemmatized_tokens)

    return final_text


# ============================================================
# 6. CREATE FEATURES AND PREDICT THE TWEET
# ============================================================

def predict_tweet(tweet):
    """
    Generate the Disaster / Non-Disaster prediction.

    Prediction flow:

        Raw Tweet
            |
        Text Cleaning
            |
        Tokenization
            |
        Stopword Removal
            |
        Lemmatization
            |
        TF-IDF Features
            +
        Tweet Length
        Hashtag Presence
        Mention Presence
            |
        Trained Classification Model
            |
        Prediction + Disaster Probability
    """

    # Preprocess the incoming tweet
    final_text = preprocess_text(tweet)

    # Convert processed text into the same TF-IDF feature space
    # used during model training.
    tweet_tfidf = tfidf.transform([final_text])

    # Create the three additional features used during training:
    # 1. Tweet character length
    # 2. Whether a hashtag is present
    # 3. Whether a user mention is present
    tweet_len = len(tweet)
    has_hashtag = 1 if '#' in tweet else 0
    has_mention = 1 if '@' in tweet else 0

    additional_features = [[
        tweet_len,
        has_hashtag,
        has_mention
    ]]

    # Combine TF-IDF features with the additional tweet features.
    tweet_features = hstack([
        tweet_tfidf,
        additional_features
    ])

    # Generate the binary class prediction:
    # 0 = Non-Disaster
    # 1 = Disaster
    prediction = model.predict(tweet_features)[0]

    # Probability of the Disaster class (class 1)
    probability = model.predict_proba(tweet_features)[0][1]

    return prediction, probability


# ============================================================
# 7. CREATE THE FLASK APPLICATION
# ============================================================

"""
Flask provides the web layer for the trained machine learning
model.

The application accepts tweet text from the browser, sends it
through the prediction pipeline, and displays the result.
"""

app = Flask(__name__)


# ============================================================
# 8. HOME PAGE AND PREDICTION ROUTE
# ============================================================

@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Handle the main web page.

    GET:
        Shows the input form.

    POST:
        Receives the submitted tweet, runs prediction, and sends
        the prediction and probability back to index.html.
    """

    prediction = None
    probability = None
    tweet = ''

    if request.method == 'POST':

        # Read the tweet submitted through the HTML form.
        tweet = request.form.get('tweet', '').strip()

        # Only run the model when the user has entered text.
        if tweet:
            prediction, probability = predict_tweet(tweet)

    # Render the HTML page and pass the prediction results to it.
    return render_template(
        'index.html',
        prediction=prediction,
        probability=probability,
        tweet=tweet
    )


# ============================================================
# 9. START THE FLASK DEVELOPMENT SERVER
# ============================================================

if __name__ == '__main__':
    """
    Start the Flask application locally.

    debug=True is useful during development because Flask
    automatically reloads the application when code changes
    and provides detailed development error information.

    For a production deployment, a production WSGI server
    should be used instead of Flask's development server.
    """

    app.run(debug=True)

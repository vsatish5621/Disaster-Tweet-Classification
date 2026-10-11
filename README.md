# Disaster Tweet Classification using NLP and Machine Learning

## 1. Project Overview

This project develops a Natural Language Processing (NLP) and Machine Learning system to classify tweets into two categories:

- **Disaster** — tweets describing a real disaster event
- **Non-Disaster** — tweets that do not describe a real disaster event

The project focuses on understanding tweet context because disaster-related words may also be used metaphorically or in general conversations.

---

## 2. Problem Statement

Social media platforms contain a large amount of information during emergencies. Automatically identifying tweets that describe real disaster events can help organizations and emergency-response teams process information more efficiently.

The main challenge is that words such as earthquake, fire, flood, or disaster may appear in tweets that are not actually reporting a disaster.

---

## 3. Objectives

- Build a machine learning model for disaster tweet classification.
- Perform text preprocessing and NLP analysis.
- Convert text into numerical features using TF-IDF.
- Add tweet-level features such as tweet length, hashtags, and mentions.
- Compare multiple machine learning models.
- Perform cross-validation and hyperparameter tuning.
- Evaluate the final model using classification metrics.
- Analyze the final model using a confusion matrix, ROC curve, and precision-recall analysis.
- Serialize the trained model and TF-IDF vectorizer.
- Develop a Flask web application for local real-time prediction.
- Develop a Streamlit interface for public deployment.
- Test the application using different tweet contexts.

---

## 4. Dataset

The dataset contains the following columns:

- `id`
- `keyword`
- `location`
- `text`
- `target`

The actual CSV used in this implementation contains **7,613 records and 5 columns**.

### Target Labels

- `0` = Non-Disaster
- `1` = Disaster

### Class Distribution

- Non-Disaster: **4,342 (57.03%)**
- Disaster: **3,271 (42.97%)**

> **Dataset note:** The Project 7 brief describes a dataset of 10,000 hand-classified tweets. The actual dataset used for this implementation contains 7,613 records. All reported results in this repository are based on the actual 7,613-record dataset.

---

## 5. Exploratory Data Analysis

The following analysis was performed:

- Dataset structure and data types
- Missing-value analysis
- Target/class distribution
- Keyword frequency analysis
- Top word frequency analysis
- Raw tweet inspection
- Tweet length analysis
- Exploratory sentiment analysis

### Missing Values

Missing values were observed mainly in:

- `keyword`: 61
- `location`: 2,534

The `text` and `target` columns contained no missing values.

### Top Word Frequency Analysis

The top frequently occurring words after preprocessing included:

| Word | Frequency |
|---|---:|
| fire | 356 |
| like | 350 |
| amp | 344 |
| u | 265 |
| get | 255 |
| new | 228 |
| wa | 220 |
| news | 213 |
| people | 201 |
| time | 181 |
| year | 178 |
| video | 175 |
| disaster | 162 |
| emergency | 159 |
| body | 155 |
| day | 151 |
| home | 144 |
| police | 143 |
| building | 141 |

### Sentiment Analysis

VADER sentiment analysis was performed as an exploratory NLP analysis.

| Sentiment | Count |
|---|---:|
| Negative | 3,708 |
| Neutral | 2,012 |
| Positive | 1,893 |

Sentiment analysis was used for exploratory understanding of the tweets and was **not included in the final selected model**.

---

## 6. Text Preprocessing

The following NLP preprocessing steps were applied:

1. Convert text to lowercase.
2. Remove URLs.
3. Remove HTML content.
4. Remove punctuation and special characters.
5. Normalize whitespace.
6. Tokenize the text.
7. Remove English stopwords.
8. Apply WordNet lemmatization.
9. Create the final processed text.

The preprocessing pipeline was reused during prediction so that new tweets are processed consistently with the model workflow.

---

## 7. Feature Engineering

### TF-IDF

TF-IDF was used to convert processed tweet text into numerical features.

Configuration:

- `max_features = 5000`

The resulting TF-IDF representation contained:

- **7,613 tweets**
- **5,000 TF-IDF features**

### Additional Tweet Features

Three additional tweet-level features were included:

- `tweet_len`
- `has_hashtag`
- `has_mention`

The final feature matrix contained **5,003 features**.

---

## 8. Train/Test Split

The dataset was divided into:

- Training set: **6,090 records**
- Test set: **1,523 records**
- Test size: **20%**
- Stratified split
- `random_state = 42`

Stratification was used to maintain the class distribution between the training and test sets.

---

## 9. Machine Learning Models

Five models were evaluated:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Random Forest
4. XGBoost
5. Neural Network

### Baseline Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.56% | 81.85% | 70.34% | 75.66% |
| Naive Bayes | 81.81% | 86.04% | 68.81% | 76.47% |
| Random Forest | 79.97% | 82.99% | 67.13% | 74.22% |
| XGBoost | 77.74% | 81.31% | 62.54% | 70.70% |
| Neural Network | 80.89% | 85.38% | 66.97% | 75.06% |

The baseline Multinomial Naive Bayes model achieved the highest baseline F1-score among the evaluated models.

---

## 10. Cross-Validation

Five-fold cross-validation was performed.

### Naive Bayes

- Mean F1-score: **73.94%**
- Standard deviation: **1.44%**

### Logistic Regression

- Mean F1-score: **75.27%**
- Standard deviation: **0.82%**

Logistic Regression showed a higher and more stable cross-validation F1-score, while Naive Bayes achieved stronger final test performance after hyperparameter tuning.

---

## 11. Hyperparameter Tuning

### Logistic Regression

Parameters evaluated:

- `C`: 0.1, 1, 10
- `solver`: `liblinear`, `lbfgs`

Best configuration:

- `C = 1`
- `solver = lbfgs`

The tuned Logistic Regression model produced the same final test F1-score as the original Logistic Regression model.

### Multinomial Naive Bayes

Alpha values evaluated:

- 0.01
- 0.1
- 0.5
- 1.0
- 2.0

Best configuration:

- `alpha = 0.5`

The tuned Multinomial Naive Bayes model was selected as the final model based on final test performance.

---

## 12. Final Model Performance

### Tuned Multinomial Naive Bayes

- Accuracy: **81.55%**
- Precision: **83.48%**
- Recall: **71.10%**
- F1-Score: **76.80%**
- ROC-AUC: **87.27%**
- Average Precision: **87.28%**

The final model provides a reasonable balance between precision and recall for disaster tweet classification.

---

## 13. Confusion Matrix

| | Predicted Non-Disaster | Predicted Disaster |
|---|---:|---:|
| Actual Non-Disaster | 777 | 92 |
| Actual Disaster | 189 | 465 |

Interpretation:

- **True Negatives (TN): 777**
- **False Positives (FP): 92**
- **False Negatives (FN): 189**
- **True Positives (TP): 465**

---

## 14. Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Non-Disaster | 0.80 | 0.89 | 0.85 |
| Disaster | 0.83 | 0.71 | 0.77 |

---

## 15. ROC and Precision-Recall Analysis

The final model was evaluated using ROC and Precision-Recall analysis.

- **ROC-AUC: 0.8727**
- **Average Precision: 0.8728**

These results indicate good class discrimination across different classification thresholds.

---

## 16. Overfitting Check

The final model produced:

- Training F1-score: **83.82%**
- Test F1-score: **76.80%**
- Difference: **7.02 percentage points**

This indicates a moderate training-to-test performance gap, with reasonable generalization to unseen test data.

---

## 17. Model Serialization

The trained model and TF-IDF vectorizer were saved as:

- `disaster_tweet_model.pkl`
- `tfidf_vectorizer.pkl`

These files are loaded by the Flask application for local prediction and by the Streamlit application for public deployment.

---

## 18. Web Application

A Flask web application was initially developed to provide a simple local real-time prediction interface.

For public deployment, a Streamlit interface was created using the same trained model, TF-IDF vectorizer, and feature-engineering logic.

### Prediction Flow

```text
User Tweet
    ↓
Text Cleaning
    ↓
Tokenization
    ↓
Stopword Removal
    ↓
Lemmatization
    ↓
TF-IDF Transformation
    ↓
Additional Features
    ↓
Tuned Multinomial Naive Bayes Model
    ↓
Prediction + Confidence
    ↓
Web Interface
```

### Application Features

- User tweet input
- Real-time classification
- Disaster / Non-Disaster prediction
- Prediction confidence
- Text preprocessing
- TF-IDF transformation
- Tweet-level feature extraction

---

## 19. Live Application Deployment

The Project 7 application was publicly deployed using **Streamlit Community Cloud**.

### Live Application

**https://disaster-tweet-classification-baylfnxfscsp4xt4dmffxa.streamlit.app/**

### Deployment Details

- **Framework:** Streamlit
- **Hosting Platform:** Streamlit Community Cloud
- **Programming Language:** Python 3.13
- **Machine Learning Model:** Tuned Multinomial Naive Bayes
- **Text Representation:** TF-IDF
- **Additional Features:** Tweet length, hashtag presence, and mention presence
- **Model Artifact:** `disaster_tweet_model.pkl`
- **Vectorizer Artifact:** `tfidf_vectorizer.pkl`

---

## 20. Live Application Validation

The deployed application was tested using different tweet contexts.

### Test Case 1 — Real Disaster Context

> Breaking news: flood waters are rising rapidly in the area. Residents are being evacuated now. @news #Flood #Emergency

**Result:** Disaster Tweet — **93.14% confidence**

### Test Case 2 — Metaphorical Context

> I had a disaster trying to cook dinner tonight, completely ruined the recipe.

**Result:** Non-Disaster Tweet — **79.69% confidence**

These tests demonstrate that the deployed application can distinguish between a tweet describing an actual disaster event and a metaphorical use of the word "disaster".

---

## 21. GitHub Repository

**https://github.com/vsatish5621/Disaster-Tweet-Classification**

### Main Project Files

- `app.py` — Flask application
- `disaster_classifier_app.py` — Streamlit deployment application
- `disaster_tweet_model.pkl` — trained machine learning model
- `tfidf_vectorizer.pkl` — trained TF-IDF vectorizer
- `requirements.txt` — project dependencies
- `Templates/index.html` — Flask web interface
- `README.md` — project documentation
- `.gitignore` — ignored development files
- `.python-version` — Python version configuration

---

## 22. Technical Limitation

In this implementation, the TF-IDF vectorizer was fitted before the train/test split. Therefore, vocabulary information from the complete dataset was available during feature construction.

This introduces a degree of **data leakage**, so the reported test metrics should not be interpreted as a completely leakage-free benchmark.

For a future improved implementation, TF-IDF should be fitted only on the training data, preferably using a scikit-learn `Pipeline`.

The current results are retained because they represent the completed implementation used for this Project 7 submission.

---

## 23. Future Improvements

- Fit TF-IDF only on the training set using a Pipeline.
- Experiment with Word2Vec or GloVe embeddings.
- Add sentiment and other linguistic features to the model.
- Experiment with transformer-based NLP models.
- Perform more extensive hyperparameter tuning.
- Improve handling of slang, emojis, hashtags, and social-media abbreviations.
- Test the model on a completely independent unseen dataset.
- Add batch prediction functionality.
- Improve the Streamlit interface with visual analytics.
- Add model monitoring and real-time prediction statistics.

---

## 24. Conclusion

This project demonstrates an end-to-end NLP and Machine Learning workflow for disaster tweet classification.

The implementation includes:

- Exploratory Data Analysis
- Text preprocessing
- Tokenization and lemmatization
- TF-IDF feature engineering
- Tweet-level feature engineering
- Multiple machine learning models
- Cross-validation
- Hyperparameter tuning
- Model evaluation
- Confusion matrix analysis
- ROC-AUC and Precision-Recall analysis
- Model serialization
- Flask local application
- Streamlit public deployment
- Live prediction validation

The final tuned Multinomial Naive Bayes model achieved:

**Accuracy: 81.55%**

**F1-Score: 76.80%**

**ROC-AUC: 87.27%**

The project was successfully deployed as a public Streamlit application and validated using both real-disaster and metaphorical tweet examples.

---

## 25. Project Submission Links

### GitHub Repository

https://github.com/vsatish5621/Disaster-Tweet-Classification

### Live Application

https://disaster-tweet-classification-baylfnxfscsp4xt4dmffxa.streamlit.app/

---

## Project Information

**Project:** Disaster Tweet Classification using NLP and Machine Learning

**Submitted By:** Satish Vonteddu

**Program:** Data Analytics Internship

**Organization:** NextHikes IT Solutions

**Batch:** L-110226-W2PM-B15

**Submission Date:** 15 October 2026

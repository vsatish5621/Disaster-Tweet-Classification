# Disaster Tweet Classification using NLP and Machine Learning

## 1. Project Overview

This project develops a Natural Language Processing (NLP) and Machine Learning system to classify tweets into two categories:

- **Disaster** — tweets describing a real disaster event
- **Non-Disaster** — tweets that do not describe a real disaster event

The project focuses on understanding tweet context because disaster-related words may also be used metaphorically or in general conversations.

## 2. Problem Statement

Social media platforms contain a large amount of information during emergencies. Automatically identifying tweets that describe real disaster events can help organizations and emergency-response teams process information more efficiently.

The main challenge is that words such as earthquake, fire, flood, or disaster may appear in tweets that are not actually reporting a disaster.

## 3. Objectives

- Build a machine learning model for disaster tweet classification.
- Perform text preprocessing and NLP analysis.
- Convert text into numerical features using TF-IDF.
- Add tweet-level features such as tweet length, hashtags, and mentions.
- Compare multiple machine learning models.
- Tune the selected model.
- Evaluate the final model using classification metrics and curves.
- Serialize the trained model and TF-IDF vectorizer.
- Develop a Flask web application for real-time prediction.
- Test the application using different tweet contexts.

## 4. Dataset

The dataset contains the following columns:

- `id`
- `keyword`
- `location`
- `text`
- `target`

The actual CSV used in this implementation contains **7,613 records and 5 columns**.

Target labels:

- `0` = Non-Disaster
- `1` = Disaster

### Class Distribution

- Non-Disaster: **4,342 (57.03%)**
- Disaster: **3,271 (42.97%)**

> Note: The Project 7 brief describes a dataset of 10,000 hand-classified tweets. The actual dataset used for this implementation contains 7,613 records.

## 5. Exploratory Data Analysis

The following analysis was performed:

- Dataset structure and data types
- Missing-value analysis
- Target/class distribution
- Keyword frequency analysis
- Raw tweet inspection
- Text length analysis

Missing values were observed mainly in:

- `keyword`: 61
- `location`: 2,534

The `text` and `target` columns contained no missing values.

## 6. Text Preprocessing

The following NLP preprocessing steps were applied:

1. Convert text to lowercase.
2. Remove URLs.
3. Remove HTML content.
4. Remove punctuation and special characters.
5. Normalize whitespace.
6. Tokenize the text.
7. Remove English stopwords.
8. Apply lemmatization.
9. Create the final processed text.

## 7. Feature Engineering

### TF-IDF

TF-IDF was used to convert processed tweet text into numerical features.

Configuration:

- `max_features = 5000`

The resulting TF-IDF matrix contained:

- **7,613 tweets**
- **5,000 TF-IDF features**

### Additional Features

Three additional tweet-level features were included:

- `tweet_len`
- `has_hashtag`
- `has_mention`

The final feature matrix contained **5,003 features**.

## 8. Train/Test Split

The dataset was divided into:

- Training set: **6,090 records**
- Test set: **1,523 records**
- Test size: **20%**
- Stratified split
- `random_state = 42`

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

## 10. Cross-Validation

Five-fold cross-validation was performed.

### Naive Bayes

- Mean F1-score: **73.94%**
- Standard deviation: **1.44%**

### Logistic Regression

- Mean F1-score: **75.27%**
- Standard deviation: **0.82%**

Logistic Regression showed a higher and more stable cross-validation F1-score, while Naive Bayes achieved stronger final test performance after tuning.

## 11. Hyperparameter Tuning

### Logistic Regression

Parameters evaluated:

- `C`: 0.1, 1, 10
- `solver`: liblinear, lbfgs

Best configuration:

- `C = 1`
- `solver = lbfgs`

### Naive Bayes

Alpha values evaluated:

- 0.01
- 0.1
- 0.5
- 1.0
- 2.0

Best configuration:

- `alpha = 0.5`

The tuned Naive Bayes model was selected as the final model based on final test performance.

## 12. Final Model Performance

### Tuned Multinomial Naive Bayes

- Accuracy: **81.55%**
- Precision: **83.48%**
- Recall: **71.10%**
- F1-Score: **76.80%**
- ROC-AUC: **87.27%**
- Average Precision: **87.28%**

### Confusion Matrix

| | Predicted Non-Disaster | Predicted Disaster |
|---|---:|---:|
| Actual Non-Disaster | 777 | 92 |
| Actual Disaster | 189 | 465 |

## 13. Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Non-Disaster | 0.80 | 0.89 | 0.85 |
| Disaster | 0.83 | 0.71 | 0.77 |

## 14. Overfitting Check

The final model produced:

- Training F1-score: **83.82%**
- Test F1-score: **76.80%**
- Difference: **7.02 percentage points**

This indicates a moderate training-to-test performance gap, with reasonable generalization to unseen test data.

## 15. Model Serialization

The trained model and TF-IDF vectorizer were saved as:

- `disaster_tweet_model.pkl`
- `tfidf_vectorizer.pkl`

These files are loaded by the Flask application to make predictions on new tweets.

## 16. Flask Web Application

A Flask web application was developed to provide a simple user interface.

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
Trained Naive Bayes Model
    ↓
Prediction + Disaster Probability
    ↓
Web Interface
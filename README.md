# Restaurant-Review-Sentiment-Analysis
# Restaurant Review Sentiment Analysis

## Overview
This project performs **sentiment analysis** on Yelp restaurant reviews using **Natural Language Processing (NLP)** and **Machine Learning**. The goal is to classify customer reviews as **positive** or **negative** based on their star ratings.

## Features
✅ **Text Preprocessing** (Lowercasing, Tokenization, Stopword Removal, Lemmatization)  
✅ **TF-IDF Vectorization** for feature extraction  
✅ **Naïve Bayes Classifier** for sentiment classification  
✅ **Model Performance Evaluation** (Accuracy, Precision, Recall, F1-score)  
✅ **Confusion Matrix Visualization**  
✅ **Saves Trained Model for Deployment**  

## Dataset
- The dataset consists of **Yelp restaurant reviews**.
- It contains review text and a star rating (1 to 5).
- **Binary Sentiment Classification:**
  - ⭐ 1-2 stars → **Negative (0)**
  - ⭐ 4-5 stars → **Positive (1)**
  - ⭐ 3 stars → **Neutral (Removed)**

## Tech Stack
- **Python**
- **NLTK** (Natural Language Toolkit)
- **Scikit-learn** (Machine Learning)
- **Pandas & NumPy** (Data Processing)
- **Matplotlib & Seaborn** (Visualization)


## Output
- **Model Accuracy & Classification Report**
- **Confusion Matrix Heatmap**
- **Saved Model & TF-IDF Vectorizer for Future Predictions**


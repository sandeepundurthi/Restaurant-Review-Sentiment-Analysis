# 🍽️ Restaurant Review Sentiment Analysis

This project applies Natural Language Processing (NLP) and Machine Learning techniques to classify sentiment in Yelp restaurant reviews. Reviews are labeled as **Positive** or **Negative** based on star ratings.

---

## 📊 Overview

- **Objective**: Automatically classify customer sentiments to gain insights from restaurant reviews.
- **Target**: Binary classification — Positive (4–5 stars) or Negative (1–2 stars).
- **Removed**: Neutral reviews (3 stars) for a clear classification boundary.

---
## 📁 Folder Structure
'''
Restaurant-Review-Sentiment-Analysis/
├── data/
│   ├── yelp.csv
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   └── Restaurent_review.ipynb
├── README.md
├── requirements.txt
'''
---
## 🧠 Features

- Text preprocessing:
  - Lowercasing
  - Tokenization
  - Stopword removal
  - Lemmatization
- Feature extraction using **TF-IDF**
- Sentiment classification with **Naïve Bayes**
- Model evaluation: Accuracy, Precision, Recall, F1-score
- Visualization: Confusion Matrix Heatmap
- Saves trained model and vectorizer for deployment

---

## 📁 Dataset

- Source: [Yelp Reviews Dataset on Kaggle](https://www.kaggle.com/datasets/omkarsabnis/yelp-reviews-dataset)
- Includes: Review text + star ratings (1–5)
- Preprocessing:
  - 1–2 stars → Negative (0)
  - 4–5 stars → Positive (1)
  - 3 stars → Removed

---

## 🛠️ Tech Stack

| Component        | Libraries Used                    |
|------------------|-----------------------------------|
| Preprocessing    | NLTK                              |
| ML Model         | Scikit-learn (Naïve Bayes)        |
| Data Handling    | Pandas, NumPy                     |
| Visualization    | Matplotlib, Seaborn               |

---

## 📈 Output

- ✅ Model Accuracy & Metrics Report
- 🔍 Confusion Matrix Heatmap
- 💾 Saved Model (`sentiment_model.pkl`) & TF-IDF Vectorizer (`tfidf_vectorizer.pkl`)

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/sandeepundurthi/Restaurant-Review-Sentiment-Analysis.git
cd Restaurant-Review-Sentiment-Analysis

# Install dependencies
pip install -r requirements.txt

# Run the notebook or your own pipeline using the model
```

---

## 📬 Contact

Made with ❤️ by [Sandeep Undurthi](https://github.com/sandeepundurthi)

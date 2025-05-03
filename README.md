# 📝 Product Review Sentiment Analysis

This project uses Natural Language Processing (NLP) to analyze customer reviews from an e-commerce platform and classify them into **positive** or **negative** sentiments.

## 🚀 Key Features

- Text preprocessing (stopword removal, stemming)
- Feature extraction using TF-IDF
- Sentiment classification using Logistic Regression
- Exploratory Data Analysis (EDA) using Seaborn & Matplotlib
- Accuracy: ~85% on test data
- Confusion matrix, precision-recall metrics for evaluation

## 📊 Tech Stack

- Python, Pandas, NumPy
- Scikit-learn, NLTK
- Seaborn, Matplotlib
- Jupyter Notebook

## 📁 Folder Overview

- `data/`: CSV file containing review text and sentiment labels  
- `notebooks/`: Jupyter notebook for training & evaluation  
- `models/`: Serialized `.pkl` ML model  
- `app.py`: (Optional) Deployment-ready Streamlit/Flask app

## 🧪 Sample Result

> *Review:* "Absolutely loved the product quality!"  
> *Prediction:* **Positive**

## 🧰 How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/SentimentAnalysis.ipynb

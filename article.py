import os
import sys
import time
import json
import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import nltk
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, GlobalMaxPool1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
    Trainer,
    TrainingArguments,
    pipeline
)
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, Bidirectional
import streamlit as st
from transformers import DistilBertTokenizerFast, TFDistilBertModel
import pickle


import tensorflow as tf

import os
import boto3
import botocore
import re
import string
import emoji
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import streamlit as st

stop_words = set(stopwords.words('english'))
import mysql

from sqlalchemy import create_engine, text



            




# Set credentials before importing boto3
os.environ["AWS_ACCESS_KEY_ID"] = "AKIAQ2K25GY4IETXYXM2"
os.environ["AWS_SECRET_ACCESS_KEY"] = "J9vtnWa8It2O9bH/9REn/0zkJQJ2AksADe3SLCbt"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"



# Now create the S3 client
s3 = boto3.client("s3")

# Example test call to confirm credentials work
try:
    identity = boto3.client("sts").get_caller_identity()
    print("✅ Authenticated as:", identity["Arn"])
except Exception as e:
    print("❌ Credentials error:", e)


# os.environ["AWS_ACCESS_KEY_ID"] ="AKIAQ2K25GY4IETXYXM2"
# os.environ["AWS_SECRET_ACCESS_KEY"] = "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"
# os.environ["AWS_DEFAULT_REGION"] = "us-east-1"








BUCKET_NAME = "news-article-classifier"
KEY = "gru_model.h5"
LOCAL_PATH = "gru_model.h5"

s3 = boto3.client("s3")

try:
    if not os.path.exists(LOCAL_PATH):
        s3.download_file(BUCKET_NAME, KEY, LOCAL_PATH)
        print(f"Downloaded {KEY} to {LOCAL_PATH}")
    else:
        print(f"File {LOCAL_PATH} already exists locally")
except Exception as e:
    print("Download failed:", e)


if os.path.exists(LOCAL_PATH):
    GRU_model = tf.keras.models.load_model(LOCAL_PATH)
    print("Model loaded successfully!")
else:
    print("Model file not found!")

# Your AWS credentials should already be set via environment variables or config

BUCKET_NAME = "news-article-classifier"
KEY = "bert_classifier.h5"
LOCAL_PATH = "bert_classifier.h5"

s3 = boto3.client("s3")

try:
    if not os.path.exists(LOCAL_PATH):
        print(f"Downloading {KEY} from bucket {BUCKET_NAME}...")
        s3.download_file(BUCKET_NAME, KEY, LOCAL_PATH)
        print("Download complete.")
    else:
        print(f"{LOCAL_PATH} already exists locally.")
except Exception as e:
    print("Download failed:", e)

# Check if file exists
if os.path.exists(LOCAL_PATH):
    classifier = tf.keras.models.load_model(LOCAL_PATH)
    print(f"✅ File {LOCAL_PATH} is present!")
else:
    print(f"❌ File {LOCAL_PATH} is missing!")




BUCKET_NAME = "news-article-classifier"
KEY = "LogisticRegressor_model.pkl"
LOCAL_PATH = "LogisticRegressor_model.pkl"

s3 = boto3.client("s3")

try:
    if not os.path.exists(LOCAL_PATH):
        print(f"Downloading {KEY}...")
        s3.download_file(BUCKET_NAME, KEY, LOCAL_PATH)
        print(f"Downloaded {LOCAL_PATH}")
    else:
        print(f"File {LOCAL_PATH} already exists.")
except Exception as e:
    print(f"Error downloading {KEY}: {e}")

# Now you can safely load your model


if os.path.exists(LOCAL_PATH):
    LR_model = joblib.load(LOCAL_PATH)
    print("Logistic Regressor model loaded!")
else:
    print("Model file missing, cannot load.")



BUCKET_NAME = "news-article-classifier"
KEY = "grutokenizer (1).pkl"  # exact S3 key
LOCAL_PATH = "grutokenizer (1).pkl"  # exact local filename

s3 = boto3.client("s3")

try:
    if not os.path.exists(LOCAL_PATH):
        print(f"Downloading {KEY}...")
        s3.download_file(BUCKET_NAME, KEY, LOCAL_PATH)
        print(f"Downloaded {LOCAL_PATH}")
    else:
        print(f"{LOCAL_PATH} already exists.")
except Exception as e:
    print(f"Download failed: {e}")


if os.path.exists(LOCAL_PATH):
    tokenizer = joblib.load(LOCAL_PATH)
    print("Tokenizer loaded!")
else:
    print("Tokenizer file missing, cannot load.")



BUCKET_NAME = "news-article-classifier"
KEY = "tfidf_vectorizer(50000).pkl"  # exact S3 key
LOCAL_PATH = "tfidf_vectorizer(50000).pkl"  # local filename

s3 = boto3.client("s3")

try:
    if not os.path.exists(LOCAL_PATH):
        print(f"Downloading {KEY} from S3...")
        s3.download_file(BUCKET_NAME, KEY, LOCAL_PATH)
        print(f"Downloaded {LOCAL_PATH}")
    else:
        print(f"{LOCAL_PATH} already exists locally.")
except Exception as e:
    print(f"Download failed: {e}")

# Then load the vectorizer
import joblib
if os.path.exists(LOCAL_PATH):
    vectorizer = joblib.load(LOCAL_PATH)
    print("Vectorizer loaded successfully!")
else:
    print("Vectorizer file missing, cannot load.")
REGION = "us-east-1"




#set home page with titles of other pages
st.sidebar.title('HOME')
page=st.sidebar.radio("Getpage",["Project Info","word clouds for each category & EDA","Article Category Predictor"
                                 ])

if page=="Project Info":
    st.title("Multi-Model Article Classification System")

    st.write("""In today’s fast-paced digital world, organizing and categorizing text content 
             is more important than ever. News agencies, content platforms, and aggregators deal
              with massive volumes of unstructured data—and sorting it into meaningful categories is key
              to improving user experience and content management.
 """)

    st.write("""This app tackles that challenge head-on by offering a smart, scalable,
              and secure article classification system powered by cutting-edge Natural Language Processing (NLP) techniques.
 """)
    st.write("""What does it do?
Users can submit news articles, and the system will classify them into one of the following categories:\n
• 	 World\n
• 	 Business\n
• 	 Sports\n
• 	 Science & Technology\n

 """)
elif page=="word clouds for each category & EDA":
    st.header("News Article Classifier Comparison Table")
    st.image(r"C:\Users\shanm\OneDrive\Desktop\classifier_comparison.png")
    st.header("Average word count of news article(title and description combined)")
    st.write("Following the removal of HTML tags, URLs, emojis," \
    " punctuation, and excess whitespace, the cleaned text in the combined column" \
    " yields an average word length of 26.")
    
    st.header('word clouds')
    word_cloud=st.selectbox("choose category", ["Business","World", "Sci/Tech","Sports"])
    if word_cloud=="Business":
        st.image(r"C:\Users\shanm\OneDrive\Desktop\wordcloud.png")
    if word_cloud=="World":
        st.image(r"C:\Users\shanm\OneDrive\Desktop\wordcloud1.png")
    if word_cloud=="Sci/Tech":
        st.image(r"C:\Users\shanm\OneDrive\Desktop\wordcloud2.png")
    if word_cloud=="Sports":
        st.image(r"C:\Users\shanm\OneDrive\Desktop\wordcloud3.png")        
    confusion_matrix=st.selectbox("choose model", ["Logistic Regressor","GRU Classifier", "DistilBERT Classifier"])
    if confusion_matrix=="Logistic Regressor":
        st.image(r"C:\Users\shanm\Downloads\lr_confusion_matrix.png")      
    if confusion_matrix=="GRU Classifier":
        st.image(r"C:\Users\shanm\OneDrive\Desktop\gruconfusion_matrix.png")      
    if confusion_matrix=="DistilBERT Classifier":
        st.image(r"C:\Users\shanm\OneDrive\Desktop\bert_confusion_matrix.png")         





elif page=="Article Category Predictor":
    DB_USER="admin"
    DB_PASS="Rajamohan1238"
    DB_HOST="database-1.cwhm4q4qu9zx.us-east-1.rds.amazonaws.com"
    DB_PORT=3306
    DB_NAME="finalproject"
    engine = create_engine(f'mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}')

    with st.form("login_form"):
        username=st.text_input('Enter your name')
        submitted=st.form_submit_button('Login')
    if submitted and username:
        with engine.connect() as conn:
            conn.execute(
                text("INSERT INTO user_logins (user_name) VALUES (:username)"),
                {"username": username}
            )
        st.success(f"Welcome, {username}! Your login has been saved.")
    st.markdown("Enter an article or news snippet below to predict its category.")
    article = st.text_area("Enter article content or paste text here")

    model_choice = st.selectbox("Choose a model:", ["Logistic Regressor","GRU Classifier", "DistilBERT Classifier"])
    if model_choice == "Logistic Regressor":
            





            stop_words = set(ENGLISH_STOP_WORDS)

            def clean_text(text):
                text = BeautifulSoup(text, "html.parser").get_text()
                text = re.sub(r'http\S+|www\S+', '', text)
                text = emoji.replace_emoji(text, replace='')
                text = text.translate(str.maketrans('', '', string.punctuation))
                text = text.lower()
                text = re.sub(r'\s+', ' ', text).strip()
                tokens = text.split()
                tokens = [word for word in tokens if word not in stop_words]
                return ' '.join(tokens)

            def preprocess_article(article_text):
                cleaned = clean_text(article_text)
                vectorized = vectorizer.transform([cleaned])
                return vectorized

            def predict_article_category(article_text):
                label_map = {
                    0: "World",
                    1: "Sports",
                    2: "Business",
                    3: "Sci/Tech"
                }
                X_input = preprocess_article(article_text)
                prediction = LR_model.predict(X_input)[0]
                confidence = np.max(LR_model.predict_proba(X_input)) * 100
                return label_map.get(prediction, "Unknown"), confidence



            if st.button("🔍 Predict"):
                if article.strip():
                    category, confidence = predict_article_category(article)
                    st.success(f"**Predicted Category:** {category}")
                    st.info(f"**Confidence:** {confidence:.2f}%")
                else:
                    st.warning("Please enter some text before predicting.")

    if model_choice == "GRU Classifier":
            # model = load_model(r"C:\deskto\final_project\models\gru_model.h5")

            label_map = {
                        0: "World",
                        1: "Sports",
                        2: "Business",
                        3: "Sci/Tech"
                    }

            

            

            if st.button("🔍 Predict"):
                if article:
                    def clean_text(text):
                        # Remove HTML tags
                        text = BeautifulSoup(text, "html.parser").get_text()

                        # Remove URLs
                        text = re.sub(r'http\S+|www\S+', '', text)

                        # Remove emojis
                        text = emoji.replace_emoji(text, replace='')

                        # Remove punctuation
                        text = text.translate(str.maketrans('', '', string.punctuation))

                        # Lowercase everything
                        text = text.lower()

                        # Remove extra whitespace
                        text = re.sub(r'\s+', ' ', text).strip()

                        # Tokenize and remove stopwords
                        tokens = text.split()
                        tokens = [word for word in tokens if word not in stop_words]

                        return ' '.join(tokens)
                    article = clean_text(article)
                    seq = tokenizer.texts_to_sequences([article])
                    padded = pad_sequences(seq, maxlen=128)
                    pred_probs = GRU_model.predict(padded)
                    pred_class = np.argmax(pred_probs)
                    
                    confidence = np.max(pred_probs) * 100

                   
                    st.success(f"Predicted Label: {label_map[pred_class]}")
                    st.info(f"**Confidence:** {confidence:.2f}%")
                else:
                    st.warning("Please enter some text before predicting.")

    if model_choice == "DistilBERT Classifier":


            #  Load tokenizer and DistilBERT encoder
            tokenizer1 = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
            bert_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")

            label_map = {
                0: "World",
                1: "Sports",
                2: "Business",
                3: "Sci/Tech"
            }

       
            st.set_page_config(page_title="DistilBERT Text Classifier", layout="centered")
 

            if st.button("🔍 Predict"):
                if article.strip():
                    #  Tokenize input
                    encoding = tokenizer1(
                        [article],
                        truncation=True,
                        padding='max_length',
                        max_length=128,
                        return_tensors='tf'
                    )

                    #  Extract CLS embedding
                    outputs = bert_model(
                        input_ids=encoding['input_ids'],
                        attention_mask=encoding['attention_mask']
                    )
                    cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, 768)

                    #  Predict
                    preds = classifier.predict(cls_embedding)
                    pred_label = label_map[np.argmax(preds)]
                    confidence = np.max(preds) * 100

                    # Display result
                    st.success(f"**Predicted Label:** {pred_label}")
                    st.info(f"**Confidence:** {confidence:.2f}%")
                else:
                    st.warning("Please enter some text before predicting.")

             
    with engine.connect() as conn:
            conn.execute(
                text("INSERT INTO user_logins (article) VALUES (:article)"),
                {"article": article}
            )

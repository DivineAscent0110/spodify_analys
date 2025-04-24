import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

def clean_data(df):
    df = df.dropna(subset=['review_text', 'review_rating'])
    df = df[(df['review_rating'] >= 1) & (df['review_rating'] <= 5)]
    df['processed_review_text'] = df['review_text'].apply(preprocess_text)
    return df


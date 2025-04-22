import re
import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os
from dotenv import load_dotenv
import kagglehub
from tqdm import tqdm
from joblib import dump, load
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer  # Đổi sang TF-IDF

# Tải dữ liệu từ NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
load_dotenv()

class TwitterClient(object):
    """Client tương tác với Twitter API"""
    def __init__(self):
        pass
    
    def clean_tweet(self, tweet):
        """Tiền xử lý tweet, loại bỏ các ký tự không cần thiết"""
        return ' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    
    def get_tweet_sentiment(self, tweet):
        """Dự đoán cảm xúc của tweet"""
        analysis = TextBlob(self.clean_tweet(tweet))
        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1

def load_tweets_from_csv(file_path):
    """Tải dữ liệu từ file CSV"""
    try:
        df = pd.read_csv(file_path, encoding='latin1', header=None)
        df.columns = ["sentiment", "id", "date", "query", "user", "text"]
        df["sentiment"] = df["sentiment"].replace({0: -1, 4: 1})
        
        tweets = [{'text': row['text'], 'sentiment': row['sentiment']} for _, row in df.iterrows()]
        return tweets
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        return None

def preprocess_text(text, use_lemmatization=True):
    """Tiền xử lý văn bản: chuyển về chữ thường, loại bỏ stopwords và lemmatization"""
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def train_and_evaluate(tweets, model_type='naive_bayes', model_path="../models/TFIDF_NaiveBayes1_model.sav", c=1.0):
    """Huấn luyện mô hình và đánh giá kết quả"""
    texts = [tweet['text'] for tweet in tweets]
    sentiments = [tweet['sentiment'] for tweet in tweets]
    
    # Tiền xử lý văn bản
    processed_texts = [preprocess_text(text) for text in tqdm(texts, desc="Preprocessing text")]

    # Chuyển văn bản thành vector TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(2,3))
    X = vectorizer.fit_transform(processed_texts)

    y = np.array(sentiments)
    
    # Mã hóa nhãn
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Chia dữ liệu thành tập huấn luyện và kiểm thử
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=2, stratify=y_encoded)

    print(f"Số lượng mẫu trong tập huấn luyện trước SMOTE: {X_train.shape[0]}")

    # Áp dụng SMOTE để cân bằng dữ liệu
    smote = SMOTE(random_state=2)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Số lượng mẫu trong tập huấn luyện sau SMOTE: {X_train_resampled.shape[0]}")

    # Huấn luyện mô hình
    try:
        if model_type == 'naive_bayes':
            model = MultinomialNB( )
        else:
            raise ValueError('Loại mô hình không hợp lệ.')
        
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
    except ValueError as e:
        print(f"ValueError trong quá trình huấn luyện model {model_type}: {e}")
        return None, None

    # Lưu mô hình, vectorizer và label_encoder
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump((model, vectorizer, label_encoder), model_path)
    print(f"Model, Vectorizer, and LabelEncoder saved to {model_path}")
    
    # Đánh giá mô hình
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    y_test_decoded = label_encoder.inverse_transform(y_test)
    
    accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
    labels = [-1, 0, 1]
    report = classification_report(y_test_decoded, y_pred_decoded, labels=labels, zero_division=0)

    return accuracy, report

def load_model(model_path="../models/TFIDF_NaiveBayes1_model.sav"):
    """Tải mô hình, vectorizer và label encoder đã huấn luyện"""
    try:
        model, vectorizer, label_encoder = load(model_path)
        print(f"Model, Vectorizer, and LabelEncoder loaded from {model_path}")
        return model, vectorizer, label_encoder
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def main():
    api = TwitterClient()

    # Tải dataset từ Kaggle
    dataset_path = kagglehub.dataset_download("kazanova/sentiment140")
    csv_file = os.path.join(dataset_path, "training.1600000.processed.noemoticon.csv")
    print("Data path:" + str(csv_file))

    tweets = load_tweets_from_csv(csv_file)
    if not tweets:
        print("Không tìm thấy tweets.")
        return

    print(f"Số lượng tweet đã lấy: {len(tweets)}")

    # Thử nghiệm với các giá trị C khác nhau
    for c_value in [1.0]:
        accuracy, report = train_and_evaluate(tweets, model_type='naive_bayes', c=c_value)
        if accuracy is not None:
            print(f"  Model: Naive Bayes")
            print(f"  Accuracy: {accuracy:.2f}")
            print(f"  Classification Report:\n{report}")

    # Tải mô hình đã huấn luyện và kiểm tra
    loaded_model = load_model()
    if loaded_model:
        print("\nMô hình đã được tải thành công!")

if __name__ == "__main__":
    main()

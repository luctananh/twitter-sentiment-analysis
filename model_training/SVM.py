import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC  # Thay đổi từ SVC thành LinearSVC
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
from imblearn.over_sampling import SMOTE  # Thêm SMOTE
from sklearn.preprocessing import LabelEncoder  # Thêm LabelEncoder

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
load_dotenv()

class TwitterClient(object):
    def __init__(self):
        pass
    def clean_tweet(self, tweet):
        return ' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    def get_tweet_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))
        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1

def load_tweets_from_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding='latin1', header=None)
        df.columns = ["sentiment", "id", "date", "query", "user", "text"]
        df["sentiment"] = df["sentiment"].replace({0: -1, 4: 1})
        
        tweets = []
        for index, row in df.iterrows():
            parsed_tweet = {}
            parsed_tweet['text'] = row['text']
            parsed_tweet['sentiment'] = row['sentiment']
            tweets.append(parsed_tweet)
        return tweets
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        return None

def preprocess_text(text, use_lemmatization=True):
    text = text.lower()  # Chuyển văn bản về chữ thường
    stop_words = set(stopwords.words('english'))  # Lấy stopwords
    words = word_tokenize(text)  # Tách câu thành từ
    words = [word for word in words if word not in stop_words]  # Loại bỏ stopwords
    if use_lemmatization:
       lemmatizer = WordNetLemmatizer()  # Khởi tạo lemmatizer
       words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatize từ
    return ' '.join(words)  # Trả về văn bản đã tiền xử lý

def train_and_evaluate(tweets, vectorizer_type='tfidf', model_type='svm', model_path="models/SVM_model.sav", C=1.0): 
    texts = [tweet['text'] for tweet in tweets]
    sentiments = [tweet['sentiment'] for tweet in tweets]
    
    processed_texts = []
    for text in tqdm(texts, desc="Preprocessing text"):
       processed_texts.append(preprocess_text(text))

    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Sử dụng bigrams
        X = vectorizer.fit_transform(processed_texts)
    else:
        raise ValueError('Invalid vectorizer type. Please use "tfidf"')

    y = np.array(sentiments)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=2, stratify=y_encoded)

    print(f"Số lượng mẫu trong tập huấn luyện trước SMOTE: {X_train.shape[0]}")

    smote = SMOTE(random_state=2)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Số lượng mẫu trong tập huấn luyện sau SMOTE: {X_train_resampled.shape[0]}")

    try:
        if model_type == 'svm':
            model = LinearSVC(C=C, class_weight='balanced', dual=False)  # Thay đổi sang LinearSVC
        else:
            raise ValueError('Loại mô hình không hợp lệ.')
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
    except ValueError as e:
      print(f"ValueError trong quá trình huấn luyện model {model_type}: {e}")
      return None, None

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump((model, vectorizer, label_encoder), model_path)
    print(f"Model and Vectorizer saved to {model_path}")
    
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    y_test_decoded = label_encoder.inverse_transform(y_test)
    
    accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
    labels = [-1, 0, 1]
    report = classification_report(y_test_decoded, y_pred_decoded, labels=labels, zero_division=0)

    return accuracy, report

def load_model(model_path="models/SVM_model.sav"): 
   try:
      model = load(model_path)
      print(f"Model loaded from {model_path}")
      return model
   except Exception as e:
      print(f"Error loading model: {e}")
      return None

def main():
    api = TwitterClient()

    dataset_path = kagglehub.dataset_download("kazanova/sentiment140")
    csv_file = os.path.join(dataset_path, "training.1600000.processed.noemoticon.csv")
    print("Data path:" + str(csv_file))
    print("Chạy thử nghiệm với dữ liệu từ file csv")

    tweets = load_tweets_from_csv(csv_file)

    if not tweets:
        print("No tweets found.")
        return

    print(f"Số lượng tweet đã lấy: {len(tweets)}")

    print("\nRunning experiment with TF-IDF vectorization")
    
    for c_value in [1.0]:
        accuracy, report = train_and_evaluate(tweets, vectorizer_type='tfidf', model_type='svm', C=c_value)
        if accuracy is not None:
            print(f"  Model: LinearSVC")
            print(f"  Accuracy: {accuracy:.2f}")
            print(f"  Classification Report:\n{report}")

    loaded_model = load_model()
    if loaded_model:
        print("\nModel loaded successfully!")

if __name__ == "__main__":
    main()

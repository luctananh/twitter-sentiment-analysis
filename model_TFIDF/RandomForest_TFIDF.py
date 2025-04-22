import re
import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
load_dotenv()

class TwitterClient(object):
    def __init__(self):  # Corrected from init to __init__
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
    text = text.lower() # Chuyển văn bản về chữ thường
    stop_words = set(stopwords.words('english')) # Lấy stopwords
    words = word_tokenize(text) # Tách câu thành từ
    words = [word for word in words if word not in stop_words] # Loại bỏ stopwords
    if use_lemmatization:
        lemmatizer = WordNetLemmatizer() # Khởi tạo lemmatizer
        words = [lemmatizer.lemmatize(word) for word in words] # Lemmatize từ
    return " ".join(words) # Trả về chuỗi các từ đã tiền xử lý

def train_and_evaluate(tweets, vectorizer_type='tfidf', model_type='random_forest', model_path="models/TFIDF_RandomForest_models.sav", C=1.0):
    texts = [tweet['text'] for tweet in tweets]
    sentiments = [tweet['sentiment'] for tweet in tweets]

    # Tiền xử lý văn bản
    processed_texts = []
    for text in tqdm(texts, desc="Preprocessing text"):
       processed_texts.append(preprocess_text(text))

    # Sử dụng TF-IDF để vector hóa văn bản
    if vectorizer_type == 'tfidf':
        # Khởi tạo TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer()

        # Fit và transform văn bản đã tiền xử lý
        X = tfidf_vectorizer.fit_transform(processed_texts)
    else:
        raise ValueError('Invalid vectorizer type. Please use "tfidf"')

    y = np.array(sentiments)

    # Mã hóa nhãn
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Chia dữ liệu thành tập huấn luyện và kiểm thử
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=2, stratify=y_encoded)

    print(f"Số lượng mẫu trong tập huấn luyện trước SMOTE: {X_train.shape[0]}")

    # Áp dụng SMOTE để xử lý mất cân bằng lớp
    smote = SMOTE(random_state=2)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"Số lượng mẫu trong tập huấn luyện sau SMOTE: {X_train_resampled.shape[0]}")

    # Lựa chọn và huấn luyện mô hình RandomForest
    try:
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100,max_depth=50, random_state=42, class_weight='balanced', verbose=1)  # Thêm class_weight='balanced'
        else:
            raise ValueError('Loại mô hình không hợp lệ.')
        
        # Thêm vòng lặp để theo dõi quá trình huấn luyện
        for i in tqdm(range(10), desc="Training RandomForest"):  # Ví dụ: chia quá trình huấn luyện thành 10 bước
            model.fit(X_train_resampled, y_train_resampled)  # Huấn luyện mô hình trong mỗi bước
            # In một thông báo để biết chương trình không bị đơ
            print(f"Hoàn thành bước huấn luyện {i+1}/10")
        
        y_pred = model.predict(X_test)
    except ValueError as e:
        print(f"ValueError trong quá trình huấn luyện model {model_type}: {e}")
        return None, None

    # Lưu mô hình đã huấn luyện với tên mới
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump((model, tfidf_vectorizer, label_encoder), model_path)
    print(f"Model and Vectorizer saved to {model_path}")

    # Giải mã nhãn dự đoán
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    y_test_decoded = label_encoder.inverse_transform(y_test)

    # Đánh giá mô hình
    accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
    labels = [-1, 0, 1]
    report = classification_report(y_test_decoded, y_pred_decoded, labels=labels, zero_division=0)

    return accuracy, report

def load_model(model_path="models/TFIDF_RandomForest_models.sav"):
    """Load the trained model from a file."""
    try:
        model = load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def main():
    api = TwitterClient()

    # Download the dataset from Kaggle
    dataset_path = kagglehub.dataset_download("kazanova/sentiment140")
    csv_file = os.path.join(dataset_path, "training.1600000.processed.noemoticon.csv")
    print("Data path:" + str(csv_file))
    print("Chạy thử nghiệm với dữ liệu từ file csv")

    tweets = load_tweets_from_csv(csv_file)

    if not tweets:
        print("No tweets found.")
        return


    print("\nRunning experiment with TF-IDF vectorization")

    accuracy, report = train_and_evaluate(tweets, vectorizer_type='tfidf', model_type='random_forest')
    if accuracy is not None:
        print(f"  Model: Random Forest")
        print(f"  Accuracy: {accuracy:.2f}")
        print(f"  Classification Report:\n{report}")

    # Load the model and test it
    loaded_model = load_model()
    if loaded_model:
        print("\nModel loaded successfully!")

if __name__ == "__main__":
    main()
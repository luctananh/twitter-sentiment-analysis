import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import LabelEncoder

from config import (
    MODEL_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    USE_LEMMATIZATION,
    TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF,
    TFIDF_MAX_DF,
    TFIDF_SUBLINEAR,
    LR_C,
    LR_MAX_ITER,
)

load_dotenv()


# Khởi tạo tài nguyên NLTK chỉ một lần
def init_nltk():
    """Tải xuống tài nguyên NLTK nếu chưa tồn tại."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")


init_nltk()

# Bộ nhớ cache toàn cục cho stopwords và lemmatizer
_STOPWORDS = None
_LEMMATIZER = None


def get_stopwords():
    """Lấy stopwords từ cache để tránh tải lại."""
    global _STOPWORDS
    if _STOPWORDS is None:
        _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


def get_lemmatizer():
    """Lấy instance lemmatizer từ cache."""
    global _LEMMATIZER
    if _LEMMATIZER is None:
        _LEMMATIZER = WordNetLemmatizer()
    return _LEMMATIZER


# Đọc tweets từ file CSV
def load_tweets_from_csv(file_path):
    """Đọc tweets từ file CSV hiệu quả hơn sử dụng vectorized operations."""
    try:
        df = pd.read_csv(file_path, encoding="latin1", header=None)
        df.columns = ["sentiment", "id", "date", "query", "user", "text"]
        df["sentiment"] = df["sentiment"].replace({0: -1, 4: 1})

        tweets = df[["text", "sentiment"]].to_dict("records")
        return tweets
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        return None


def preprocess_text(text):
    """Tiền xử lý văn bản: làm sạch, tokenization, lemmatization."""
    # Làm sạch tweet: xoá @mentions, URLs, ký tự đặc biệt
    text = " ".join(
        re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split()
    )

    # Chuyển sang chữ thường
    text = text.lower()

    # Xoá stopwords
    stop_words = get_stopwords()
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]

    # Áp dụng lemmatization theo config
    if USE_LEMMATIZATION:
        lemmatizer = get_lemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


def train_and_evaluate(
    tweets,
    vectorizer_type="tfidf",
    model_type="logistic_regression",
    model_path=MODEL_PATH,
    C=LR_C,
):
    """Huấn luyện và đánh giá mô hình với TF-IDF vectorization."""
    texts = [tweet["text"] for tweet in tweets]
    sentiments = [tweet["sentiment"] for tweet in tweets]

    # Tiền xử lý văn bản
    processed_texts = [
        preprocess_text(text) for text in tqdm(texts, desc="Tiền xử lý văn bản")
    ]

    # Vectorization
    if vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            sublinear_tf=TFIDF_SUBLINEAR,
        )
        X = vectorizer.fit_transform(processed_texts)
    else:
        raise ValueError('Loại vectorizer không hợp lệ. Vui lòng dùng "tfidf"')

    y = np.array(sentiments)

    # Mã hóa biến mục tiêu thành các giá trị số
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Chia dữ liệu thành tập huấn luyện và tập kiểm thử
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )

    print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    # Huấn luyện mô hình
    try:
        if model_type == "logistic_regression":
            model = LogisticRegression(max_iter=LR_MAX_ITER, C=C)
        else:
            raise ValueError("Loại mô hình không hợp lệ.")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    except ValueError as e:
        print(f"ValueError trong quá trình huấn luyện model {model_type}: {e}")
        return None, None

    # Lưu mô hình đã huấn luyện với tất cả thành phần cần thiết
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump((model, vectorizer, label_encoder), model_path)
    print(f"Mô hình và Vectorizer đã được lưu vào {model_path}")

    # Giải mã các nhãn dự đoán
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    y_test_decoded = label_encoder.inverse_transform(y_test)

    # Đánh giá — chỉ dùng nhãn thực tế trong dataset (không có neutral=0)
    accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
    existing_labels = sorted(label_encoder.classes_.tolist())
    report = classification_report(
        y_test_decoded, y_pred_decoded, labels=existing_labels, zero_division=0
    )

    return accuracy, report


def load_model(model_path=MODEL_PATH):
    """Tải mô hình đã huấn luyện và tất cả thành phần từ file."""
    try:
        model, vectorizer, label_encoder = load(model_path)
        print(f"Mô hình đã được tải từ {model_path}")
        return model, vectorizer, label_encoder
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return None, None, None


def predict_sentiment(text, model, vectorizer, label_encoder):
    """Dự đoán cảm tính cho văn bản sử dụng mô hình đã tải."""
    if model is None or vectorizer is None or label_encoder is None:
        print("Các thành phần mô hình chưa được tải đúng.")
        return None

    processed_text = preprocess_text(text)
    X = vectorizer.transform([processed_text])
    y_pred_encoded = model.predict(X)[0]
    y_pred = label_encoder.inverse_transform([y_pred_encoded])[0]
    return y_pred


def main():
    # Tải dataset từ Kaggle
    dataset_path = kagglehub.dataset_download("kazanova/sentiment140")
    csv_file = os.path.join(dataset_path, "training.1600000.processed.noemoticon.csv")
    print(f"Đường dẫn dữ liệu: {csv_file}")
    print("Đang đọc tweets từ file CSV...\n")

    tweets = load_tweets_from_csv(csv_file)

    if not tweets:
        print("Không tìm thấy tweets.")
        return

    print(f"Số lượng tweet đã lấy: {len(tweets)}")
    print(
        f"\nCấu hình: C={LR_C} | ngram={TFIDF_NGRAM_RANGE} | min_df={TFIDF_MIN_DF} | lemma={USE_LEMMATIZATION}"
    )
    print("Thực thi thử nghiệm với TF-IDF + Logistic Regression\n")

    accuracy, report = train_and_evaluate(tweets)
    if accuracy is not None:
        print(f"  Mô hình: logistic_regression")
        print(f"  C: {LR_C}")
        print(f"  Độ chính xác: {accuracy:.2f}")
        print(f"  Báo cáo phân loại:\n{report}")

    # Tải mô hình và kiểm tra với dự đoán mẫu
    model, vectorizer, label_encoder = load_model()
    if model is not None:
        print("\nMô hình đã được tải thành công!")
        sample_tweet = "I love this product, it's amazing!"
        sentiment = predict_sentiment(sample_tweet, model, vectorizer, label_encoder)
        print(f"Dự đoán mẫu: '{sample_tweet}' -> Cảm tính: {sentiment}")


if __name__ == "__main__":
    main()

import os
import joblib
import traceback
import numpy as np
import torch
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
from gensim.models import Word2Vec
import tweepy
from dotenv import load_dotenv
import streamlit as st  # Thêm import streamlit
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import kagglehub  # Import kagglehub

load_dotenv()  # Load environment variables from .env file

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Thêm hàm clean_tweet để loại bỏ mentions, ký tự đặc biệt và URLs
def clean_tweet(tweet):
    return ' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

# Tiền xử lý văn bản
def preprocess_text(text, use_lemmatization=True):
    text = clean_tweet(text)  # Clean tweet first
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum() and word not in stop_words]
    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Định nghĩa các mô hình và đường dẫn
MODEL_FILES = {
    "Logistic Regression (TF-IDF)": "models/TFIDF_LogisticRegression_model.sav",
    "RandomForest (TF-IDF)": "models/TFIDF_RandomForest_model.sav",
    "NaiveBayes (TF-IDF)": "models/TFIDF_NaiveBayes_model.sav",
    "LinearSVC (TF-IDF)": "models/TFIDF_SVM_model.sav",
    "Naive Bayes (BoW)": "models/NaiveBayes_model.sav",
    "Random Forest (W2v)": "models/RandomForest_models.sav",
    "Random Forest (TinyBERT)": "models/RandomForest2_models.sav",
    "SVM (Word2Vec)": "models/SVM_word2vec_model.sav",
}

# Tải mô hình
def load_model(model_path, selected_model_name):
    if not os.path.exists(model_path):
        st.error(f"Không tìm thấy mô hình: {model_path}")
        return None, None, None
    try:
        model, vectorizer, label_encoder = joblib.load(model_path)
        return model, vectorizer, label_encoder
    except Exception as e:
        st.error(f"Lỗi tải mô hình {selected_model_name}: {e}")
        st.text(traceback.format_exc())
        return None, None, None

# Chuyển đổi văn bản thành vector với Word2Vec
def word2vec_transform(text, model):
    words = text.split()
    vector_size = model.vector_size
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

# Dự đoán cảm xúc
def predict_sentiment(text, model, vectorizer, label_encoder, model_name):
    cleaned_text = preprocess_text(text)

    # Vector hóa văn bản
    if isinstance(vectorizer, TfidfVectorizer) or isinstance(vectorizer, CountVectorizer):
        text_vectorized = vectorizer.transform([cleaned_text])
    elif isinstance(vectorizer, Word2Vec):
        text_vectorized = word2vec_transform(cleaned_text, vectorizer).reshape(1, -1)
    elif isinstance(vectorizer, BertTokenizer):
        tokenizer = vectorizer
        model_bert = BertModel.from_pretrained("bert-base-uncased")
        inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            text_vectorized = model_bert(**inputs).last_hidden_state[:, 0, :].numpy()
        
        # Cắt bớt chiều từ 768 về 312 bằng slicing
        if text_vectorized.shape[1] > 312:
            text_vectorized = text_vectorized[:, :312]  # Cắt bớt số chiều
        elif text_vectorized.shape[1] < 312:
            st.warning(f"Số chiều không đủ. Bỏ qua mẫu này.")
            return "Lỗi"

    else:
        st.error("Vectorizer không hỗ trợ hoặc không hợp lệ.")
        return "Lỗi"

    # Kiểm tra số chiều đầu vào
    if text_vectorized.shape[1] != model.n_features_in_:
        st.error(f"Lỗi: Mô hình yêu cầu {model.n_features_in_} features, nhưng dữ liệu có {text_vectorized.shape[1]} features.")
        return "Lỗi"

    # Dự đoán kết quả
    prediction = model.predict(text_vectorized)
    decoded_prediction = label_encoder.inverse_transform(prediction)[0]

    # Điều chỉnh lại kết quả theo yêu cầu
    if decoded_prediction == 1:
        return "Tích cực"
    elif decoded_prediction == 0:
        return "Trung lập"
    else:
        return "Tiêu cực"

# Tìm kiếm và phân tích Twitter
def get_tweets(query, api_client, tweet_count=10):
    try:
        response = api_client.search_recent_tweets(
            query=query,
            tweet_fields=["text"],
            max_results=tweet_count
        )
        if response.data:
            return [tweet.text for tweet in response.data]
        else:
            return []
    except tweepy.errors.TooManyRequests as e:
        print(f"Error fetching tweets: {e}")
        st.error("Do chính sách của Twitter, bạn cần chờ 15 phút trước khi thực hiện yêu cầu tiếp theo.")
        return None
    except Exception as e:
        print(f"Error fetching tweets: {e}")
        return []

# Hàm đánh giá mô hình
def evaluate_model(model, vectorizer, label_encoder, model_name, eval_df):
    # Tiền xử lý dữ liệu đánh giá
    eval_df['cleaned_text'] = eval_df['text'].apply(preprocess_text)

    # Vector hóa dữ liệu đánh giá
    if isinstance(vectorizer, TfidfVectorizer) or isinstance(vectorizer, CountVectorizer):
        X_eval = vectorizer.transform(eval_df['cleaned_text'])
    elif isinstance(vectorizer, Word2Vec):
        X_eval = np.array([word2vec_transform(text, vectorizer) for text in eval_df['cleaned_text']])
    elif isinstance(vectorizer, BertTokenizer):
        tokenizer = vectorizer
        model_bert = BertModel.from_pretrained("bert-base-uncased")
        X_eval = []
        for text in eval_df['cleaned_text']:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                text_vectorized = model_bert(**inputs).last_hidden_state[:, 0, :].numpy()
                if text_vectorized.shape[1] > 312:
                    text_vectorized = text_vectorized[:, :312]
                elif text_vectorized.shape[1] < 312:
                    st.warning(f"Số chiều không đủ. Bỏ qua mẫu này.")
                    continue
                X_eval.append(text_vectorized)
        X_eval = np.concatenate(X_eval, axis=0)
    else:
        st.error("Vectorizer không hỗ trợ hoặc không hợp lệ.")
        return None, None

    # Chuyển đổi nhãn
    y_eval = label_encoder.transform(eval_df['sentiment'])
    # Dự đoán
    y_pred = model.predict(X_eval)
    # Đánh giá
    accuracy = accuracy_score(y_eval, y_pred)
    report = classification_report(y_eval, y_pred, output_dict=True) # output_dict=True

    return accuracy, report
# Giao diện ứng dụng Streamlit
def main():
    st.title("Phân tích cảm xúc Twitter")

    # Lựa chọn tab
    selected_tab = st.radio("Chọn chức năng", ("Phân tích cảm xúc Twitter", "Tìm kiếm và phân tích Twitter", "Đánh giá mô hình"))

    if selected_tab == "Phân tích cảm xúc Twitter":
        # Phần phân tích cảm xúc Twitter
        selected_model_name = st.selectbox("Chọn mô hình phân tích cảm xúc", list(MODEL_FILES.keys()))
        model_path = MODEL_FILES[selected_model_name]

        text_input = st.text_area("Nhập văn bản:")

        if st.button("Phân Tích"):
            model, vectorizer, label_encoder = load_model(model_path, selected_model_name)

            if model is None:
                st.error("Không thể tải model.")
            else:
                sentiment = predict_sentiment(text_input, model, vectorizer, label_encoder, selected_model_name)
                st.write(f"Kết quả: **{sentiment}**")

    elif selected_tab == "Tìm kiếm và phân tích Twitter":
        # Phần tìm kiếm và phân tích Twitter
        bearer_token = os.getenv("BEARER_TOKEN")

        if not bearer_token:
            st.error("Vui lòng thiết lập biến môi trường cho Twitter Bearer Token.")
            return

        api_client = tweepy.Client(bearer_token=bearer_token)

        search_query = st.text_input("Nhập từ khóa tìm kiếm Twitter:")
        tweet_count = st.number_input("Số lượng Twitter muốn lấy:", min_value=1, max_value=20, value=10)

        # Thêm nút Tìm kiếm
        if st.button("Tìm kiếm"):
            if search_query:
                tweets = get_tweets(search_query, api_client, tweet_count)
                if tweets is not None:
                    if tweets:
                        st.session_state.tweets = tweets  # Lưu bài viết tìm được vào session_state
                        st.write("Kết quả tìm kiếm:")
                        for tweet in tweets:
                            st.markdown(f"- **{tweet}")
                    else:
                        st.write("Không tìm thấy Twitter nào.")
                else:
                    st.write("Đã xảy ra lỗi khi lấy bài viết.")
            else:
                st.write("Vui lòng nhập từ khóa tìm kiếm.")

        # Chọn mô hình và phân tích cảm xúc cho các bài viết tìm được
        if 'tweets' in st.session_state:
            selected_model_name = st.selectbox("Chọn mô hình phân tích cảm xúc", list(MODEL_FILES.keys()))
            model_path = MODEL_FILES[selected_model_name]

            if st.button("Phân tích"):
                model, vectorizer, label_encoder = load_model(model_path, selected_model_name)

                if model:
                    for tweet in st.session_state.tweets:
                        sentiment = predict_sentiment(tweet, model, vectorizer, label_encoder, selected_model_name)
                        st.markdown(f"- **{tweet}**")
                        st.write(f"  Cảm xúc: **{sentiment}**")
                else:
                    st.error("Không thể tải model.")
        else:
            st.write("Vui lòng tìm kiếm bài viết trước khi phân tích.")

    elif selected_tab == "Đánh giá mô hình":

        selected_model_name = st.selectbox("Chọn mô hình để đánh giá", list(MODEL_FILES.keys()))
        model_path = MODEL_FILES[selected_model_name]

        # Hiển thị thông tin mô hình đã chọn
        st.write(f"Bạn đã chọn mô hình: **{selected_model_name}**")

        # Lựa chọn nguồn dữ liệu: Kaggle hoặc Upload File
        data_source = st.radio("Chọn nguồn dữ liệu đánh giá:", ("Kaggle", "Upload File"))

        # Tải dataset từ Kaggle (chỉ khi nút được nhấn và lựa chọn Kaggle)
        csv_file = None  # Khởi tạo csv_file ở đây
        if data_source == "Kaggle":
            try:
                dataset_path = kagglehub.dataset_download("kazanova/sentiment140")
                csv_file = os.path.join(dataset_path, "training.1600000.processed.noemoticon.csv")
                st.success(f"Dataset đã được tải xuống thành công từ: {dataset_path}")
            except Exception as e:
                st.error(f"Lỗi khi tải dataset từ Kaggle: {e}")
                csv_file = None

        # Upload File (chỉ khi nút được nhấn và lựa chọn Upload File)
        elif data_source == "Upload File":
            uploaded_file = st.file_uploader("Tải lên file CSV chứa dữ liệu đánh giá", type="csv")
            if uploaded_file is not None:
                csv_file = uploaded_file
            else:
                csv_file = None

        # Khởi tạo các biến trong session state nếu chưa có
        if 'evaluation_done' not in st.session_state:
            st.session_state.evaluation_done = False
        if 'accuracy' not in st.session_state:
            st.session_state.accuracy = None
        if 'report_df' not in st.session_state:
            st.session_state.report_df = None

        # Tạo một empty container để chứa nút "Bắt đầu đánh giá"
        eval_button_container = st.empty()

        if csv_file is not None:
            with eval_button_container: #hiển thị nút khi có csv_file
                if st.button("Bắt đầu đánh giá"): # Kiểm tra csv_file trước khi thực hiện
                    try:
                        if data_source == "Upload File":
                            eval_df = pd.read_csv(csv_file, encoding='latin1', header=None, names=["sentiment", "id", "date", "query", "user", "text"])
                        else:  # data_source == "Kaggle"
                            eval_df = pd.read_csv(csv_file, encoding='latin1', header=None, names=["sentiment", "id", "date", "query", "user", "text"])

                        # Remap sentiment labels
                        eval_df["sentiment"] = eval_df["sentiment"].replace({0: -1, 4: 1})

                        # Xác định số lượng mẫu tối đa có thể lấy từ mỗi lớp
                        num_positive = len(eval_df[eval_df['sentiment'] == 1])
                        num_negative = len(eval_df[eval_df['sentiment'] == -1])
                        sample_size = min(160000, num_positive, num_negative)

                        # Cân bằng dữ liệu đánh giá (tối đa 10000 mẫu, sample_size tích cực, sample_size tiêu cực)
                        positive_samples = eval_df[eval_df['sentiment'] == 1].sample(n=sample_size, random_state=42)  # Lấy ngẫu nhiên mẫu tích cực
                        negative_samples = eval_df[eval_df['sentiment'] == -1].sample(n=sample_size, random_state=42)  # Lấy ngẫu nhiên mẫu tiêu cực
                        eval_df = pd.concat([positive_samples, negative_samples])  # Kết hợp lại
                        eval_df = eval_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Trộn ngẫu nhiên và reset index

                    except Exception as e:
                        st.error(f"Lỗi khi đọc hoặc xử lý file CSV: {e}")
                        eval_df = None

                    if eval_df is not None:
                        model, vectorizer, label_encoder = load_model(model_path, selected_model_name)

                        if model is None:
                            st.error("Không thể tải model.")
                        else:
                            with st.spinner("Đang đánh giá mô hình..."):  # Thêm spinner
                                accuracy, report = evaluate_model(model, vectorizer, label_encoder, selected_model_name, eval_df.copy()) #truyền bản copy để tránh lỗi
                            if accuracy is not None:
                                st.session_state.evaluation_done = True
                                st.session_state.accuracy = accuracy
                                st.session_state.report_df = pd.DataFrame(report).transpose()
                                st.session_state.report_df = st.session_state.report_df.applymap(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
                            else:
                                st.error("Đã xảy ra lỗi trong quá trình đánh giá.")
                                st.session_state.evaluation_done = False # Reset nếu có lỗi
                    else:
                        st.session_state.evaluation_done = False # Reset nếu có lỗi

        else:
            if data_source is not None: #hiển thị cảnh báo nếu đã chọn nguồn nhưng chưa có file
                st.warning("Vui lòng tải lên một tập tin CSV hoặc chờ dataset từ 'Kaggle' được tải xuống thành công.")

        # Hiển thị kết quả đánh giá (nếu có)
        if st.session_state.evaluation_done and st.session_state.accuracy is not None:
            st.metric("Accuracy:", f"{st.session_state.accuracy:.2f}")
            st.dataframe(st.session_state.report_df)

if __name__ == "__main__":
    main()
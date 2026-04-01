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
import tweepy
from dotenv import load_dotenv
import streamlit as st

load_dotenv()  # Load environment variables from .env file

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)


# Thêm hàm clean_tweet để loại bỏ mentions, ký tự đặc biệt và URLs
def clean_tweet(tweet):
    return " ".join(
        re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()
    )


# Tiền xử lý văn bản
def preprocess_text(text, use_lemmatization=True):
    text = clean_tweet(text)  # Clean tweet first
    text = text.lower()
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum() and word not in stop_words]
    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)


# Tự động quét thư mục models/ để lấy danh sách mô hình
MODELS_DIR = "models"


def get_model_files():
    if not os.path.exists(MODELS_DIR):
        return {}
    return {
        os.path.splitext(f)[0]: os.path.join(MODELS_DIR, f)
        for f in sorted(os.listdir(MODELS_DIR))
        if f.endswith(".sav")
    }


MODEL_FILES = get_model_files()


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


# Dự đoán cảm xúc
def predict_sentiment(text, model, vectorizer, label_encoder, model_name):
    cleaned_text = preprocess_text(text)

    # Vector hóa văn bản
    if isinstance(vectorizer, TfidfVectorizer) or isinstance(
        vectorizer, CountVectorizer
    ):
        text_vectorized = vectorizer.transform([cleaned_text])
    elif isinstance(vectorizer, BertTokenizer):
        tokenizer = vectorizer
        model_bert = BertModel.from_pretrained("bert-base-uncased")
        inputs = tokenizer(
            cleaned_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
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
        st.error(
            f"Lỗi: Mô hình yêu cầu {model.n_features_in_} features, nhưng dữ liệu có {text_vectorized.shape[1]} features."
        )
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
            query=query, tweet_fields=["text"], max_results=tweet_count
        )
        if response.data:
            return [tweet.text for tweet in response.data]
        else:
            return []
    except tweepy.errors.TooManyRequests as e:
        print(f"Error fetching tweets: {e}")
        st.error(
            "Do chính sách của Twitter, bạn cần chờ 15 phút trước khi thực hiện yêu cầu tiếp theo."
        )
        return None
    except Exception as e:
        print(f"Error fetching tweets: {e}")
        return []


# Giao diện ứng dụng Streamlit
def main():
    st.set_page_config(
        page_title="Twitter Sentiment Analysis",
        page_icon="🐦",
        layout="wide",
    )

    # Sidebar
    with st.sidebar:
        st.markdown("## 📖 Hướng Dẫn")
        st.markdown(
            "**📝 Phân Tích Cảm Xúc**\n"
            "- Nhập văn bản để phân tích cảm xúc\n"
            "- Chọn mô hình ML phù hợp\n\n"
            "**🔍 Tìm Kiếm Twitter**\n"
            "- Tìm tweets theo từ khóa\n"
            "- Phân tích cảm xúc hàng loạt\n"
            "- Cần Twitter API Bearer Token"
        )
        st.markdown("---")
        st.info(
            "**Phiên bản:** v1.0\n\n"
            "**Công nghệ:** TF-IDF, BERT, Word2Vec\n\n"
            "**Framework:** Streamlit + Scikit-learn"
        )

    # Header
    st.markdown(
        "<h1 style='text-align:center; color:#1DA1F2;'>🐦 Phân Tích Cảm Xúc Twitter</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#657786;'>Công cụ phân tích cảm xúc cho Twitter & văn bản</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Tabs
    tab1, tab2 = st.tabs(["📝 Phân Tích Cảm Xúc", "🔍 Tìm Kiếm Twitter"])

    # ==================== TAB 1 ====================
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_model = st.selectbox(
                "🤖 Chọn mô hình phân tích", list(MODEL_FILES.keys())
            )

        text_input = st.text_area(
            "✍️ Nhập văn bản cần phân tích:",
            placeholder="Ví dụ: I love this product! It's amazing...",
            height=150,
        )

        col1, col2, _ = st.columns([1, 1, 2])
        with col1:
            analyze_btn = st.button("🔍 Phân Tích", use_container_width=True)
        with col2:
            if st.button("🗑️ Xóa", use_container_width=True):
                st.rerun()

        if analyze_btn:
            if not text_input.strip():
                st.warning("⚠️ Vui lòng nhập văn bản!")
            else:
                model, vectorizer, label_encoder = load_model(
                    MODEL_FILES[selected_model], selected_model
                )
                if model is None:
                    st.error("❌ Không thể tải mô hình.")
                else:
                    with st.spinner("Đang phân tích..."):
                        sentiment = predict_sentiment(
                            text_input, model, vectorizer, label_encoder, selected_model
                        )
                    st.markdown("---")
                    if sentiment == "Tích cực":
                        st.success(f"✅ Kết quả: **{sentiment}**")
                    elif sentiment == "Trung lập":
                        st.info(f"ℹ️ Kết quả: **{sentiment}**")
                    else:
                        st.error(f"❌ Kết quả: **{sentiment}**")

    # ==================== TAB 2 ====================
    with tab2:
        bearer_token = os.getenv("BEARER_TOKEN")
        if not bearer_token:
            st.error("❌ Thiếu BEARER_TOKEN trong file .env")
            st.info(
                "1. Đăng ký Twitter Developer Account\n"
                "2. Tạo Bearer Token\n"
                "3. Thêm `BEARER_TOKEN=...` vào file `.env`"
            )
        else:
            api_client = tweepy.Client(bearer_token=bearer_token)

            col1, col2 = st.columns([2, 1])
            with col1:
                search_query = st.text_input(
                    "🔎 Từ khóa tìm kiếm:",
                    placeholder="#AI, #Python, sentiment...",
                )
            with col2:
                tweet_count = st.number_input(
                    "📊 Số lượng:", min_value=1, max_value=100, value=10
                )

            if st.button("🔍 Tìm Kiếm", use_container_width=True):
                if not search_query.strip():
                    st.warning("⚠️ Vui lòng nhập từ khóa!")
                else:
                    with st.spinner("Đang tìm kiếm..."):
                        tweets = get_tweets(search_query, api_client, tweet_count)
                    if tweets is None:
                        pass  # error already shown
                    elif tweets:
                        st.session_state.tweets = tweets
                        st.success(f"✅ Tìm thấy {len(tweets)} tweets")
                        for i, t in enumerate(tweets, 1):
                            st.markdown(
                                f"**{i}.** {t[:120]}{'...' if len(t) > 120 else ''}"
                            )
                    else:
                        st.info("Không tìm thấy tweets phù hợp.")

            if "tweets" in st.session_state and st.session_state.tweets:
                st.markdown("---")
                selected_model = st.selectbox(
                    "🤖 Chọn mô hình", list(MODEL_FILES.keys()), key="search_model"
                )

                if st.button("🔍 Phân Tích Cảm Xúc", use_container_width=True):
                    model, vectorizer, label_encoder = load_model(
                        MODEL_FILES[selected_model], selected_model
                    )
                    if model is None:
                        st.error("❌ Không thể tải mô hình.")
                    else:
                        pos = neg = neu = 0
                        with st.spinner("Đang phân tích..."):
                            for i, tweet in enumerate(st.session_state.tweets, 1):
                                sentiment = predict_sentiment(
                                    tweet,
                                    model,
                                    vectorizer,
                                    label_encoder,
                                    selected_model,
                                )
                                col1, col2 = st.columns([4, 1])
                                with col1:
                                    st.markdown(
                                        f"**{i}.** {tweet[:80]}{'...' if len(tweet) > 80 else ''}"
                                    )
                                with col2:
                                    if sentiment == "Tích cực":
                                        st.success(sentiment)
                                        pos += 1
                                    else:
                                        st.error(sentiment)
                                        neg += 1
                                st.divider()

                        st.markdown("#### 📈 Tóm Tắt")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("✅ Tích cực", pos)
                        c2.metric("ℹ️ Trung lập", neu)
                        c3.metric("❌ Tiêu cực", neg)


if __name__ == "__main__":
    main()

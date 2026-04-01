# 🐦 Phân Tích Cảm Xúc Twitter

Ứng dụng web phân tích cảm xúc (sentiment analysis) cho văn bản và tweets từ Twitter. Xác định xem nội dung mang thái độ **Tích cực**, **Tiêu cực** bằng các mô hình Machine Learning.

## Tính Năng

- **📝 Phân tích cảm xúc văn bản:** Nhập văn bản bất kỳ, chọn mô hình ML và nhận kết quả phân tích ngay lập tức.
- **🔍 Tìm kiếm & phân tích Twitter:** Tìm tweets theo từ khóa qua Twitter API v2, phân tích cảm xúc hàng loạt với thống kê tóm tắt.
- **🤖 Tự động nhận diện mô hình:** Ứng dụng tự quét thư mục `models/` để lấy danh sách mô hình `.sav` có sẵn — chỉ cần thêm file model mới vào thư mục là dùng được.

## Công Nghệ Sử Dụng

| Thành phần    | Công nghệ                                                   |
| ------------- | ----------------------------------------------------------- |
| Giao diện     | Streamlit                                                   |
| Xử lý dữ liệu | Pandas, NumPy                                               |
| ML/NLP        | Scikit-learn (TF-IDF, Logistic Regression, LinearSVC), NLTK |
| Twitter API   | Tweepy (API v2)                                             |
| Lưu mô hình   | Joblib                                                      |
| Cấu hình      | python-dotenv                                               |

## Cấu Trúc Dự Án

```
├── app.py                 # Ứng dụng chính (Streamlit)
├── prepare_dataset.py     # Script chuẩn bị dữ liệu
├── requirements.txt       # Thư viện phụ thuộc
├── .env                   # Bearer Token cho Twitter API (tự tạo)
├── models/                # Mô hình đã huấn luyện (.sav)
│   ├── TFIDF_LinearSVC_model.sav
│   └── TFIDF_LogisticRegression_model.sav
├── model_TFIDF/           # Script huấn luyện mô hình
│   ├── config.py
│   ├── LinearSVC_TFIDF.py
│   └── LogisticRegression_TFIDF.py
└── kaggle_datasets/       # Dữ liệu huấn luyện (Sentiment140)
```

## Cài Đặt

### 1. Clone repository

```bash
git clone https://github.com/luctananh/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```

### 2. Tạo môi trường ảo

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 4. Cấu hình Twitter API (tùy chọn)

Tạo file `.env` trong thư mục gốc nếu muốn dùng tính năng tìm kiếm Twitter:

```
BEARER_TOKEN=your_twitter_bearer_token_here
```

> Lấy Bearer Token tại [Twitter Developer Portal](https://developer.twitter.com/)

### 5. Chuẩn bị mô hình

Dự án đã có sẵn 2 mô hình trong `models/`. Để huấn luyện thêm:

```bash
cd model_TFIDF
python LogisticRegression_TFIDF.py
python LinearSVC_TFIDF.py
```

Mô hình mới sẽ được lưu vào `models/` và tự động xuất hiện trong ứng dụng.

## Chạy Ứng Dụng

```bash
streamlit run app.py
```

Truy cập `http://localhost:8501` trên trình duyệt.

## Sử Dụng

### Tab "Phân Tích Cảm Xúc"

1. Chọn mô hình từ dropdown
2. Nhập văn bản cần phân tích
3. Nhấn **Phân Tích** → nhận kết quả: ✅ Tích cực / ❌ Tiêu cực

### Tab "Tìm Kiếm Twitter"

1. Nhập từ khóa tìm kiếm (hashtag, keyword...)
2. Chọn số lượng tweets
3. Nhấn **Tìm Kiếm** → xem danh sách tweets
4. Chọn mô hình → nhấn **Phân Tích Cảm Xúc** → xem kết quả từng tweet kèm thống kê tổng hợp

## Mô Hình Có Sẵn

| Mô hình             | Vector hóa | File                                 |
| ------------------- | ---------- | ------------------------------------ |
| Logistic Regression | TF-IDF     | `TFIDF_LogisticRegression_model.sav` |
| LinearSVC (SVM)     | TF-IDF     | `TFIDF_LinearSVC_model.sav`          |

Thêm mô hình mới bằng cách đặt file `.sav` vào thư mục `models/`. Mỗi file `.sav` cần chứa tuple `(model, vectorizer, label_encoder)`.

## Tiền Xử Lý Văn Bản

Pipeline xử lý trước khi đưa vào mô hình:

1. Loại bỏ @mentions, URL, ký tự đặc biệt
2. Chuyển thành chữ thường
3. Tokenization (NLTK)
4. Loại bỏ stopwords
5. Lemmatization

## Bộ Dữ Liệu

Sử dụng **Sentiment140** (~1.6 triệu tweets) từ Kaggle. Nhãn được ánh xạ: `0 → -1 (Tiêu cực)`, `4 → 1 (Tích cực)`.

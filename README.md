# Phân Tích Cảm Xúc Twitter Sử Dụng Machine Learning (Chi Tiết)

## Giới Thiệu Tổng Quan

Dự án này cung cấp một ứng dụng web tương tác để phân tích cảm xúc (sentiment analysis) cho các đoạn văn bản, đặc biệt là các tweet từ mạng xã hội Twitter. Mục tiêu là xác định xem một đoạn văn bản thể hiện thái độ **Tích cực (Positive)**, **Tiêu cực (Negative)**, hay **Trung lập (Neutral)**. Ứng dụng được xây dựng bằng Python, sử dụng thư viện Streamlit cho giao diện người dùng và triển khai nhiều mô hình Machine Learning/NLP khác nhau để thực hiện việc phân loại.

Người dùng có thể nhập trực tiếp văn bản, tìm kiếm các tweet mới nhất trên Twitter theo từ khóa, hoặc đánh giá hiệu suất của các mô hình được cung cấp bằng bộ dữ liệu chuẩn.

## Các Tính Năng Chính

*   **Phân Tích Cảm Xúc Theo Yêu Cầu:**
    *   Nhập hoặc dán bất kỳ đoạn văn bản nào vào ô nhập liệu.
    *   Lựa chọn một trong nhiều mô hình Machine Learning đã được huấn luyện trước.
    *   Nhận kết quả dự đoán cảm xúc (Tích cực, Tiêu cực, Trung lập) cho văn bản đó.
*   **Tìm Kiếm và Phân Tích Tweet Thời Gian Thực:**
    *   Nhập một từ khóa hoặc hashtag để tìm kiếm các tweet gần đây trên Twitter (sử dụng Twitter API v2).
    *   Chỉ định số lượng tweet mong muốn lấy về (có giới hạn do chính sách API).
    *   Xem danh sách các tweet tìm được.
    *   Chọn một mô hình và phân tích cảm xúc cho từng tweet trong kết quả tìm kiếm.
*   **Đánh Giá Hiệu Suất Mô Hình:**
    *   Chọn một mô hình để đánh giá.
    *   Lựa chọn nguồn dữ liệu đánh giá:
        *   **Kaggle:** Tự động tải xuống bộ dữ liệu Sentiment140 nổi tiếng.
        *   **Upload File:** Tải lên file CSV của riêng bạn (phải có cột 'text' và 'sentiment').
    *   Xem các chỉ số đánh giá chi tiết bao gồm **Accuracy**, **Precision**, **Recall**, **F1-score** cho từng lớp cảm xúc, được trình bày trong bảng Classification Report.
*   **Hỗ Trợ Đa Mô Hình & Kỹ Thuật Vector Hóa:** Cung cấp sự linh hoạt và khả năng so sánh giữa các phương pháp khác nhau:
    *   **TF-IDF Based:** Logistic Regression, Random Forest, Naive Bayes, LinearSVC (SVM).
    *   **Bag-of-Words (BoW) Based:** Naive Bayes.
    *   **Word Embedding Based (Word2Vec):** Random Forest, SVM.
    *   **Transformer Embedding Based (TinyBERT):** Random Forest (sử dụng features trích xuất từ TinyBERT).

## Công Nghệ và Thư Viện Sử Dụng

*   **Ngôn ngữ lập trình:** Python 3.8+
*   **Giao diện Web:** Streamlit
*   **Machine Learning:**
    *   Scikit-learn (LogisticRegression, RandomForestClassifier, MultinomialNB, LinearSVC, train_test_split, metrics)
*   **Xử lý Ngôn ngữ Tự nhiên (NLP):**
    *   NLTK (Tokenization, Stopwords Removal, Lemmatization)
    *   Gensim (Word2Vec model training and loading)
    *   Transformers (Hugging Face - BertTokenizer, BertModel for feature extraction)
    *   Scikit-learn (TfidfVectorizer, CountVectorizer)
    *   TextBlob (Sử dụng trong script huấn luyện ban đầu để gán nhãn sơ bộ, không dùng trong app chính)
*   **Xử lý và Thao tác Dữ liệu:**
    *   Pandas (Đọc/ghi CSV, DataFrame manipulation)
    *   NumPy (Tính toán số học, xử lý mảng)
*   **Tương tác API:**
    *   Tweepy (Giao tiếp với Twitter API v2)
*   **Quản lý Môi trường & Cấu hình:**
    *   `python-dotenv` (Tải biến môi trường từ file `.env`)
*   **Tải Dữ liệu:**
    *   KaggleHub (`kagglehub`) (Tải dataset tự động từ Kaggle)
*   **Lưu/Tải Mô hình:**
    *   Joblib (Lưu và tải các đối tượng Python, bao gồm mô hình Scikit-learn và vectorizers)
*   **Xử lý Mất cân bằng Dữ liệu (Trong quá trình huấn luyện):**
    *   Imbalanced-learn (`imblearn` - SMOTE)

## Bộ Dữ Liệu

*   **Huấn luyện & Đánh giá chính:** Bộ dữ liệu **Sentiment140** được sử dụng làm cơ sở.
    *   **Nguồn:** Có thể tải tự động qua `kagglehub` hoặc tải thủ công từ Kaggle ([link](https://www.kaggle.com/datasets/kazanova/sentiment140)).
    *   **Mô tả:** Chứa khoảng 1.6 triệu tweet được gán nhãn thủ công (0 = Tiêu cực, 2 = Trung lập (ít dùng), 4 = Tích cực).
    *   **Ánh xạ nhãn:** Trong dự án này, nhãn được ánh xạ lại: **0 -> -1 (Tiêu cực)**, **4 -> 1 (Tích cực)**. Nhãn trung lập (nếu có) thường bị loại bỏ trong quá trình huấn luyện ban đầu hoặc mô hình tự học cách phân loại dựa trên đặc trưng. Trong ứng dụng `app.py`, các mô hình được huấn luyện để dự đoán 3 lớp (-1, 0, 1) hoặc 2 lớp (-1, 1) và kết quả được chuẩn hóa về Tích cực/Tiêu cực/Trung lập.
*   **Tiền xử lý Dữ liệu (Text Preprocessing):**
    1.  **Làm sạch Tweet (`clean_tweet`):** Loại bỏ tên người dùng (@mentions), các ký tự đặc biệt (không phải chữ và số), và các URL.
    2.  **Chuyển thành chữ thường (`lower()`):** Đồng nhất hóa văn bản.
    3.  **Tokenization (`word_tokenize`):** Tách văn bản thành các từ (tokens).
    4.  **Loại bỏ Stopwords (`stopwords`):** Xóa các từ phổ biến không mang nhiều ý nghĩa (ví dụ: "the", "a", "is").
    5.  **Loại bỏ từ không phải chữ/số (`isalnum()`):** Đảm bảo chỉ giữ lại các token hợp lệ.
    6.  **Lemmatization (`WordNetLemmatizer`):** Đưa các từ về dạng gốc (lemma) để giảm thiểu sự biến đổi hình thái (ví dụ: "running", "ran" -> "run").
*   **Cân bằng Dữ liệu (Trong quá trình huấn luyện):** Kỹ thuật **SMOTE (Synthetic Minority Over-sampling Technique)** được áp dụng trên tập huấn luyện để tạo ra các mẫu tổng hợp cho lớp thiểu số, giúp mô hình học tốt hơn trên các tập dữ liệu mất cân bằng.

## Các Mô Hình Machine Learning và Kỹ Thuật Vector Hóa Sử Dụng

Dự án này triển khai và cho phép so sánh nhiều cách tiếp cận khác nhau để phân tích cảm xúc, kết hợp các thuật toán phân loại với các phương pháp biểu diễn văn bản (vector hóa) khác nhau:

---

**1. TF-IDF (Term Frequency-Inverse Document Frequency) Based Models:**

TF-IDF là một kỹ thuật phổ biến để biến đổi văn bản thành vector số, nhấn mạnh tầm quan trọng của một từ trong một tài liệu cụ thể so với toàn bộ kho tài liệu.

*   **Logistic Regression (TF-IDF)**
    *   **Mô hình:** `LogisticRegression` từ Scikit-learn.
    *   **Vector Hóa:** `TfidfVectorizer` (thường kết hợp cả unigrams và bigrams, ví dụ: `ngram_range=(1, 2)`).
    *   **Mô tả:** Mô hình tuyến tính đơn giản nhưng hiệu quả, thường là một baseline tốt cho các bài toán phân loại văn bản. Nó dự đoán xác suất thuộc về một lớp dựa trên tổ hợp tuyến tính của các feature TF-IDF.
    *   **File Model:** `models/TFIDF_LogisticRegression_model.sav` *(Kiểm tra lại đường dẫn thực tế)*

*   **Random Forest (TF-IDF)**
    *   **Mô hình:** `RandomForestClassifier` từ Scikit-learn.
    *   **Vector Hóa:** `TfidfVectorizer`.
    *   **Mô tả:** Mô hình ensemble dựa trên nhiều cây quyết định. Có khả năng nắm bắt các mối quan hệ phi tuyến tính và thường mạnh mẽ hơn Logistic Regression, nhưng có thể cần nhiều tài nguyên hơn để huấn luyện.
    *   **File Model:** `models/TFIDF_RandomForest_model.sav` *(Kiểm tra lại đường dẫn thực tế)*

*   **Naive Bayes (TF-IDF)**
    *   **Mô hình:** `MultinomialNB` từ Scikit-learn (phù hợp với count-based features như TF-IDF).
    *   **Vector Hóa:** `TfidfVectorizer`.
    *   **Mô tả:** Mô hình xác suất dựa trên định lý Bayes với giả định "ngây thơ" về tính độc lập giữa các feature. Rất nhanh và hiệu quả về mặt tính toán, đặc biệt tốt với dữ liệu văn bản và kích thước lớn.
    *   **File Model:** `models/TFIDF_NaiveBayes_model.sav` *(Kiểm tra lại đường dẫn thực tế)*

*   **Linear Support Vector Classifier (LinearSVC) (TF-IDF)**
    *   **Mô hình:** `LinearSVC` từ Scikit-learn.
    *   **Vector Hóa:** `TfidfVectorizer`.
    *   **Mô tả:** Một dạng của Support Vector Machine (SVM) với kernel tuyến tính. Rất hiệu quả trong không gian đặc trưng cao chiều (như TF-IDF) và thường cho kết quả tốt trong phân loại văn bản. Mục tiêu là tìm siêu phẳng tối ưu phân tách các lớp dữ liệu.
    *   **File Model:** `models/TFIDF_SVM_model.sav` *(Kiểm tra lại đường dẫn thực tế)*

---

**2. Bag-of-Words (BoW) Based Models:**

BoW là một kỹ thuật đơn giản hơn TF-IDF, biểu diễn văn bản bằng cách đếm tần suất xuất hiện của mỗi từ, bỏ qua thứ tự từ.

*   **Naive Bayes (BoW)**
    *   **Mô hình:** `MultinomialNB` từ Scikit-learn.
    *   **Vector Hóa:** `CountVectorizer`.
    *   **Mô tả:** Tương tự như Naive Bayes với TF-IDF, nhưng sử dụng tần suất từ thô thay vì trọng số TF-IDF. Vẫn giữ được tốc độ và hiệu quả tính toán.
    *   **File Model:** `models/NaiveBayes_model.sav` *(Kiểm tra lại đường dẫn thực tế - Tên file này có thể cần làm rõ hơn là BoW hay TF-IDF)*

---

**3. Word Embedding (Word2Vec) Based Models:**

Word2Vec là một kỹ thuật học biểu diễn từ (word embedding) tạo ra các vector dày đặc (dense vectors) cho từ, nắm bắt được ngữ nghĩa và mối quan hệ giữa các từ. Vector của một đoạn văn bản thường được tính bằng cách lấy trung bình các vector của các từ trong đoạn đó.

*   **Random Forest (Word2Vec)**
    *   **Mô hình:** `RandomForestClassifier` từ Scikit-learn.
    *   **Vector Hóa:** Mô hình `Word2Vec` từ Gensim + tính trung bình vector.
    *   **Mô tả:** Sử dụng Random Forest trên các đặc trưng ngữ nghĩa được học bởi Word2Vec. Có thể nắm bắt ý nghĩa sâu sắc hơn so với các phương pháp dựa trên tần suất từ.
    *   **File Model:** `models/RandomForest_models.sav` *(Kiểm tra lại đường dẫn thực tế)*

*   **Support Vector Machine (SVM) (Word2Vec)**
    *   **Mô hình:** `SVC` (có thể là LinearSVC hoặc SVC với kernel khác) từ Scikit-learn.
    *   **Vector Hóa:** Mô hình `Word2Vec` từ Gensim + tính trung bình vector.
    *   **Mô tả:** Kết hợp khả năng phân loại mạnh mẽ của SVM với các biểu diễn ngữ nghĩa từ Word2Vec.
    *   **File Model:** `models/SVM_word2vec_model.sav` *(Kiểm tra lại đường dẫn thực tế)*

---

**4. Transformer Embedding (TinyBERT) Based Models:**

Sử dụng các mô hình Transformer được huấn luyện trước (như BERT) để trích xuất các đặc trưng (embeddings) cho văn bản. Các embedding này thường nắm bắt ngữ cảnh và ngữ nghĩa rất tốt. TinyBERT là một phiên bản nhỏ gọn hơn của BERT.

*   **Random Forest (TinyBERT Features)**
    *   **Mô hình:** `RandomForestClassifier` từ Scikit-learn.
    *   **Vector Hóa:** Trích xuất embedding từ `BertModel` (ví dụ: lấy embedding của token [CLS] hoặc trung bình các token) từ thư viện Transformers, sau đó có thể giảm chiều hoặc xử lý thêm. Trong `app.py`, có bước cắt chiều về 312.
    *   **Mô tả:**ận dụng sức mạnh của các mô hình ngôn ngữ lớn để tạo ra các đặc trưng chất lượng cao, sau đó sử dụng Random Forest để phân loại. Thường cho hiệu suất cao nhưng đòi hỏi nhiều tài nguyên hơn để trích xuất đặc trưng.
    *   **File Model:** `models/RandomForest2_models.sav` *(Kiểm tra lại đường dẫn thực tế)*

---

**Lưu ý:** Tên file mô hình (`.sav`) được liệt kê ở trên là ví dụ dựa trên mã nguồn `app.py` bạn cung cấp. Vui lòng kiểm tra và xác nhận lại đường dẫn và tên file chính xác trong thư mục `models/` và `model_TFIDF/` của dự án bạn. Mỗi file `.sav` thường chứa cả đối tượng mô hình đã huấn luyện, đối tượng vectorizer tương ứng, và đối tượng label encoder.

## Cài Đặt Chi Tiết

**Yêu cầu:**

*   Python (phiên bản 3.8 trở lên được khuyến nghị).
*   Pip (trình quản lý gói Python).
*   Git (để clone repository).

**Các bước thực hiện:**

1.  **Clone Repository:**
    Mở terminal hoặc command prompt và chạy lệnh sau:
    ```bash
    git clone https://github.com/luctananh/twitter-sentiment-analysis.git
    cd twitter-sentiment-analysis
    ```

2.  **Tạo và Kích hoạt Môi trường ảo (Highly Recommended):**
    Việc này giúp cô lập các thư viện của dự án, tránh xung đột với các dự án khác.
    ```bash
    python -m venv venv
    ```
    Kích hoạt môi trường ảo:
    *   Trên **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   Trên **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    Bạn sẽ thấy tiền tố `(venv)` xuất hiện ở đầu dòng lệnh.

3.  **Cài đặt các Thư viện Phụ thuộc:**
    Tạo một file tên là `requirements.txt` trong thư mục gốc của dự án. Nội dung file này nên liệt kê tất cả các thư viện cần thiết. Bạn có thể bắt đầu với danh sách sau (điều chỉnh nếu cần):
    ```txt
    streamlit
    pandas
    numpy
    scikit-learn
    nltk
    gensim
    transformers
    torch # Hoặc torch-cpu nếu không có GPU
    tweepy
    python-dotenv
    kagglehub
    joblib
    textblob # Nếu dùng script huấn luyện có textblob
    imbalanced-learn # Nếu dùng script huấn luyện có SMOTE
    # Thêm các thư viện khác nếu bạn có sử dụng
    ```
    Sau đó, chạy lệnh sau để cài đặt:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Tải Dữ liệu NLTK:**
    Các gói NLTK (`punkt`, `wordnet`, `stopwords`) là cần thiết cho việc tiền xử lý văn bản. Chạy lệnh Python sau trong terminal (khi môi trường ảo đã được kích hoạt):
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    ```
    *(Ứng dụng `app.py` cũng có cơ chế tự động tải nếu thiếu, nhưng chạy trước sẽ đảm bảo hơn).*

5.  **Thiết lập Biến Môi trường cho Twitter API:**
    *   Để sử dụng chức năng "Tìm kiếm và phân tích Twitter", bạn cần có **Bearer Token** từ **Twitter API v2**. Lấy token này từ [Twitter Developer Portal](https://developer.twitter.com/).
    *   Tạo một file mới trong thư mục gốc của dự án tên là `.env`.
    *   Mở file `.env` và thêm dòng sau, thay thế `YOUR_TWITTER_BEARER_TOKEN_HERE` bằng token thực của bạn:
        ```dotenv
        BEARER_TOKEN=YOUR_TWITTER_BEARER_TOKEN_HERE
        ```
    *   **CỰC KỲ QUAN TRỌNG:** Thêm file `.env` vào file `.gitignore` của bạn để đảm bảo bạn không vô tình đưa khóa API bí mật lên GitHub hoặc các hệ thống quản lý phiên bản khác. Mở hoặc tạo file `.gitignore` và thêm dòng:
        ```gitignore
        .env
        venv/
        __pycache__/
        *.pyc
        kaggle_datasets/ # Có thể thêm nếu không muốn commit dataset tải về
        ```

6.  **Chuẩn bị Mô hình Đã Huấn luyện:**
    *   Ứng dụng `app.py` yêu cầu các file mô hình (`.sav`) đã được huấn luyện trước.
    *   **Cách 1: Sử dụng mô hình có sẵn (Nếu được cung cấp):** Đảm bảo rằng các file `.sav` (ví dụ: `TFIDF_LogisticRegression_model.sav`, `RandomForest_models.sav`, ...) nằm đúng trong các thư mục được định nghĩa trong biến `MODEL_FILES` của `app.py` (thường là `models/` hoặc `model_TFIDF/`).
    *   **Cách 2: Tự huấn luyện mô hình:** Nếu repository không kèm theo file `.sav`, bạn cần chạy các script huấn luyện được cung cấp (ví dụ: các file trong thư mục `BCT_Project_3` hoặc file `twitter_sentiment.py`). Ví dụ lệnh chạy (điều chỉnh theo tên script thực tế):
        ```bash
        cd model_TFIDF
        python LogisticRegression_TFIDF.py.py
        # ... tương tự chạy cho tất cả các mô hình bạn muốn sử dụng
        ```
        *Lưu ý:* Quá trình huấn luyện có thể tốn nhiều thời gian và tài nguyên máy tính, đặc biệt là với bộ dữ liệu lớn như Sentiment140. Đảm bảo script huấn luyện lưu các file `.sav` vào đúng thư mục (`models/` hoặc `model_TFIDF/`).

## Cách Chạy Ứng Dụng

1.  **Đảm bảo Môi trường ảo đã được Kích hoạt:** Kiểm tra xem `(venv)` có hiển thị ở đầu dòng lệnh không. Nếu không, chạy lại lệnh kích hoạt ở Bước 2 của phần Cài Đặt.
2.  **Chạy Ứng dụng Streamlit:**
    Từ thư mục gốc của dự án, chạy lệnh sau trong terminal:
    ```bash
    streamlit run app.py
    ```
3.  **Truy cập Ứng dụng:**
    Streamlit sẽ tự động mở một tab mới trong trình duyệt web của bạn, trỏ đến địa chỉ local (thường là `http://localhost:8501`). Nếu không, hãy mở trình duyệt và truy cập URL được hiển thị trong terminal.
4.  **Sử dụng Giao diện:**
    *   **Chọn Chức năng:** Sử dụng các nút radio ở đầu trang để chuyển đổi giữa các tab:
        *   **"Phân tích cảm xúc Twitter":** Nhập văn bản vào ô `Nhập văn bản:`, chọn mô hình từ dropdown `Chọn mô hình phân tích cảm xúc`, sau đó nhấn nút `Phân Tích`. Kết quả sẽ hiển thị bên dưới.
        *   **"Tìm kiếm và phân tích Twitter":** Nhập từ khóa vào ô `Nhập từ khóa tìm kiếm Twitter:`, chọn số lượng tweet mong muốn, nhấn nút `Tìm kiếm`. Sau khi kết quả hiển thị, chọn mô hình và nhấn nút `Phân tích` để xem cảm xúc của từng tweet. (Yêu cầu `.env` đã cấu hình đúng).
        *   **"Đánh giá mô hình":** Chọn mô hình muốn đánh giá. Chọn nguồn dữ liệu (`Kaggle` hoặc `Upload File`). Nếu chọn `Upload File`, hãy tải lên file CSV của bạn. Nhấn nút `Bắt đầu đánh giá`. Chờ quá trình xử lý hoàn tất và xem kết quả Accuracy cùng bảng Classification Report.

## Cấu Trúc Thư Mục Dự Án (Ví dụ)
```plaintext
.
├── kaggle_datasets/ 
├── models/ 
├── model_TFIDF/ 
├── venv/ 
├── .env
├── .gitignore 
├── app.py 
├── cách chạy.txt 
├── README.md 
├── requirements.txt 
├── rustup-init.exe 
└── ... # Các file cấu hình
```
## Đánh Giá Mô Hình (Trong Ứng Dụng)

Tab "Đánh giá mô hình" cho phép bạn kiểm tra hiệu suất của các mô hình trên một tập dữ liệu lớn hơn (cân bằng giữa các lớp tích cực và tiêu cực được lấy từ Sentiment140 hoặc file bạn cung cấp). Quá trình này bao gồm:

1.  Tải dữ liệu (từ Kaggle hoặc file upload).
2.  Ánh xạ lại nhãn sentiment (-1, 1).
3.  Lấy mẫu cân bằng (ví dụ: 160,000 mẫu tích cực, 160,000 mẫu tiêu cực).
4.  Tiền xử lý văn bản (`preprocess_text`).
5.  Vector hóa văn bản bằng vectorizer tương ứng với mô hình đã chọn (TF-IDF, Word2Vec, BERT).
6.  Dự đoán nhãn bằng mô hình đã chọn.
7.  Tính toán và hiển thị **Accuracy** và **Classification Report** (Precision, Recall, F1-score).

## Các Hướng Phát Triển Tiềm Năng

*   **Thêm Mô Hình:** Tích hợp các mô hình hiện đại hơn (ví dụ: các biến thể BERT lớn hơn, XLNet, RoBERTa).
*   **Tinh chỉnh Mô hình (Fine-tuning):** Fine-tune các mô hình Transformer trực tiếp trên dữ liệu Twitter thay vì chỉ dùng features.
*   **Cải thiện Tiền xử lý:** Xử lý tốt hơn tiếng lóng (slang), biểu tượng cảm xúc (emojis), viết tắt.
*   **Phân tích Theo Chủ đề (Topic Modeling):** Kết hợp phân tích cảm xúc với xác định chủ đề đang được thảo luận.
*   **Trực quan hóa Dữ liệu:** Thêm biểu đồ hiển thị phân phối cảm xúc, các từ phổ biến.
*   **Triển khai (Deployment):** Đóng gói ứng dụng bằng Docker và triển khai lên các nền tảng cloud (Heroku, AWS, Google Cloud).
*   **Xử lý Trung lập Tốt hơn:** Cải thiện khả năng phân loại các tweet trung lập.

Hy vọng bản README chi tiết này sẽ giúp người dùng hiểu rõ hơn về dự án và cách sử dụng nó!
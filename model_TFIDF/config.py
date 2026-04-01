# ============================================================
#  config.py — Cấu hình tham số cho toàn bộ pipeline
#  Chỉnh sửa tại đây, không cần động vào file chính
# ============================================================

# ── Đường dẫn model ─────────────────────────────────────────
MODEL_PATH = "../models/TFIDF_LinearSVC_model.sav"

# ── Dữ liệu ─────────────────────────────────────────────────
TEST_SIZE = 0.2  # tỉ lệ tập kiểm thử
RANDOM_STATE = 2  # seed để tái tạo kết quả

# ── Tiền xử lý ──────────────────────────────────────────────
USE_LEMMATIZATION = False  # True: bật lemmatization (chậm hơn, ít lợi với tweet)

# ── TF-IDF ──────────────────────────────────────────────────
TFIDF_NGRAM_RANGE = (1, 2)  # (1,2): unigrams + bigrams
TFIDF_MIN_DF = 5  # loại token xuất hiện < N docs (lọc noise)
TFIDF_MAX_DF = 0.95  # loại token xuất hiện > 95% docs (vô nghĩa)
TFIDF_SUBLINEAR = True  # log-scale TF

# ── LinearSVC ───────────────────────────────────────────────
SVC_C = 0.1  # regularization: nhỏ → tổng quát, lớn → fit sát train
SVC_MAX_ITER = 5000  # số vòng lặp tối đa để hội tụ


# ----------------------------LogisticRegression--------------------------------#

# ── Đường dẫn model ─────────────────────────────────────────
MODEL_PATH = "../models/TFIDF_LogisticRegression_model.sav"

# ── Dữ liệu ─────────────────────────────────────────────────
TEST_SIZE = 0.2  # tỉ lệ tập kiểm thử
RANDOM_STATE = 2  # seed để tái tạo kết quả

# ── Tiền xử lý ──────────────────────────────────────────────
USE_LEMMATIZATION = True  # True: bật lemmatization (chậm hơn, ít lợi với tweet)

# ── TF-IDF ──────────────────────────────────────────────────
TFIDF_NGRAM_RANGE = (1, 2)  # (1,2): unigrams + bigrams
TFIDF_MIN_DF = 5  # loại token xuất hiện < N docs (lọc noise)
TFIDF_MAX_DF = 0.95  # loại token xuất hiện > 95% docs (vô nghĩa)
TFIDF_SUBLINEAR = True  # log-scale TF

# ── Logistic Regression ──────────────────────────────────────
LR_C = 1.0  # regularization: nhỏ → tổng quát, lớn → fit sát train
LR_MAX_ITER = 2000  # số vòng lặp tối đa để hội tụ

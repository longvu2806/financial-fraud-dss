"""
hieu_base.py
============
Nhiệm vụ Ngày 1 - Hiếu (Baseline Modeler)
Mô hình 1: Logistic Regression (class_weight='balanced' + threshold tuning)
Mô hình 2: Decision Tree (giới hạn depth + class_weight='balanced' + threshold tuning)
Mục tiêu: Đẩy precision & recall của lớp Fraud lên mức cân bằng (~0.4 mỗi chỉ số),
          thay vì baseline "tệ" mặc định hoàn toàn.

CẬP NHẬT QUAN TRỌNG (đồng bộ với pipeline fill_data_to_gold_proxy.py):
  Đọc trực tiếp 2 file đã chia sẵn ở tầng Gold:
       - data/3_gold/train_gold_downsampled.parquet  (tập train, đã downsample 1:123)
       - data/3_gold/test_gold_original.parquet      (tập test, giữ nguyên phân phối thực, KHÔNG downsample)
  Đây CHÍNH XÁC là 2 file mà vu_advanced.py (model 6/7/8) cũng đọc,
  đảm bảo Model 1, 2 của Hiếu được test trên cùng một tập test_gold_original
  với các model khác -> so sánh công bằng trong bảng tổng kết cuối ngày.

CẬP NHẬT MỚI (tối ưu precision/recall fraud ~0.4):
  1. Logistic Regression: thêm class_weight='balanced' để model không bị thiên
     lệch về lớp Legit chiếm đa số.
  2. Decision Tree: giới hạn max_depth + min_samples_leaf + class_weight='balanced'
     để chống overfit nặng (bản cũ depth=37, 5694 lá -> precision fraud chỉ 0.02).
  3. THRESHOLD TUNING: Vì điểm cân bằng precision=recall=0.4 không nhất thiết
     rơi đúng vào ngưỡng mặc định 0.5, ta quét ngưỡng trên các điểm xác suất
     của TẬP TRAIN (không đụng vào test, tránh leakage) để tìm ngưỡng cho ra
     precision & recall gần 0.4 nhất, rồi áp ngưỡng đó để dự đoán trên test.

Xuất kết quả: results/results_hieu.csv (qua base_model.save_metrics_to_csv)
"""

import sys
import time
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
from base_model import save_metrics_to_csv

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Đường dẫn — khớp với GOLD_DIR trong fill_data_to_gold_proxy.py
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR     = PROJECT_ROOT / "data" / "3_gold"
TRAIN_FILE   = GOLD_DIR / "train_gold_downsampled.parquet"
TEST_FILE    = GOLD_DIR / "test_gold_original.parquet"

LABEL_COL = "is_fraud"

# Mục tiêu cân bằng precision/recall của lớp Fraud (đặt 0.40 theo yêu cầu)
TARGET_SCORE = 0.40

# Các cột không phải feature số/flag — loại trước khi đưa vào model.
# Khớp với cols_to_drop trong vu_advanced.py (model 6/7/8) để đảm bảo
# 2 bên dùng đúng cùng một bộ feature.
ID_TEXT_COLS_TO_DROP = [
    "id", "date", "errors", "merchant_city", "merchant_state",
    "zip", "acct_open_date",
]


def _drop_non_feature_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Loại các cột ID/text rác, khớp với cách Vũ xử lý trước khi train."""
    cols_to_drop = [c for c in df.columns if c.endswith("_id") or c in ID_TEXT_COLS_TO_DROP]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=cols_to_drop, errors="ignore")


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode cột object/category bằng .cat.codes — khớp với vu_advanced.py."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes
    return df


# ---------------------------------------------------------------------------
# 1. Load dữ liệu từ 2 file Gold đã chia sẵn (train downsample / test gốc)
# ---------------------------------------------------------------------------
def load_data():
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {TRAIN_FILE} hoặc {TEST_FILE}.\n"
            f"Hãy chạy notebooks/fill_data_to_gold_proxy.py trước để sinh ra "
            f"2 file train/test ở tầng Gold."
        )

    print(f"[INFO] Đọc tập TRAIN (đã downsample 1:123): {TRAIN_FILE}")
    start = time.time()
    df_train = pd.read_parquet(TRAIN_FILE)
    print(f"[INFO] Đọc tập TEST (giữ nguyên phân phối thực): {TEST_FILE}")
    df_test = pd.read_parquet(TEST_FILE)
    print(f"[INFO] Load xong trong {time.time()-start:.1f}s")
    print(f"[INFO] Train: {len(df_train):,} dòng | Test: {len(df_test):,} dòng")

    for name, df in [("train", df_train), ("test", df_test)]:
        if LABEL_COL not in df.columns:
            raise ValueError(f"Thiếu cột {LABEL_COL} trong tập {name}.")

    print(f"[INFO] Tỷ lệ fraud train (sau downsample): {df_train[LABEL_COL].mean():.4%}")
    print(f"[INFO] Tỷ lệ fraud test (phân phối thực) : {df_test[LABEL_COL].mean():.4%}")

    # --- Loại cột ID/text rác + encode category, khớp với vu_advanced.py ---
    df_train = _drop_non_feature_cols(df_train)
    df_test = _drop_non_feature_cols(df_test)
    df_train = _encode_categoricals(df_train)
    df_test = _encode_categoricals(df_test)

    X_train = df_train.drop(columns=[LABEL_COL])
    y_train = df_train[LABEL_COL].astype(np.int8)
    X_test = df_test.drop(columns=[LABEL_COL])
    y_test = df_test[LABEL_COL].astype(np.int8)

    # Đảm bảo 2 tập có đúng cùng bộ cột & thứ tự (đề phòng lệch do encode category khác nhau)
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    missing_in_test = set(X_train.columns) - set(common_cols)
    missing_in_train = set(X_test.columns) - set(common_cols)
    if missing_in_test:
        print(f"[WARN] Cột có ở train nhưng không có ở test (bỏ qua): {missing_in_test}")
    if missing_in_train:
        print(f"[WARN] Cột có ở test nhưng không có ở train (bỏ qua): {missing_in_train}")

    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    print(f"[INFO] Sử dụng {len(common_cols)} features chung cho 2 tập.")

    # --- Điền NaN bằng median của tập train (tránh leakage) ---
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    # Phòng trường hợp còn cột không phải numeric chưa fillna được
    X_train = X_train.fillna(-999)
    X_test = X_test.fillna(-999)

    return X_train, X_test, y_train, y_test, common_cols


# ---------------------------------------------------------------------------
# 1b. Tìm ngưỡng (threshold) cho precision & recall của Fraud gần TARGET nhất
# ---------------------------------------------------------------------------
def find_threshold_for_target(y_true, y_prob, target=TARGET_SCORE, n_steps=999):
    """
    Quét threshold từ ~0 đến ~1, với mỗi threshold tính precision & recall
    của lớp Fraud (label=1). Chọn threshold làm cho cả 2 chỉ số gần `target`
    nhất theo nghĩa: minimize max(|precision-target|, |recall-target|).

    Hàm này nên được gọi trên tập TRAIN (out-of-fold hoặc chính train),
    KHÔNG gọi trên test, để tránh leakage / overfit ngưỡng vào test.
    """
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_t = 0.5
    best_score = np.inf
    best_p, best_r = 0.0, 0.0

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        # Bỏ qua threshold làm model dự đoán toàn 0 hoặc toàn 1 (vô nghĩa)
        if y_pred_t.sum() == 0 or y_pred_t.sum() == len(y_pred_t):
            continue
        p = precision_score(y_true, y_pred_t, zero_division=0)
        r = recall_score(y_true, y_pred_t, zero_division=0)
        score = max(abs(p - target), abs(r - target))
        if score < best_score:
            best_score = score
            best_t = t
            best_p, best_r = p, r

    print(f"[THRESHOLD] Ngưỡng chọn (trên TRAIN): {best_t:.3f} "
          f"-> Precision≈{best_p:.3f} | Recall≈{best_r:.3f} (mục tiêu {target})")
    return best_t


# ---------------------------------------------------------------------------
# 2. Mô hình 1 — Logistic Regression (class_weight='balanced' + threshold tuning)
# ---------------------------------------------------------------------------
def train_logistic_regression(X_train, X_test, y_train, y_test):
    print("\n" + "="*55)
    print("[MODEL 1] Logistic Regression (balanced + threshold) - đang train...")
    print("="*55)
    start = time.time()

    # Chuẩn hoá (bắt buộc cho Logistic Regression)
    scaler  = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # class_weight='balanced' giúp model không bị thiên lệch về lớp đa số (Legit)
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train_scaled, y_train)

    # --- Lấy xác suất trên TRAIN để chọn threshold (không đụng test) ---
    y_prob_train = model.predict_proba(X_train_scaled)[:, 1]
    best_threshold = find_threshold_for_target(y_train, y_prob_train, target=TARGET_SCORE)

    # --- Áp threshold đã chọn lên TEST ---
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= best_threshold).astype(int)

    print(f"[INFO] Thời gian train: {time.time()-start:.2f}s")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    save_metrics_to_csv("Hieu_LogisticRegression", y_test, y_pred, y_prob)


# ---------------------------------------------------------------------------
# 3. Mô hình 2 — Decision Tree (giới hạn depth + balanced + threshold tuning)
# ---------------------------------------------------------------------------
def train_decision_tree(X_train, X_test, y_train, y_test):
    print("\n" + "="*55)
    print("[MODEL 2] Decision Tree (giới hạn depth + balanced + threshold) - đang train...")
    print("="*55)
    start = time.time()

    # Giới hạn độ phức tạp để chống overfit nặng (bản cũ: depth=37, 5694 lá).
    # max_depth & min_samples_leaf là 2 tham số quan trọng nhất để kiểm soát
    # việc cây "học vẹt" theo nhiễu của tập train downsample 1:123.
    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=6,
        min_samples_leaf=50,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    # --- Lấy xác suất trên TRAIN để chọn threshold ---
    y_prob_train = model.predict_proba(X_train)[:, 1]
    best_threshold = find_threshold_for_target(y_train, y_prob_train, target=TARGET_SCORE)

    # --- Áp threshold đã chọn lên TEST ---
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_threshold).astype(int)

    print(f"[INFO] Thời gian train: {time.time()-start:.2f}s")
    print(f"[INFO] Tree depth thực tế: {model.get_depth()} | Số lá: {model.get_n_leaves():,}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    save_metrics_to_csv("Hieu_DecisionTree", y_test, y_pred, y_prob)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 55)
    print("  HIẾU - BASELINE MODELS (Ngày 1, đã tối ưu precision/recall Fraud ~0.4)")
    print("  Đọc train_gold_downsampled.parquet (train, downsample 1:123)")
    print("  Đọc test_gold_original.parquet (test, phân phối thực, khớp Vũ)")
    print("=" * 55)

    X_train, X_test, y_train, y_test, feature_cols = load_data()

    train_logistic_regression(X_train, X_test, y_train, y_test)
    train_decision_tree(X_train, X_test, y_train, y_test)

    print("\n[✓] Hoàn thành Ngày 1 - Hiếu!")
    print(f"[✓] Kết quả đã lưu vào results/results_hieu.csv")


if __name__ == "__main__":
    main()
"""
hieu_base.py
============
Nhiệm vụ Ngày 1 - Hiếu (Baseline Modeler)
Mô hình 1: Logistic Regression (Mặc định) - KHÔNG tune
Mô hình 2: Decision Tree (Mặc định)       - KHÔNG tune
Mục tiêu: Tạo baseline tệ để làm nổi bật các mô hình nâng cao.

Thay đổi so với phiên bản cũ:
  - Đọc từ tầng GOLD (featured_transactions.parquet) thay vì Silver
  - Dùng 35 features đầy đủ từ fill_data_to_gold.py
  - Chia train/test theo thời gian (Temporal Split) thay vì ngẫu nhiên
  - In thông tin mốc thời gian để kiểm tra

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
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
from base_model import save_metrics_to_csv

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Đường dẫn
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR     = PROJECT_ROOT / "data" / "3_gold"
GOLD_FILE    = GOLD_DIR / "featured_transactions.parquet"

# ---------------------------------------------------------------------------
# Feature columns — lấy từ fill_data_to_gold.py
# ---------------------------------------------------------------------------

# Số lượng (numeric)
NUMERIC_COLS = [
    "amount",
    "abs_amount",
    "net_amount",
    "log_abs_amount",
    "time_since_last_txn",
    "txn_count_1h",
    "amount_mean_7d",
    "amount_std_7d",
    "amount_zscore",
    "merchant_error_count_24h",
    "distance_km",
    "travel_speed_kmh",
    "days_since_open",
    "credit_limit",
    "user_age",
    "yearly_income",
    "total_debt",
    "credit_score",
]

# Cờ nhị phân (binary flags)
FLAG_COLS = [
    "flag_is_refund",
    "flag_rapid_sequence",
    "flag_burst_1h",
    "flag_amount_spike",
    "flag_near_credit_limit",
    "flag_round_amount",
    "flag_high_mcc_risk",
    "flag_dark_web_card",
    "flag_chip_bypass",
    "flag_new_card",
    "flag_pin_stale",
    "flag_state_hop",
    "flag_international",
    "flag_night_txn",
    "flag_far_from_home",
    "flag_impossible_travel",
    "flag_hot_merchant",
    "flag_debt_pressure",
    "is_new_merchant",
    "is_weekend",
]

LABEL_COL = "is_fraud"
DATE_COL  = "date"

# ---------------------------------------------------------------------------
# 1. Load dữ liệu từ Gold & Temporal Split
# ---------------------------------------------------------------------------
def load_data():
    if not GOLD_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy Gold file: {GOLD_FILE}\n"
            f"Hãy chạy fill_data_to_gold.py trước."
        )

    print(f"[INFO] Đọc Gold file: {GOLD_FILE}")
    start = time.time()
    df = pd.read_parquet(GOLD_FILE)
    print(f"[INFO] Tổng dòng: {len(df):,} | Load: {time.time()-start:.1f}s")

    # --- Kiểm tra cột bắt buộc ---
    missing_cols = [c for c in [DATE_COL, LABEL_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu cột bắt buộc trong Gold file: {missing_cols}")

    # --- Chuyển đổi date & sắp xếp ---
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    print(f"[INFO] Khoảng thời gian: {df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}")
    print(f"[INFO] Tỷ lệ fraud: {df[LABEL_COL].mean():.4%}")

    # --- Chọn features có trong file ---
    all_feature_cols = NUMERIC_COLS + FLAG_COLS
    available_cols = [c for c in all_feature_cols if c in df.columns]
    missing_features = [c for c in all_feature_cols if c not in df.columns]
    if missing_features:
        print(f"[WARN] Thiếu {len(missing_features)} features (sẽ bỏ qua): {missing_features}")
    print(f"[INFO] Sử dụng {len(available_cols)} features: {available_cols}")

    # --- Temporal Split 80/20 ---
    cutoff_date = df[DATE_COL].quantile(0.8)  # tự động lấy mốc 80%
    # quantile trên datetime trả về timestamp
    if not isinstance(cutoff_date, pd.Timestamp):
        cutoff_date = pd.Timestamp(cutoff_date)

    train_df = df[df[DATE_COL] <= cutoff_date]
    test_df  = df[df[DATE_COL] >  cutoff_date]

    print(f"\n[INFO] ── Temporal Split ──")
    print(f"        Mốc cutoff  : {cutoff_date.date()}")
    print(f"        Train       : {train_df[DATE_COL].min().date()} → {train_df[DATE_COL].max().date()} | {len(train_df):,} dòng")
    print(f"        Test        : {test_df[DATE_COL].min().date()} → {test_df[DATE_COL].max().date()} | {len(test_df):,} dòng")
    print(f"        Train fraud : {train_df[LABEL_COL].mean():.4%}")
    print(f"        Test fraud  : {test_df[LABEL_COL].mean():.4%}")

    X_train = train_df[available_cols].copy()
    y_train = train_df[LABEL_COL].copy()
    X_test  = test_df[available_cols].copy()
    y_test  = test_df[LABEL_COL].copy()

    # --- Điền NaN bằng median của tập train (tránh leakage) ---
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test  = X_test.fillna(medians)

    return X_train, X_test, y_train, y_test, available_cols


# ---------------------------------------------------------------------------
# 2. Mô hình 1 — Logistic Regression (có scale vì LR cần chuẩn hoá)
# ---------------------------------------------------------------------------
def train_logistic_regression(X_train, X_test, y_train, y_test):
    print("\n" + "="*55)
    print("[MODEL 1] Logistic Regression (Mặc định) - đang train...")
    print("="*55)
    start = time.time()

    # Chuẩn hoá (bắt buộc cho Logistic Regression)
    scaler  = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Hoàn toàn mặc định: không class_weight, không tune
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print(f"[INFO] Thời gian train: {time.time()-start:.2f}s")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    save_metrics_to_csv("Hieu_LogisticRegression", y_test, y_pred, y_prob)


# ---------------------------------------------------------------------------
# 3. Mô hình 2 — Decision Tree (Mặc định, không giới hạn depth → overfit)
# ---------------------------------------------------------------------------
def train_decision_tree(X_train, X_test, y_train, y_test):
    print("\n" + "="*55)
    print("[MODEL 2] Decision Tree (Mặc định) - đang train...")
    print("="*55)
    start = time.time()

    # Hoàn toàn mặc định: không giới hạn depth → overfit nặng trên train
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"[INFO] Thời gian train: {time.time()-start:.2f}s")
    print(f"[INFO] Tree depth thực tế: {model.get_depth()} | Số lá: {model.get_n_leaves():,}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    save_metrics_to_csv("Hieu_DecisionTree", y_test, y_pred, y_prob)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 55)
    print("  HIẾU - BASELINE MODELS (Ngày 1)")
    print("  Đọc từ data/3_gold/featured_transactions.parquet")
    print("  Chia dữ liệu: TEMPORAL SPLIT (80% cũ / 20% mới)")
    print("=" * 55)

    X_train, X_test, y_train, y_test, feature_cols = load_data()

    train_logistic_regression(X_train, X_test, y_train, y_test)
    train_decision_tree(X_train, X_test, y_train, y_test)

    print("\n[✓] Hoàn thành Ngày 1 - Hiếu!")
    print(f"[✓] Kết quả đã lưu vào results/results_hieu.csv")


if __name__ == "__main__":
    main()
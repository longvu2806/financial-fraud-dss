"""
hieu_base.py
============
Nhiệm vụ Ngày 1 - Hiếu (Baseline Modeler)
Mô hình 1: Logistic Regression (Mặc định) - KHÔNG tune
Mô hình 2: Decision Tree (Mặc định)       - KHÔNG tune
Mục tiêu: Tạo baseline tệ để làm nổi bật các mô hình nâng cao.
Xuất kết quả: results/results_hieu.csv (qua base_model.save_metrics_to_csv)
"""

import sys
import time
import warnings
import glob
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.append(str(Path(__file__).resolve().parent))
from base_model import save_metrics_to_csv

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Đường dẫn
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SILVER_DIR   = PROJECT_ROOT / "data" / "2_silver"

FEATURE_COLS = [
    "amount", "merchant_id", "card_id", "user_id",
    "tx_hour", "tx_day_of_week", "is_night_tx",
]
LABEL_COL = "is_fraud"


# ---------------------------------------------------------------------------
# 1. Load toàn bộ 134 parts
# ---------------------------------------------------------------------------
def load_data():
    parts = sorted(glob.glob(str(SILVER_DIR / "transactions_cleaned_part_*.parquet")))
    if not parts:
        raise FileNotFoundError(f"Không tìm thấy parquet files trong {SILVER_DIR}")

    print(f"[INFO] Load {len(parts)} parts từ Silver...")
    start = time.time()

    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    print(f"[INFO] Tổng dòng: {len(df):,} | Load: {time.time()-start:.1f}s")
    print(f"[INFO] Tỷ lệ fraud: {df[LABEL_COL].mean():.4%}")

    X = df[FEATURE_COLS].copy()
    y = df[LABEL_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 2. Mô hình 1 — Logistic Regression (Mặc định, KHÔNG tune)
# ---------------------------------------------------------------------------
def train_logistic_regression(X_train, X_test, y_train, y_test):
    print("\n[MODEL 1] Logistic Regression (Mặc định) - đang train...")
    start = time.time()

    # Hoàn toàn mặc định: không class_weight, không tune
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"[INFO] Thời gian train: {time.time()-start:.2f}s")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    save_metrics_to_csv("Hieu_LogisticRegression", y_test, y_pred, y_prob)


# ---------------------------------------------------------------------------
# 3. Mô hình 2 — Decision Tree (Mặc định, KHÔNG tune)
# ---------------------------------------------------------------------------
def train_decision_tree(X_train, X_test, y_train, y_test):
    print("\n[MODEL 2] Decision Tree (Mặc định) - đang train...")
    start = time.time()

    # Hoàn toàn mặc định: không giới hạn depth -> overfit nặng trên train
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"[INFO] Thời gian train: {time.time()-start:.2f}s")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    save_metrics_to_csv("Hieu_DecisionTree", y_test, y_pred, y_prob)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 55)
    print("  HIẾU - BASELINE MODELS (Ngày 1)")
    print("  Load toàn bộ 134 parts từ data/2_silver/")
    print("=" * 55)

    X_train, X_test, y_train, y_test = load_data()
    train_logistic_regression(X_train, X_test, y_train, y_test)
    train_decision_tree(X_train, X_test, y_train, y_test)

    print("\n[✓] Hoàn thành Ngày 1 - Hiếu!")


if __name__ == "__main__":
    main()
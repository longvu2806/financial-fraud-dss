"""
regenerate_test_stream.py
---------------------------
Sửa lỗi: test_stream.csv cũ (do Cường tạo) thiếu 10 cột feature so với
dữ liệu thật Vũ dùng để train model (test_gold_original.parquet ở tầng
3_gold). Script này tạo LẠI test_stream.csv lấy đúng từ nguồn gold đó,
đảm bảo khớp 100% cột feature mà model xgboost_v8.pkl cần.

Cách chạy (từ thư mục gốc project, financial-fraud-dss/):
    python regenerate_test_stream.py

Nếu muốn chỉ định số dòng khác 50 hoặc seed khác:
    python regenerate_test_stream.py --n 50 --seed 42
"""

import os
import argparse
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GOLD_TEST_PATH = os.path.join(BASE_DIR, "data", "3_gold", "test_gold_original.parquet")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "test_stream.csv")


def main(n_rows: int, seed: int):
    if not os.path.exists(GOLD_TEST_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy {GOLD_TEST_PATH}. "
            "Kiểm tra lại đường dẫn data/3_gold/test_gold_original.parquet có tồn tại không "
            "(đây là file Vũ dùng để train/test model)."
        )

    print(f"⏳ Đang nạp dữ liệu gold từ: {GOLD_TEST_PATH}")
    df_test = pd.read_parquet(GOLD_TEST_PATH)
    print(f"✅ Tổng số dòng trong tập test gold: {len(df_test):,}")
    print(f"✅ Tổng số cột: {len(df_test.columns)}")

    if len(df_test) < n_rows:
        raise ValueError(f"Tập test chỉ có {len(df_test)} dòng, không đủ {n_rows} dòng yêu cầu.")

    sample_df = df_test.sample(n=n_rows, random_state=seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    sample_df.to_csv(OUTPUT_PATH, index=False)

    print(f"📦 Đã lưu {n_rows} dòng mẫu mới vào: {OUTPUT_PATH}")
    print(f"📋 Các cột: {list(sample_df.columns)}")

    if "is_fraud" in sample_df.columns:
        fraud_count = int(sample_df["is_fraud"].sum())
        print(f"⚠️  Số dòng gian lận thật trong mẫu: {fraud_count}/{n_rows}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Số dòng lấy mẫu (mặc định 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (mặc định 42)")
    args = parser.parse_args()

    main(n_rows=args.n, seed=args.seed)
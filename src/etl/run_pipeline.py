# run_pipeline.py
import os
import sys
import logging
import pandas as pd
import shutil

# Đảm bảo Python định vị được các module trong thư mục src/
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir)) if "src" in current_dir else current_dir
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.etl.extract import extract_csv, extract_json
from src.etl.transform import (
    clean_users_data, 
    clean_cards_data, 
    clean_mcc_data, 
    clean_labels_data, 
    clean_transaction_chunk
)
from src.etl.load import PostgresLoader

# Cấu hình đường dẫn thư mục Silver để lưu trữ file backup vật lý
SILVER_DIR = "data/2_silver"

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

def main():
    logging.info("🚀 BẮT ĐẦU KÍCH HOẠT HỆ THỐNG PIPELINE ETL CAIXABANK (GỘP NHÃN FACT)...")
    
    # 🧹 TỰ ĐỘNG DỌN RÁC THƯ MỤC SILVER CŨ ĐỂ TRÁNH FILE THỪA LẦN CHẠY TRƯỚC
    if os.path.exists(SILVER_DIR):
        logging.info(f"🧹 Phát hiện folder {SILVER_DIR} cũ, đang dọn sạch file rác...")
        shutil.rmtree(SILVER_DIR)
    os.makedirs(SILVER_DIR, exist_ok=True)
    
    # ----------------------------------------------------------------
    # KHỞI TẠO HẠ TẦNG DATABASE
    # ----------------------------------------------------------------
    loader = PostgresLoader()
    loader.execute_schema()  # Tự động đọc schema.sql để tạo bảng sạch trên Postgres
    
    # ----------------------------------------------------------------
    # PHASE 1: TRÍCH XUẤT, LÀM SẠCH VÀ CHUẨN BỊ BẢNG TĨNH
    # ----------------------------------------------------------------
    logging.info("⏳ PHASE 1: Đang dọn rác và nạp các bảng hồ sơ tĩnh...")
    
    # Chỉ truyền tên file thô, hàm extract sẽ tự tìm trong data/1_bronze/
    df_users_raw = extract_csv("users_data.csv")
    df_cards_raw = extract_csv("cards_data.csv")
    mcc_dict = extract_json("mcc_codes.json")
    fraud_dict = extract_json("train_fraud_labels.json")
    
    fraud_dict = fraud_dict.get("target", fraud_dict) #unwrap
    # Gọi các hàm làm sạch của đồng đội
    df_users_clean = clean_users_data(df_users_raw)
    df_cards_clean = clean_cards_data(df_cards_raw)
    df_mcc_clean = clean_mcc_data(mcc_dict)
    df_labels_clean = clean_labels_data(fraud_dict)
    
    # 💾 Sao lưu vật lý các bảng tĩnh sang tầng Silver dưới dạng Parquet
    df_users_clean.to_parquet(os.path.join(SILVER_DIR, "users_cleaned.parquet"), index=False)
    df_cards_clean.to_parquet(os.path.join(SILVER_DIR, "cards_cleaned.parquet"), index=False)
    df_mcc_clean.to_parquet(os.path.join(SILVER_DIR, "mcc_cleaned.parquet"), index=False)
    if df_labels_clean is not None:
        df_labels_clean.to_parquet(os.path.join(SILVER_DIR, "labels_cleaned.parquet"), index=False)
    
    # Nạp Bulk Load các bảng Dimension vào PostgreSQL
    loader.fast_load_dataframe(df_users_clean, "dim_users")
    loader.fast_load_dataframe(df_cards_clean, "dim_cards")
    loader.fast_load_dataframe(df_mcc_clean, "dim_mcc")
    
    # Giải phóng RAM ngay lập tức cho các bảng tĩnh
    del df_users_raw, df_users_clean, df_cards_raw, df_cards_clean, mcc_dict
    
    # ----------------------------------------------------------------
    # PHASE 2: STREAM CHUNK, MERGE NHÃN AI, SAO LƯU & BULK LOAD BẢNG FACT
    # ----------------------------------------------------------------
    logging.info("⏳ PHASE 2: Đang stream luồng giao dịch khổng lồ, gán nhãn và sao lưu...")
    
    chunk_size = 100000
    transaction_chunks = extract_csv("transactions_data.csv", chunksize=chunk_size)
    
    # 🎯 DANH SÁCH CỘT CHUẨN: Khớp 100% thứ tự với bảng fact_transactions mới trong schema.sql
    POSTGRES_TRANSACTION_COLUMNS = [
        'transaction_id', 'date', 'user_id', 'card_id', 'amount', 
        'use_chip', 'merchant_id', 'merchant_city', 'merchant_state', 
        'zip', 'mcc', 'errors', 'tx_hour', 'tx_day_of_week', 'is_night_tx', 'is_fraud'
    ]
    
    chunk_count = 0
    for raw_chunk in transaction_chunks:
        chunk_count += 1
        logging.info(f"📦 Đang xử lý khối giao dịch thứ {chunk_count}...")
        
        # Phòng thủ lỗi tịnh tiến index của cơ chế cắt khối pandas
        raw_chunk = raw_chunk.reset_index(drop=True)
        
        # 1. 🎯 TRUYỀN BIẾN CHUẨN: Gọi hàm dọn rác và TỰ GÁN NHÃN ngay trong tầng Transform
        # Đưa từ điển 'fraud_dict' bốc từ Phase 1 vào để ánh xạ trực tiếp
        cleaned_chunk = clean_transaction_chunk(raw_chunk, fraud_dict=fraud_dict)
            
        # 💾 2. Sao lưu vật lý khối giao dịch hoàn hảo vào tầng 2_silver dưới dạng Parquet
        output_path = os.path.join(SILVER_DIR, f"transactions_cleaned_part_{chunk_count}.parquet")
        cleaned_chunk.to_parquet(output_path, index=False)
        
        # 🎯 CHỐT CHẶN CỘT THỪA: Chỉ lọc đúng những cột có trong cấu trúc PostgreSQL
        load_chunk = cleaned_chunk[POSTGRES_TRANSACTION_COLUMNS]
        
        # 3. Đổ siêu tốc dữ liệu hoàn hảo vào bảng fact_transactions trong PostgreSQL bằng lệnh COPY
        loader.fast_load_dataframe(load_chunk, "fact_transactions")
        
    # 🧼 Giải phóng RAM sau khi đã stream xong toàn bộ 13 triệu dòng
    if 'fraud_dict' in locals() or 'fraud_dict' in globals():
        del fraud_dict
        
    loader.close()
    logging.info("🎉 THÀNH CÔNG RỰC RỠ! Dữ liệu đã được gán nhãn qua Dictionary và nạp đầy vào Postgres!")

if __name__ == "__main__":
    main()
# run_pipeline.py
import os
import sys
import logging
import pandas as pd

# Đảm bảo Python định vị được các module trong thư mục src/
# 🎯 ĐOẠN SỬA CHÍ MẠNG: Ép Python tìm ngược về thư mục gốc dự án
current_dir = os.path.dirname(os.path.abspath(__file__))
# Nếu file run_pipeline.py nằm ở thư mục gốc, dùng: os.path.abspath(current_dir)
# Nếu file run_pipeline.py nằm trong src/etl/, dùng: os.path.dirname(os.path.dirname(current_dir))
root_dir = os.path.dirname(os.path.dirname(current_dir)) 

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
os.makedirs(SILVER_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

def main():
    logging.info("🚀 BẮT ĐẦU KÍCH HOẠT HỆ THỐNG PIPELINE ETL CAIXABANK (CÓ SAO LƯU SILVER)...")
    
    # ----------------------------------------------------------------
    # KHỞI TẠO HẠ TẦNG DATABASE
    # ----------------------------------------------------------------
    loader = PostgresLoader()
    loader.execute_schema()
    
    # ----------------------------------------------------------------
    # PHASE 1: TRÍCH XUẤT, LÀM SẠCH, BACKUP & NẠP CÁC BẢNG TĨNH (DIMENSIONS)
    # ----------------------------------------------------------------
    logging.info("⏳ PHASE 1: Đang dọn rác, sao lưu và nạp các bảng hồ sơ tĩnh...")
    
    df_users_raw = extract_csv("users_data.csv")
    df_cards_raw = extract_csv("cards_data.csv")
    mcc_dict = extract_json("mcc_codes.json")
    fraud_dict = extract_json("train_fraud_labels.json")
    
    df_users_clean = clean_users_data(df_users_raw)
    df_cards_clean = clean_cards_data(df_cards_raw)
    df_mcc_clean = clean_mcc_data(mcc_dict)
    df_labels_clean = clean_labels_data(fraud_dict)
    
    # 💾 BỔ SUNG: Ghi bản sao lưu vật lý dạng Parquet cho 4 bảng tĩnh vào 2_silver
    logging.info("💾 Đang ghi file sao lưu các bảng tĩnh vào data/2_silver/...")
    df_users_clean.to_parquet(os.path.join(SILVER_DIR, "users_cleaned.parquet"), index=False)
    df_cards_clean.to_parquet(os.path.join(SILVER_DIR, "cards_cleaned.parquet"), index=False)
    df_mcc_clean.to_parquet(os.path.join(SILVER_DIR, "mcc_cleaned.parquet"), index=False)
    df_labels_clean.to_parquet(os.path.join(SILVER_DIR, "labels_cleaned.parquet"), index=False)
    
    # Thực hiện Bulk Load vào PostgreSQL
    loader.fast_load_dataframe(df_users_clean, "dim_users")
    loader.fast_load_dataframe(df_cards_clean, "dim_cards")
    loader.fast_load_dataframe(df_mcc_clean, "dim_mcc")
    
    # Giải phóng RAM để chuẩn bị xử lý dữ liệu lớn
    del df_users_raw, df_users_clean, df_cards_raw, df_cards_clean, mcc_dict, fraud_dict
    
    # ----------------------------------------------------------------
    # PHASE 2: STREAM CHUNK, MERGE NHÃN, BACKUP & BULK LOAD BẢNG FACT
    # ----------------------------------------------------------------
    logging.info("⏳ PHASE 2: Đang stream luồng giao dịch khổng lồ, gán nhãn và sao lưu...")
    
    chunk_size = 100000
    transaction_chunks = extract_csv("transactions_data.csv", chunksize=chunk_size)
    
    # 🎯 SỬA BỔ SUNG: Ép kiểu khóa chính của bảng nhãn về số nguyên trước khi vào vòng lặp
    if df_labels_clean is not None:
        df_labels_clean['transaction_id'] = df_labels_clean['transaction_id'].astype(int)
    
    chunk_count = 0
    for raw_chunk in transaction_chunks:
        chunk_count += 1
        logging.info(f"📦 Đang xử lý khối giao dịch thứ {chunk_count}...")
        
        # 1. Đồng đội dọn rác và kỹ nghệ đặc trưng thời gian
        cleaned_chunk = clean_transaction_chunk(raw_chunk)
        
        # 🎯 SỬA BỔ SUNG: Đảm bảo khóa của chunk giao dịch cũng là số nguyên để khớp với bảng nhãn
        cleaned_chunk['transaction_id'] = cleaned_chunk['transaction_id'].astype(int)
        
        # 2. Làm giàu dữ liệu: Nhúng trực tiếp nhãn 'is_fraud' trên RAM
        if df_labels_clean is not None:
            cleaned_chunk = cleaned_chunk.merge(df_labels_clean, on="transaction_id", how="left")
            cleaned_chunk["is_fraud"] = cleaned_chunk["is_fraud"].fillna(0).astype("int8")
        
        # 3. Ghi bản sao lưu vật lý từng Part giao dịch sạch vào 2_silver
        output_path = os.path.join(SILVER_DIR, f"transactions_cleaned_part_{chunk_count}.parquet")
        cleaned_chunk.to_parquet(output_path, index=False)
        
        # 4. Đổ siêu tốc khối dữ liệu hoàn hảo này vào bảng fact_transactions trong Postgres
        loader.fast_load_dataframe(cleaned_chunk, "fact_transactions")

if __name__ == "__main__":
    main()
# src/etl/load.py
import pandas as pd
import os
import glob
from sqlalchemy import create_engine

SILVER_DIR = "data/2_silver"

def get_postgres_engine():
    """Tạo kết nối chuẩn tới PostgreSQL Database"""
    # Bạn nên đọc các thông số này từ file config/database.ini để bảo mật mật khẩu
    username = "postgres"
    password = "your_secure_password"
    host = "localhost"
    port = "5432"
    database = "dss_fraud_db"
    
    conn_str = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
    return create_engine(conn_str)

def load_data_to_postgres():
    print("🐘 Bắt đầu giai đoạn Load: Nạp dữ liệu vào PostgreSQL...")
    engine = get_postgres_engine()
    
    # Tìm tất cả các file phân mảnh đã được biến đổi ở bước Transform
    parquet_files = glob.glob(os.path.join(SILVER_DIR, "transformed_part_*.parquet"))
    
    if not parquet_files:
        print("❌ Không tìm thấy dữ liệu đã biến đổi. Hãy chạy transform.py trước!")
        return

    is_first_chunk = True
    
    for file_path in parquet_files:
        print(f"⏳ Đang nạp file: {os.path.basename(file_path)}...")
        df_chunk = pd.read_parquet(file_path)
        
        # Đẩy dữ liệu vào bảng 'gold_dss_features' trong Postgres
        # Nếu là chunk đầu tiên thì ghi đè (replace), các chunk sau thì ghi nối tiếp (append)
        if is_first_chunk:
            df_chunk.to_sql(name="gold_dss_features", con=engine, if_exists="replace", index=False)
            is_first_chunk = False
        else:
            df_chunk.to_sql(name="gold_dss_features", con=engine, if_exists="append", index=False)
            
    print("🎉 Quá trình ETL thành công tốt đẹp! Cơ sở dữ liệu PostgreSQL đã sẵn sàng.")

if __name__ == "__main__":
    load_data_to_postgres()
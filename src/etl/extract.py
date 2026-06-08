# src/etl/extract.py
import pandas as pd
import json
import os

BRONZE_DIR = "data/1_bronze"

def extract_csv(file_name, chunksize=None):
    """Đọc file CSV thô từ vùng Bronze"""
    file_path = os.path.join(BRONZE_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {file_path}")
    
    if chunksize:
        return pd.read_csv(file_path, chunksize=chunksize)
    return pd.read_csv(file_path)

def extract_json(file_name):
    """Đọc file JSON thô từ vùng Bronze"""
    file_path = os.path.join(BRONZE_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {file_path}")
        
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    print("⏳ Đang thử nghiệm trích xuất dữ liệu mẫu...")
    df_users_test = extract_csv("users_data.csv")
    print(f"✔️ Trích xuất thành công users_data.csv. Kích thước: {df_users_test.shape}")
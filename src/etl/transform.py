# src/etl/transform.py
import pandas as pd
import logging
import os
from src.etl.extract import extract_csv, extract_json

SILVER_DIR = "data/2_silver"
os.makedirs(SILVER_DIR, exist_ok=True)

# Thiết lập hệ thống ghi log cơ bản để theo dõi tiến trình
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =========================================================================
# 🛠️ MODULE 1: CÁC HÀM LÀM SẠCH VÀ CHUẨN HÓA BẢNG TĨNH (DIMENSIONS)
# =========================================================================

def clean_users_data(df_raw):
    """Làm sạch và chuẩn hóa bảng hồ sơ Khách hàng"""
    logging.info("Đang làm sạch dữ liệu Users...")
    df = df_raw.rename(columns={'id': 'user_id', 'current_age': 'user_age'})
    
    # 1. Khử trùng lặp
    df = df.drop_duplicates(subset=['user_id'])
    
    # 2. Xử lý định dạng các cột tiền tệ
    currency_cols = ['per_capita_income', 'yearly_income', 'total_debt']
    for col in currency_cols:
        if col in df.columns:
            # Loại bỏ ký tự $ và dấu phẩy
            df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # >>> [XỬ LÝ DỮ LIỆU KHUYẾT THIẾU]: Điền bằng giá trị trung vị (median)
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    # 3. Xử lý cột điểm tín dụng
    if 'credit_score' in df.columns:
        df['credit_score'] = pd.to_numeric(df['credit_score'], errors='coerce')
        
        # >>> [XỬ LÝ DỮ LIỆU KHUYẾT THIẾU]: Điền bằng giá trị trung vị (median)
        median_credit = df['credit_score'].median()
        df['credit_score'] = df['credit_score'].fillna(median_credit)
        
    # Đồng bộ kiểu dữ liệu khóa chính phục vụ cho JOIN
    df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce').fillna(-1).astype(int)
    
    return df

def clean_cards_data(df_raw):
    """Làm sạch và chuẩn hóa bảng hồ sơ Thẻ"""
    logging.info("Đang làm sạch dữ liệu Cards...")
    df = df_raw.rename(columns={'id': 'card_id', 'client_id': 'user_id'})
    if 'clientid' in df.columns:
        df = df.rename(columns={'clientid': 'user_id'})
    
    # 1. Khử trùng lặp
    df = df.drop_duplicates(subset=['card_id'])
    
    # 2. Xử lý định dạng tiền tệ cho cột credit_limit
    if 'credit_limit' in df.columns:
        df['credit_limit'] = df['credit_limit'].astype(str).str.replace(r'[\$,]', '', regex=True)
        df['credit_limit'] = pd.to_numeric(df['credit_limit'], errors='coerce')
        
        # >>> [XỬ LÝ DỮ LIỆU KHUYẾT THIẾU]: Điền bằng giá trị trung vị (median)
        median_limit = df['credit_limit'].median()
        df['credit_limit'] = df['credit_limit'].fillna(median_limit)
    
    # 3. Chuẩn hóa chuỗi văn bản cho cột card_type
    if 'card_type' in df.columns:
        df['card_type'] = df['card_type'].astype(str).str.strip().str.lower()
        
        # >>> [XỬ LÝ DỮ LIỆU KHUYẾT THIẾU]: Điền giá trị trống bằng chuỗi 'unknown'
        df['card_type'] = df['card_type'].replace('nan', 'unknown').fillna('unknown')
        
    # Đồng bộ kiểu dữ liệu khóa ngoại để chuẩn bị lưu trữ
    df['card_id'] = pd.to_numeric(df['card_id'], errors='coerce').fillna(-1).astype(int)
    df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce').fillna(-1).astype(int)
    
    return df

def clean_mcc_data(mcc_dict):
    """Chuẩn hóa từ điển Danh mục MCC"""
    logging.info("Đang làm sạch dữ liệu MCC...")
    df = pd.DataFrame(list(mcc_dict.items()), columns=['mcc', 'merchant_category'])
    df['mcc'] = df['mcc'].astype(str).str.strip()
    return df

def clean_labels_data(fraud_dict):
    """Chuẩn hóa nhãn gian lận (Ground Truth)"""
    logging.info("Đang làm sạch dữ liệu Labels...")
    target_dict = fraud_dict.get('target', fraud_dict)
    
    df = pd.DataFrame(list(target_dict.items()), columns=['transaction_id', 'is_fraud'])
    
    # >>> [XỬ LÝ DỮ LIỆU KHUYẾT THIẾU / ĐỊNH DẠNG]: Map Yes/No về 1/0, khuyết điền mặc định là an toàn (0)
    df['is_fraud'] = df['is_fraud'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype('int8')
    df['transaction_id'] = df['transaction_id'].astype(str).str.strip()
    return df


# =========================================================================
# 🛠️ MODULE 2: HÀM LÀM SẠCH LÕI CHO GIAO DỊCH (CHUNK CLEANSING)
# =========================================================================

def clean_transaction_chunk(raw_chunk):
    """Dọn dẹp rác, chuẩn hóa định dạng và xử lý khuyết thiếu cho 1 khối giao dịch"""
    
    # 1. Chuẩn hóa tên cột
    chunk = raw_chunk.rename(columns={'id': 'transaction_id', 'client_id': 'user_id'})
    
    # 2. Khử trùng lặp giao dịch (Deduplication)
    chunk = chunk.drop_duplicates(subset=['transaction_id'])
    
    # 3. Xử lý số tiền (amount) - Giữ nguyên dấu âm (Refund) [1]
    if 'amount' in chunk.columns:
        chunk['amount'] = chunk['amount'].astype(str).str.replace(r'\$', '', regex=True)
        
        # >>> [XỬ LÝ DỮ LIỆU KHUYẾT THIẾU]: Điền khuyết thiếu bằng 0.0
        chunk['amount'] = pd.to_numeric(chunk['amount'], errors='coerce').fillna(0.0)
            
    # 4. Làm sạch cột phương thức quẹt thẻ (use_chip)
    if 'use_chip' in chunk.columns:
        chunk['use_chip'] = chunk['use_chip'].fillna('unknown')
        chunk['use_chip'] = chunk['use_chip'].astype(str).str.strip().str.lower()
        
        # >>> [XỬ LÝ DỮ LIỆU KHUYẾT THIẾU]: Thay thế giá trị trống bằng chuỗi 'unknown'
        chunk['use_chip'] = chunk['use_chip'].replace(['nan', ''], 'unknown')
        
    # Làm sạch cột lỗi giao dịch (errors)
    if 'errors' in chunk.columns:
        chunk['errors'] = chunk['errors'].fillna('no_error')
        chunk['errors'] = chunk['errors'].astype(str).str.strip().str.lower()
        
        # >>> [XỬ LÝ DỮ LIỆU KHUYẾT THIẾU]: Ô trống mang ý nghĩa nghiệp vụ là không lỗi [1]
        chunk['errors'] = chunk['errors'].replace(['nan', ''], 'no_error')

    # Xử lý cột merchant_state
    if 'merchant_state' in chunk.columns:
        chunk['merchant_state'] = chunk['merchant_state'].fillna('UNKNOWN')
        chunk['merchant_state'] = chunk['merchant_state'].astype(str).str.strip().str.upper()
        
        # >>> [XỬ LÝ DỮ LIỆU KHUYẾT THIẾU]: Điền khuyết thiếu bằng chuỗi 'UNKNOWN' [1]
        chunk['merchant_state'] = chunk['merchant_state'].replace(['NAN', ''], 'UNKNOWN')
        
    # Xử lý cột zip
    if 'zip' in chunk.columns:
        chunk['zip'] = chunk['zip'].fillna('unknown')
        chunk['zip'] = chunk['zip'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        
        # >>> [XỬ LÝ DỮ LIỆU KHUYẾT THIẾU]: Điền khuyết thiếu bằng chuỗi 'unknown' [1]
        chunk['zip'] = chunk['zip'].replace(['nan', ''], 'unknown')
        
    # 5. Ép kiểu chuẩn để đồng bộ hóa cho các bước JOIN ở tầng sau
    chunk['mcc'] = chunk['mcc'].astype(str).str.strip()
    chunk['user_id'] = pd.to_numeric(chunk['user_id'], errors='coerce').fillna(-1).astype(int)
    chunk['card_id'] = pd.to_numeric(chunk['card_id'], errors='coerce').fillna(-1).astype(int)
    chunk['transaction_id'] = chunk['transaction_id'].astype(str).str.strip()
    
    return chunk


# =========================================================================
# 🚀 HÀM ĐIỀU PHỐI (ORCHESTRATOR)
# =========================================================================

def run_etl_pipeline():
    logging.info("🚀 BẮT ĐẦU GIAI ĐOẠN TRANSFORM (BRONZE -> SILVER)...")
    
    # BƯỚC 1: Trích xuất dữ liệu thô từ Bronze
    df_users_raw = extract_csv("users_data.csv")
    df_cards_raw = extract_csv("cards_data.csv")
    mcc_dict = extract_json("mcc_codes.json")
    fraud_dict = extract_json("train_fraud_labels.json")

    # BƯỚC 2: Chạy các hàm làm sạch thô từng bảng tĩnh (Dimensions)
    df_users_clean = clean_users_data(df_users_raw)
    df_cards_clean = clean_cards_data(df_cards_raw)
    df_mcc_clean = clean_mcc_data(mcc_dict)
    df_labels_clean = clean_labels_data(fraud_dict)

    # BƯỚC 3: Lưu trữ các bảng tĩnh sạch vào thư mục Data/2 silver
    df_users_clean.to_parquet(os.path.join(SILVER_DIR, "users_cleaned.parquet"), index=False)
    df_cards_clean.to_parquet(os.path.join(SILVER_DIR, "cards_cleaned.parquet"), index=False)
    df_mcc_clean.to_parquet(os.path.join(SILVER_DIR, "mcc_cleaned.parquet"), index=False)
    df_labels_clean.to_parquet(os.path.join(SILVER_DIR, "labels_cleaned.parquet"), index=False)
    logging.info("✔️ Đã lưu trữ toàn bộ các bảng tĩnh sạch vào thư mục Data/2 silver/")

    # BƯỚC 4: Khởi tạo luồng làm sạch Chunk cho dữ liệu giao dịch khổng lồ
    chunk_count = 0
    transaction_chunks = extract_csv("transactions_data.csv", chunksize=100000)

    logging.info("⏳ Đang tiến hành làm sạch dữ liệu giao dịch theo từng khối...")
    for raw_chunk in transaction_chunks:
        chunk_count += 1
        
        # Làm sạch thô và xử lý khuyết thiếu cho từng phần giao dịch
        cleaned_chunk = clean_transaction_chunk(raw_chunk)
        
        # Lưu trực tiếp phần giao dịch sạch vào Data/2 silver/ dưới định dạng Parquet nén tối ưu
        output_path = os.path.join(SILVER_DIR, f"transactions_cleaned_part_{chunk_count}.parquet")
        cleaned_chunk.to_parquet(output_path, index=False)
        logging.info(f"✔️ Đã làm sạch và lưu Part {chunk_count}")
        
    logging.info("🎉 HOÀN THÀNH PIPELINE BRONZE -> SILVER! Toàn bộ file dữ liệu sạch đã nằm gọn trong Data/2 silver/")

if __name__ == "__main__":
    run_etl_pipeline()
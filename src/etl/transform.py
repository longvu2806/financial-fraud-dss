# src/etl/transform.py
import pandas as pd
import logging
import os
from src.etl.extract import extract_csv, extract_json

SILVER_DIR = "data/2_silver"
os.makedirs(SILVER_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =========================================================================
# 🛠️ MODULE 1: CÁC HÀM LÀM SẠCH VÀ CHUẨN HÓA BẢNG TĨNH (DIMENSIONS)
# =========================================================================
 
def clean_users_data(df_raw):
    """Làm sạch và chuẩn hóa bảng hồ sơ Khách hàng"""
    logging.info("Đang làm sạch dữ liệu Users...")
    df = df_raw.rename(columns={'id': 'user_id', 'current_age': 'user_age'})
    df = df.drop_duplicates(subset=['user_id'])
 
    money_cols = ['per_capita_income', 'yearly_income', 'total_debt']
    for col in money_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace('$', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.strip()
                .replace('nan', None)
                .astype(float)
            )
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
 
    median_credit = df['credit_score'].median()
    df['credit_score'] = df['credit_score'].fillna(median_credit)
    return df
 
 
def clean_cards_data(df_raw):
    """Làm sạch và chuẩn hóa bảng hồ sơ Thẻ"""
    logging.info("Đang làm sạch dữ liệu Cards...")
    df = df_raw.rename(columns={'id': 'card_id', 'clientid': 'user_id', 'client_id': 'user_id'})
    df = df.drop_duplicates(subset=['card_id'])
 
    if 'credit_limit' in df.columns:
        df['credit_limit'] = (
            df['credit_limit']
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
            .replace('nan', None)
            .astype(float)
        )
        median_limit = df['credit_limit'].median()
        df['credit_limit'] = df['credit_limit'].fillna(median_limit)
 
    if 'card_type' in df.columns:
        df['card_type'] = df['card_type'].astype(str).str.strip().str.lower()
        df['card_type'] = df['card_type'].replace('nan', 'unknown').fillna('unknown')
        
    # 🧼 CHUẨN HÓA BỔ SUNG: Phòng thủ cột card_brand cho bước Encoding nhãn AI
    if 'card_brand' in df.columns:
        df['card_brand'] = df['card_brand'].astype(str).str.strip().str.lower()
        df['card_brand'] = df['card_brand'].replace('nan', 'unknown').fillna('unknown')
 
    return df
 
 
def clean_mcc_data(mcc_dict):
    """Chuẩn hóa từ điển Danh mục MCC"""
    logging.info("Đang làm sạch dữ liệu MCC...")
    df = pd.DataFrame(list(mcc_dict.items()), columns=['mcc', 'merchant_category'])
    df['mcc'] = df['mcc'].astype(str).str.strip()
    return df
 
 
def clean_labels_data(fraud_dict):
    """Chuẩn hóa nhãn gian lận (Ground Truth) và dọn sạch rác hệ thống"""
    logging.info("Đang làm sạch dữ liệu Labels...")
    
    # 1. Chuyển từ điển JSON thành DataFrame ban đầu
    df = pd.DataFrame(list(fraud_dict.items()), columns=['transaction_id', 'is_fraud'])
    
    # 🎯 SỬA TẠI ĐÂY: Loại bỏ dòng metadata 'target' hoặc các ID rác chứa ký tự chữ ngay từ đầu
    df = df[df['transaction_id'].astype(str).str.isnumeric()]
    
    # 🎯 ÉP KIỂU KHÓA CHÍNH: Đảm bảo transaction_id luôn là số nguyên hệ int64 thuần túy để sẵn sàng JOIN
    df['transaction_id'] = df['transaction_id'].astype(int)
    
    # 2. Ánh xạ nhãn từ chữ (yes/no) sang số (1/0)
    df['is_fraud'] = (
        df['is_fraud']
        .astype(str).str.strip().str.lower()
        .map({'yes': 1, 'no': 0})
        .fillna(0)
        .astype('int8')
    )
    
    return df

# =========================================================================
# 🛠️ MODULE 2: HÀM LÀM SẠCH LÕI CHO GIAO DỊCH (CHUNK CLEANSING)
# =========================================================================

def clean_transaction_chunk(raw_chunk, fraud_dict=None):
    """Dọn dẹp rác, chuẩn hóa định dạng và xử lý khuyết thiếu cho 1 khối giao dịch"""
    chunk = raw_chunk.rename(columns={'id': 'transaction_id', 'client_id': 'user_id'})
    chunk = chunk.drop_duplicates(subset=['transaction_id'])
    
    # 🎯 1. ÉP KIỂU KHÓA CHÍNH TUYỆT ĐỐI (Sửa lỗi lệch định dạng ngầm)
    chunk['transaction_id'] = (
        chunk['transaction_id']
        .astype(str)
        .str.replace(r'\.0$', '', regex=True)
        .str.strip()
    )
    chunk = chunk[chunk['transaction_id'].str.isnumeric()]
    chunk['transaction_id'] = chunk['transaction_id'].astype(int)
    
    # 🎯 2. TỰ ĐỘNG GÁN NHÃN GIAN LẬN (MAPPING TỪ ĐIỂN TỐC ĐỘ CAO)
    if fraud_dict is not None:
        # Ánh xạ trực tiếp từ Dictionary gốc (yes -> 1, còn lại -> 0)
        chunk['is_fraud'] = chunk['transaction_id'].map(
            {int(str(k).strip()): (1 if str(v).strip().lower() == 'yes' else 0) 
             for k, v in fraud_dict.items() if str(k).strip().isnumeric()}
        )
        chunk['is_fraud'] = chunk['is_fraud'].fillna(0).astype('int8')
    else:
        chunk['is_fraud'] = 0
    # 3. Xử lý số tiền (amount) - Giữ nguyên dấu âm
    if 'amount' in chunk.columns:
        chunk['amount'] = (
            chunk['amount']
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
            .replace('nan', None)
            .astype(float)
        )
        chunk['amount'] = chunk['amount'].fillna(0.0)
        
        # 🎯 SỬA LỖI TOÁN HỌC: Cắt gọt Outlier tại mốc phân vị 99.9% toàn cục cố định (874.20 USD) [33]
        chunk['amount'] = chunk['amount'].clip(upper=874.20)
            
    # 4. Làm sạch cột phương thức quẹt thẻ (use_chip)
    if 'use_chip' in chunk.columns:
        chunk['use_chip'] = chunk['use_chip'].fillna('unknown')
        chunk['use_chip'] = chunk['use_chip'].astype(str).str.strip().str.lower()
        chunk['use_chip'] = chunk['use_chip'].replace(['nan', ''], 'unknown')
        
    # Làm sạch cột lỗi giao dịch (errors)
    if 'errors' in chunk.columns:
        chunk['errors'] = chunk['errors'].fillna('no_error')
        chunk['errors'] = chunk['errors'].astype(str).str.strip().str.lower()
        chunk['errors'] = chunk['errors'].replace(['nan', ''], 'no_error')

    # Xử lý cột merchant_state
    if 'merchant_state' in chunk.columns:
        chunk['merchant_state'] = chunk['merchant_state'].fillna('UNKNOWN')
        chunk['merchant_state'] = chunk['merchant_state'].astype(str).str.strip().str.upper()
        chunk['merchant_state'] = chunk['merchant_state'].replace(['NAN', ''], 'UNKNOWN')
        
    # Xử lý cột zip
    if 'zip' in chunk.columns:
        chunk['zip'] = chunk['zip'].fillna('unknown')
        chunk['zip'] = chunk['zip'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        chunk['zip'] = chunk['zip'].replace(['nan', ''], 'unknown')
        
    # 5. Ép kiểu chuẩn để đồng bộ hóa cho các bước JOIN và LOAD về sau
    chunk['mcc'] = chunk['mcc'].astype(str).str.strip()
    chunk['user_id'] = chunk['user_id'].fillna(-1).astype(int)
    chunk['card_id'] = chunk['card_id'].fillna(-1).astype(int)

    # ⏱️ TỐI ƯU HÓA: Chuẩn hóa date và Kỹ nghệ đặc trưng bằng cơ chế Vectorized siêu tốc
    if 'date' in chunk.columns:
        chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
        
        # Điền NaT bằng mốc mặc định (phòng thủ) trước khi bẻ cột số nguyên hệ thống int8
        hours = chunk['date'].dt.hour.fillna(-1).astype('int8')
        chunk['tx_hour'] = hours [35]
        chunk['tx_day_of_week'] = chunk['date'].dt.dayofweek.fillna(-1).astype('int8') [36]
        
        # Tốc độ cực nhanh nhờ toán học Vectorized thay thế cho hàm .apply(lambda) cũ
        chunk['is_night_tx'] = (hours.between(1, 5)).astype('int8') [37]
    
    return chunk


# =========================================================================
# 🚀 HÀM ĐIỀU PHỐI (ORCHESTRATOR)
# =========================================================================

# def run_etl_pipeline():
#     logging.info("🚀 BẮT ĐẦU GIAI ĐOẠN TRANSFORM (BRONZE -> SILVER)...")
    
#     df_users_raw = extract_csv("users_data.csv")
#     df_cards_raw = extract_csv("cards_data.csv")
#     mcc_dict = extract_json("mcc_codes.json")
#     fraud_dict = extract_json("train_fraud_labels.json")

#     df_users_clean = clean_users_data(df_users_raw)
#     df_cards_clean = clean_cards_data(df_cards_raw)
#     df_mcc_clean = clean_mcc_data(mcc_dict)
#     df_labels_clean = clean_labels_data(fraud_dict)

#     df_users_clean.to_parquet(os.path.join(SILVER_DIR, "users_cleaned.parquet"), index=False)
#     df_cards_clean.to_parquet(os.path.join(SILVER_DIR, "cards_cleaned.parquet"), index=False)
#     df_mcc_clean.to_parquet(os.path.join(SILVER_DIR, "mcc_cleaned.parquet"), index=False)
#     df_labels_clean.to_parquet(os.path.join(SILVER_DIR, "labels_cleaned.parquet"), index=False)
#     logging.info("✔️ Đã lưu trữ toàn bộ các bảng tĩnh sạch vào thư mục Data/2 silver/")

#     chunk_count = 0
#     transaction_chunks = extract_csv("transactions_data.csv", chunksize=100000)

#     logging.info("⏳ Đang tiến hành làm sạch dữ liệu giao dịch theo từng khối...")
#     for raw_chunk in transaction_chunks:
#         chunk_count += 1
#         cleaned_chunk = clean_transaction_chunk(raw_chunk)
        
#         output_path = os.path.join(SILVER_DIR, f"transactions_cleaned_part_{chunk_count}.parquet")
#         cleaned_chunk.to_parquet(output_path, index=False)
#         logging.info(f"✔️ Đã làm sạch và lưu Part {chunk_count}")

#     logging.info("🎉 HOÀN THÀNH PIPELINE BRONZE -> SILVER! Toàn bộ file dữ liệu sạch đã nằm gọn trong Data/2 silver/")

#if __name__ == "__main__":
    run_etl_pipeline()
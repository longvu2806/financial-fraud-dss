# src/etl/transform.py
import pandas as pd
import logging
import os
from src.etl.extract import extract_csv, extract_json

# Cấu hình thư mục đầu ra
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
 
    # ----------------------------------------------------------------
    # 2. [SỬA LỖI KIỂU DỮ LIỆU] Xóa ký tự '$' và ép về kiểu số float
    #    Ba cột này bị Pandas đọc nhầm thành str vì có tiền tố '$'
    #    Ví dụ: '$59696' -> 59696.0
    # ----------------------------------------------------------------
    money_cols = ['per_capita_income', 'yearly_income', 'total_debt']
    for col in money_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)           # Đảm bảo luôn là chuỗi trước khi thao tác
                .str.replace('$', '', regex=False)  # Xóa ký tự '$'
                .str.replace(',', '', regex=False)  # Xóa dấu ',' phân cách hàng nghìn nếu có
                .str.replace(r'^\((.+)\)$', r'-\1', regex=True)  # Thêm dòng này
                .str.strip()           # Xóa khoảng trắng thừa hai đầu
                .replace('nan', None)  # Chuyển chuỗi 'nan' về None thật để pd nhận diện
                .astype(float)         # Ép về số thực -> sẵn sàng cho tính toán toán học
            )
            # Điền khuyết thiếu bằng trung vị (median) - phòng thủ nếu có NaN phát sinh
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
 
    # 3. Xử lý dữ liệu khuyết thiếu cho điểm tín dụng (người khác đã làm, giữ nguyên)
    median_credit = df['credit_score'].median()
    df['credit_score'] = df['credit_score'].fillna(median_credit)
 
    return df
 
 
def clean_cards_data(df_raw):
    """Làm sạch và chuẩn hóa bảng hồ sơ Thẻ"""
    logging.info("Đang làm sạch dữ liệu Cards...")
    df = df_raw.rename(columns={'id': 'card_id', 'clientid': 'user_id'})
 
    # 1. Khử trùng lặp
    df = df.drop_duplicates(subset=['card_id'])
 
    # ----------------------------------------------------------------
    # 2. [SỬA LỖI KIỂU DỮ LIỆU] Xóa '$' và ép credit_limit về float
    #    Ví dụ: '$5000' -> 5000.0
    # ----------------------------------------------------------------
    if 'credit_limit' in df.columns:
        df['credit_limit'] = (
            df['credit_limit']
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.replace(r'^\((.+)\)$', r'-\1', regex=True)  # Thêm dòng này
            .str.strip()
            .replace('nan', None)
            .astype(float)
        )
        # Điền khuyết thiếu bằng trung vị - phòng thủ
        median_limit = df['credit_limit'].median()
        df['credit_limit'] = df['credit_limit'].fillna(median_limit)
 
    # ----------------------------------------------------------------
    # 3. [SỬA LỖI KIỂU DỮ LIỆU] Chuẩn hóa card_type về chữ thường
    #    Mục tiêu: đồng bộ nhãn, tránh lỗi GROUP BY / JOIN về sau
    #    Ví dụ: 'Visa ' -> 'visa', 'MASTERCARD' -> 'mastercard'
    # ----------------------------------------------------------------
    if 'card_type' in df.columns:
        df['card_type'] = (
            df['card_type']
            .astype(str)
            .str.strip()       # Xóa khoảng trắng thừa
            .str.lower()       # Đưa về chữ thường
        )
        # Điền khuyết thiếu cho chuỗi 'nan' phát sinh sau .astype(str)
        df['card_type'] = df['card_type'].replace('nan', 'unknown').fillna('unknown')
 
    return df
 
 
def clean_mcc_data(mcc_dict):
    """Chuẩn hóa từ điển Danh mục MCC"""
    logging.info("Đang làm sạch dữ liệu MCC...")
    df = pd.DataFrame(list(mcc_dict.items()), columns=['mcc', 'merchant_category'])
    # Chuẩn hóa chuỗi văn bản
    df['mcc'] = df['mcc'].astype(str).str.strip()
    return df
 
 
def clean_labels_data(fraud_dict):
    """Chuẩn hóa nhãn gian lận (Ground Truth)"""
    logging.info("Đang làm sạch dữ liệu Labels...")
    df = pd.DataFrame(list(fraud_dict.items()), columns=['transaction_id', 'is_fraud'])
    # Chuẩn hóa chữ 'Yes'/'No' thành số 1/0 và ép kiểu int8 để tối ưu RAM
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

def clean_transaction_chunk(raw_chunk):
    """Dọn dẹp rác, chuẩn hóa định dạng và xử lý khuyết thiếu cho 1 khối giao dịch"""
    
    # 1. Chuẩn hóa tên cột
    chunk = raw_chunk.rename(columns={'id': 'transaction_id', 'client_id': 'user_id'})
    
    # 2. Khử trùng lặp giao dịch (Deduplication)
    chunk = chunk.drop_duplicates(subset=['transaction_id'])
    
    # 3. Xử lý dữ liệu khuyết thiếu cho cột số tiền (amount)
    if 'amount' in chunk.columns:
        chunk['amount'] = (
            chunk['amount']
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.replace(r'^\((.+)\)$', r'-\1', regex=True)
            .str.strip()
            .replace('nan', None)
            .astype(float)
        )
        chunk['amount'] = chunk['amount'].fillna(0.0)
        percentile_99 = chunk['amount'].quantile(0.99)
        chunk['amount'] = chunk['amount'].clip(upper=percentile_99)
            
    # Làm sạch cột phương thức quẹt thẻ (use_chip)
    if 'use_chip' in chunk.columns:
        chunk['use_chip'] = chunk['use_chip'].astype(str).str.strip().str.lower()
        chunk['use_chip'] = chunk['use_chip'].replace('nan', 'unknown').fillna('unknown')
        
    # Làm sạch cột lỗi giao dịch (errors)
    if 'errors' in chunk.columns:
        chunk['errors'] = chunk['errors'].astype(str).str.strip().str.lower()
        chunk['errors'] = chunk['errors'].replace('nan', 'no_error').fillna('no_error')

    # Xử lý cột merchant_state
    if 'merchant_state' in chunk.columns:
        chunk['merchant_state'] = chunk['merchant_state'].astype(str).str.strip().str.upper()
        chunk['merchant_state'] = chunk['merchant_state'].replace('nan', 'unknow').fillna('unknow')
        
    # Xử lý cột zip (Mã định danh địa lý -> ép về chuỗi, không để dạng số float)
    if 'zip' in chunk.columns:
        # Loại bỏ phần đuôi .0 nếu Pandas vô tình hiểu nhầm zip là số float
        chunk['zip'] = chunk['zip'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        chunk['zip'] = chunk['zip'].replace('nan', 'unknown').fillna('unknown')
    # 5. Ép kiểu chuẩn bị cho JOIN (Type Casting)
    chunk['mcc'] = chunk['mcc'].astype(str).str.strip()
    chunk['user_id'] = chunk['user_id'].fillna(-1).astype(int)
    chunk['card_id'] = chunk['card_id'].fillna(-1).astype(int)

    # Tách timestamp thành 3 cột datetime mới
    if 'date' in chunk.columns:
        chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
        chunk['tx_hour'] = chunk['date'].dt.hour.astype('Int8')
        chunk['tx_day_of_week'] = chunk['date'].dt.dayofweek.astype('Int8')
        chunk['is_night_tx'] = chunk['date'].dt.hour.apply(
            lambda x: 1 if pd.notna(x) and 1 <= x <= 5 else 0
        ).astype('int8')
    
    return chunk

# =========================================================================
# 🛠️ MODULE 3: HÀM BIẾN ĐỔI & KỸ NGHỆ ĐẶC TRƯNG (FEATURE ENGINEERING)
# =========================================================================

def engineer_features(cleaned_chunk, df_users, df_cards, df_mcc, df_labels):
    """Gộp tri thức từ các bảng và sinh ra các biến toán học cho AI"""
    
    # 1. Thực hiện JOIN liên hoàn (Left Join bảo toàn giao dịch)
    merged = cleaned_chunk.merge(df_users, on='user_id', how='left')
    merged = merged.merge(df_cards, on='card_id', how='left')
    merged = merged.merge(df_mcc, on='mcc', how='left')
    merged = merged.merge(df_labels, on='transaction_id', how='left')
    
    # 2. Xử lý khoảng trống (NaN) phát sinh SAU KHI JOIN
    # Nếu giao dịch không có nhãn, mặc định là an toàn (0)
    merged['is_fraud'] = merged['is_fraud'].fillna(0).astype('int8')
    
    # 3. Hiện thực hóa Insight Đặc trưng (Features)
    if 'timestamp' in merged.columns:
        merged['timestamp'] = pd.to_datetime(merged['timestamp'], errors='coerce')
        # Insight 1: Đánh dấu giao dịch đêm muộn (1h - 5h sáng)
        merged['is_night_tx'] = merged['timestamp'].dt.hour.apply(lambda x: 1 if pd.notna(x) and 1 <= x <= 5 else 0).astype('int8')
    
    # Insight 2: Kỹ nghệ đặc trưng từ cột 'errors' (Có lỗi hay không)
    if 'errors' in merged.columns:
        # Nếu giao dịch có log lỗi (khác 'no_error'), gán bằng 1, ngược lại bằng 0
        merged['has_technical_error'] = merged['errors'].apply(lambda x: 0 if x == 'no_error' else 1).astype('int8')
    
    return merged

# =========================================================================
# 🚀 HÀM ĐIỀU PHỐI (ORCHESTRATOR)
# =========================================================================

def run_etl_pipeline():
    logging.info("🚀 BẮT ĐẦU GIAI ĐOẠN TRANSFORM (KIẾN TRÚC MODULAR)...")
    
    # BƯỚC 1: Trích xuất dữ liệu thô
    df_users_raw = extract_csv("users_data.csv")
    df_cards_raw = extract_csv("cards_data.csv")
    mcc_dict = extract_json("mcc_codes.json")
    fraud_dict = extract_json("train_fraud_labels.json")

    # BƯỚC 2: Chạy qua Trạm làm sạch tĩnh
    df_users = clean_users_data(df_users_raw)
    df_cards = clean_cards_data(df_cards_raw)
    df_mcc = clean_mcc_data(mcc_dict)
    df_labels = clean_labels_data(fraud_dict)

    # BƯỚC 3: Khởi tạo luồng xử lý Chunk cho dữ liệu siêu lớn
    chunk_count = 0
    # Đọc chunk 100k dòng/lần để bảo vệ RAM
    transaction_chunks = extract_csv("transactions_data.csv", chunksize=100000)

    logging.info("⏳ Đang stream xử lý: Làm sạch -> Kỹ nghệ -> Lưu trữ...")
    for raw_chunk in transaction_chunks:
        chunk_count += 1
        
        # Luồng A: Đưa vào máy giặt (Làm sạch chuyên sâu)
        cleaned_chunk = clean_transaction_chunk(raw_chunk)
        
        # Luồng B: Đưa vào xưởng chế tác (Gộp bảng & Sinh Feature)
        final_chunk = engineer_features(cleaned_chunk, df_users, df_cards, df_mcc, df_labels)
        
        # Luồng C: Lưu kho (Xuất file định dạng nén tối ưu)
        output_path = os.path.join(SILVER_DIR, f"transformed_part_{chunk_count}.parquet")
        final_chunk.to_parquet(output_path, index=False)
        logging.info(f"✔️ Đã xử lý trọn vẹn và lưu Part {chunk_count}")
        
    logging.info("🎉 HOÀN THÀNH PIPELINE TRANSFORM! Toàn bộ file Parquet đã nằm trong data/2_silver/")

if __name__ == "__main__":
    run_etl_pipeline()
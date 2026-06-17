import pandas as pd
import numpy as np
import os
import gc
import glob
from datetime import datetime

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN TẦNG SILVER & GOLD
# ==========================================
BASE_DATA_DIR = r"D:\DSS_CK\financial-fraud-dss\data"
SILVER_DIR = os.path.join(BASE_DATA_DIR, "2_silver")
GOLD_DIR = os.path.join(BASE_DATA_DIR, "3_gold") # Tầng Gold (Feature Store)

# Tự động tạo thư mục Gold nếu chưa tồn tại
os.makedirs(GOLD_DIR, exist_ok=True)

# File output bây giờ là PARQUET
OUTPUT_FILE = os.path.join(GOLD_DIR, "featured_transactions.parquet") 

FAST_TRAIN_MODE = True  # Bật True để lấy mẫu nén RAM (Chống sập máy)
DOWNSAMPLE_RATIO = 20   # Tỷ lệ 1 gian lận : 20 hợp pháp

# ==========================================
# HÀM TỐI ƯU HOÁ BỘ NHỚ RAM 
# ==========================================
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and not isinstance(col_type, pd.CategoricalDtype) and not str(col_type).startswith('datetime'):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)  
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            # Bảo vệ các cột ID và Date không bị chuyển thành category
            if col.endswith('_id') or col == 'id' or col == 'transaction_id' or 'date' in col:
                df[col] = df[col].astype(str)
            else:
                num_unique = len(df[col].unique())
                if num_unique / len(df) < 0.5:
                    df[col] = df[col].astype('category')
    return df

# ==========================================
# HÀM NẠP TỰ ĐỘNG CÁC KHỐI PARQUET SẠCH
# ==========================================
def load_silver_parquet_data():
    print(f"\n[Bước 1.1] Đang quét và nạp dữ liệu sạch từ: {SILVER_DIR}...")
    
    # 1. Đọc bảng Users
    cols_users = ['user_id', 'user_age', 'yearly_income', 'total_debt', 'credit_score', 'latitude', 'longitude']
    df_users = pd.read_parquet(os.path.join(SILVER_DIR, 'users_cleaned.parquet'), columns=cols_users)
    df_users.rename(columns={'latitude': 'user_lat', 'longitude': 'user_lon'}, inplace=True)
    df_users = reduce_mem_usage(df_users)
    
    # 2. Đọc bảng Cards
    cols_cards = ['card_id', 'user_id', 'card_type', 'credit_limit', 'card_on_dark_web', 'has_chip', 'acct_open_date', 'year_pin_last_changed']
    df_cards = pd.read_parquet(os.path.join(SILVER_DIR, 'cards_cleaned.parquet'), columns=cols_cards)
    df_cards = reduce_mem_usage(df_cards)
    
    # 3. Đọc linh hoạt TẤT CẢ các khối Transactions bằng Glob
    trans_pattern = os.path.join(SILVER_DIR, 'transactions_cleaned_part_*.parquet')
    trans_files = sorted(glob.glob(trans_pattern))
    print(f" -> Tự động phát hiện {len(trans_files)} khối dữ liệu transactions.")
    
    cols_trans = ['transaction_id', 'date', 'user_id', 'card_id', 'amount', 'use_chip', 'mcc', 'errors', 'merchant_state', 'merchant_id', 'is_fraud']
    
    # Check nếu có merch_lat/merch_long trong file
    temp_check = pd.read_parquet(trans_files[0])
    if 'merch_lat' in temp_check.columns:
        cols_trans.extend(['merch_lat', 'merch_long'])
        
    chunks = []
    for i, file_path in enumerate(trans_files):
        print(f"   + Đang nạp khối {i+1}/{len(trans_files)}...")
        chunk = pd.read_parquet(file_path, columns=cols_trans)
        
        if 'merch_lat' in chunk.columns:
            chunk.rename(columns={'merch_lat': 'merchant_lat', 'merch_long': 'merchant_lon'}, inplace=True)
        
        if FAST_TRAIN_MODE:
            fraud_chunk = chunk[chunk['is_fraud'] == 1]
            legit_chunk = chunk[chunk['is_fraud'] == 0]
            
            n_sample = len(fraud_chunk) * DOWNSAMPLE_RATIO
            if len(legit_chunk) > n_sample:
                legit_chunk = legit_chunk.sample(n=n_sample, random_state=42)
                
            chunk = pd.concat([fraud_chunk, legit_chunk], ignore_index=True)
            
        chunk = reduce_mem_usage(chunk)
        chunks.append(chunk)
        
    df_transactions = pd.concat(chunks, ignore_index=True)
    print(f" -> Tổng số dòng Transactions đã nạp vào RAM: {len(df_transactions):,}")
    
    return df_users, df_cards, df_transactions

# ==========================================
# LỚP FEATURE ENGINEERING 
# ==========================================
class RealDataFeatureEngineer:
    def __init__(self, df_users, df_cards, df_transactions):
        self.df_users = df_users
        self.df_cards = df_cards
        self.df_transactions = df_transactions

    def _join_data(self):
        df = self.df_transactions.merge(
            self.df_cards, 
            on=['card_id', 'user_id'], 
            how='left'
        )
        del self.df_transactions
        gc.collect()

        df = df.merge(
            self.df_users, 
            on='user_id', 
            how='left'
        )
        del self.df_users, self.df_cards
        gc.collect()

        return df

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return (R * c).astype(np.float32)

    def build_features(self):
        print("\n[Bước 2.1] Đang khớp nối dữ liệu thô (Merge & Join)...")
        df = self._join_data()

        print("[Bước 2.2] Đang chuyển đổi định dạng ngày tháng...")
        df['date'] = pd.to_datetime(df['date']) 
        df['is_error_txn'] = df['errors'].notnull().astype(np.int8)

        # --- GIAI ĐOẠN 1: TÍNH TOÁN THEO THẺ (CARD-BASED) ---
        print("[Bước 2.3] Sắp xếp theo Thẻ & Thời gian để tính Card Features...")
        df = df.sort_values(['card_id', 'date']).reset_index(drop=True)

        df['flag_is_refund'] = (df['amount'] < 0).astype(np.int8)
        df['abs_amount'] = df['amount'].abs().astype(np.float32)
        df['net_amount'] = df['amount'].astype(np.float32)
        df['log_abs_amount'] = np.log1p(df['abs_amount']).astype(np.float32)

        df['time_since_last_txn'] = df.groupby('card_id')['date'].diff().dt.total_seconds().fillna(999999).astype(np.float32)
        df['flag_rapid_sequence'] = (df['time_since_last_txn'] < 60).astype(np.int8)

        df = df.set_index('date')
        df['txn_count_1h'] = df.groupby('card_id')['transaction_id'].rolling('1h', closed='left').count().reset_index(level=0, drop=True).values
        df['txn_count_1h'] = df['txn_count_1h'].fillna(0).astype(np.int16)
        df['flag_burst_1h'] = (df['txn_count_1h'] > 5).astype(np.int8)

        rolling_7d = df.groupby('card_id')['abs_amount'].rolling('7d', closed='left')
        df['amount_mean_7d'] = rolling_7d.mean().reset_index(level=0, drop=True).values.astype(np.float32)
        df['amount_std_7d'] = rolling_7d.std().reset_index(level=0, drop=True).values.astype(np.float32)

        df = df.reset_index()
        gc.collect()

        # --- GIAI ĐOẠN 2: TÍNH TOÁN THEO ĐIỂM BÁN (MERCHANT-BASED) ---
        print("[Bước 2.4] Sắp xếp lại theo Điểm bán để tính Merchant Risk...")
        df = df.sort_values(['merchant_id', 'date']).reset_index(drop=True)
        df = df.set_index('date') 

        df['merchant_error_count_24h'] = df.groupby('merchant_id')['is_error_txn'].rolling('24h', closed='left').sum().reset_index(level=0, drop=True).values
        df['merchant_error_count_24h'] = df['merchant_error_count_24h'].fillna(0).astype(np.int16)
        df['flag_hot_merchant'] = (df['merchant_error_count_24h'] > 5).astype(np.int8)

        df = df.reset_index()
        df.drop(columns=['is_error_txn'], inplace=True)
        gc.collect()

        # --- GIAI ĐOẠN 3: CÁC LOGIC CÒN LẠI ---
        print("[Bước 2.5] Hoàn thiện các đặc trưng bảo mật và địa lý...")

        df['flag_night_txn'] = df['date'].dt.hour.between(0, 5).astype(np.int8)
        df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(np.int8)

        df['amount_zscore'] = np.where(df['amount_std_7d'] > 0, (df['abs_amount'] - df['amount_mean_7d']) / df['amount_std_7d'], 0).astype(np.float32)
        df['flag_amount_spike'] = (df['amount_zscore'] > 3.0).astype(np.int8)

        if 'credit_limit' in df.columns:
            df['flag_near_credit_limit'] = (df['abs_amount'] > (0.85 * df['credit_limit'])).astype(np.int8)
        else:
            df['flag_near_credit_limit'] = np.int8(0)

        df['flag_round_amount'] = (df['abs_amount'].isin([1, 5, 10, 20, 50, 100]) & (df['flag_is_refund'] == 0)).astype(np.int8)
        df['flag_high_mcc_risk'] = df['mcc'].isin(['4829', '6011', '7995', '5912', '6051', 4829, 6011, 7995, 5912, 6051]).astype(np.int8)

        df['flag_dark_web_card'] = (df['card_on_dark_web'] == 'Yes').astype(np.int8)
        df['flag_chip_bypass'] = ((df['has_chip'] == 'YES') & (df['use_chip'] != 'Chip Transaction')).astype(np.int8)

        if 'acct_open_date' in df.columns:
            df['acct_open_date'] = pd.to_datetime(df['acct_open_date'], errors='coerce')
            df['days_since_open'] = (df['date'] - df['acct_open_date']).dt.days.fillna(9999).astype(np.int32)
            df['flag_new_card'] = (df['days_since_open'] < 90).astype(np.int8)

        current_year = datetime.now().year
        if 'year_pin_last_changed' in df.columns:
            df['flag_pin_stale'] = ((current_year - df['year_pin_last_changed']) > 5).astype(np.int8)

        df['prev_merchant_state'] = df.groupby('card_id')['merchant_state'].shift(1)
        df['flag_state_hop'] = ((df['merchant_state'] != df['prev_merchant_state']) & (df['prev_merchant_state'].notnull()) & (df['time_since_last_txn'] < 6 * 3600)).astype(np.int8)

        df['flag_international'] = (df['merchant_state'].isnull() | (df['merchant_state'] == 'FOREIGN')).astype(np.int8)
        df['flag_night_txn'] = ((df['flag_night_txn'] == 1) & (df['flag_international'] == 0)).astype(np.int8)

        if 'user_lat' in df.columns and 'merchant_lat' in df.columns:
            df['distance_km'] = self._haversine_distance(df['user_lat'], df['user_lon'], df['merchant_lat'], df['merchant_lon'])
            df['distance_km'] = df['distance_km'].fillna(-1).astype(np.float32)
            df['flag_far_from_home'] = (df['distance_km'] > 200).astype(np.int8)

            time_hours = (df['time_since_last_txn'] / 3600.0) + 0.0001
            df['travel_speed_kmh'] = (df['distance_km'] / time_hours).astype(np.float32)
            df['flag_impossible_travel'] = ((df['travel_speed_kmh'] > 1000) & (df['distance_km'] > 100)).astype(np.int8)

        df['is_new_merchant'] = (~df.duplicated(subset=['card_id', 'merchant_id'])).astype(np.int8)

        if {'total_debt', 'yearly_income', 'credit_score'}.issubset(df.columns):
            df['flag_debt_pressure'] = ((df['total_debt'] / df['yearly_income'] > 0.8) & (df['credit_score'] < 620)).astype(np.int8)
        else:
            df['flag_debt_pressure'] = np.int8(0)

        gc.collect()
        print("=> Hoàn tất tính toán các đặc trưng (Features)!")
        return df

# ==========================================
# CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == "__main__":
    print("=== PIPELINE TẠO ĐẶC TRƯNG TỪ TẦNG SILVER VÀO TẦNG GOLD (PARQUET) ===")

    # 1. Đọc dữ liệu từ Parquet
    df_users, df_cards, df_transactions = load_silver_parquet_data()

    # 2. Xây dựng Features
    pipeline = RealDataFeatureEngineer(df_users, df_cards, df_transactions)
    final_df = pipeline.build_features()

    # Xóa rác
    del df_users, df_cards, df_transactions
    gc.collect()

    # 3. Lưu kết quả ra file Parquet để chuẩn bị cho Huấn luyện
    print(f"\n[Bước 3] Đang lưu tệp dữ liệu hoàn chỉnh ra tầng GOLD...")
    
    # Sử dụng engine pyarrow hoặc fastparquet, index=False để tối ưu dung lượng
    final_df.to_parquet(OUTPUT_FILE, index=False, engine='pyarrow')
    
    print(f" -> 🚀 Đã lưu thành công tệp Parquet tại: {OUTPUT_FILE}")
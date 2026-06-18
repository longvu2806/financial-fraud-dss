import pandas as pd
import numpy as np
import os
import gc
import glob
from datetime import datetime
from sklearn.model_selection import train_test_split

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN TẦNG SILVER & GOLD
# ==========================================
BASE_DATA_DIR = r"D:\DSS\financial-fraud-dss\data"
SILVER_DIR = os.path.join(BASE_DATA_DIR, "2_silver")
GOLD_DIR = os.path.join(BASE_DATA_DIR, "3_gold") # Tầng Gold (Feature Store)

os.makedirs(GOLD_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(GOLD_DIR, "featured_transactions.parquet") 

FAST_TRAIN_MODE = False  # BẮT BUỘC ĐỂ FALSE để nạp 100% tỷ lệ thực tế
DOWNSAMPLE_RATIO = 123   # Tỷ lệ 1 gian lận : 123 hợp pháp

# ==========================================
# HÀM TỐI ƯU HOÁ BỘ NHỚ RAM 
# ==========================================
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        
        # Kiểm tra an toàn cho các loại dữ liệu Datetime hoặc Category trên Pandas hiện đại
        is_datetime = pd.api.types.is_datetime64_any_dtype(df[col])
        is_categorical = isinstance(col_type, pd.CategoricalDtype)
        
        if col_type != object and not is_categorical and not is_datetime:
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Pandas/Numpy mới khuyến khích dùng tên chuẩn np.int8, np.float32...
            if str(col_type).startswith('int'):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)  
            elif str(col_type).startswith('float'):
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            if col.endswith('_id') or col == 'id' or col == 'transaction_id' or 'date' in col:
                df[col] = df[col].astype(str)
            else:
                num_unique = df[col].nunique()
                if num_unique / len(df) < 0.5:
                    df[col] = df[col].astype('category')
    return df

# ==========================================
# HÀM NẠP TỰ ĐỘNG CÁC KHỐI PARQUET SẠCH
# ==========================================
def load_silver_parquet_data():
    print(f"\n[Bước 1.1] Đang quét và nạp dữ liệu sạch từ: {SILVER_DIR}...")
    
    cols_users = ['user_id', 'user_age', 'yearly_income', 'total_debt', 'credit_score', 'latitude', 'longitude']
    df_users = pd.read_parquet(os.path.join(SILVER_DIR, 'users_cleaned.parquet'), columns=cols_users)
    df_users.rename(columns={'latitude': 'user_lat', 'longitude': 'user_lon'}, inplace=True)
    df_users = reduce_mem_usage(df_users)
    
    cols_cards = ['card_id', 'user_id', 'card_type', 'credit_limit', 'card_on_dark_web', 'has_chip', 'acct_open_date', 'year_pin_last_changed']
    df_cards = pd.read_parquet(os.path.join(SILVER_DIR, 'cards_cleaned.parquet'), columns=cols_cards)
    df_cards = reduce_mem_usage(df_cards)
    
    trans_pattern = os.path.join(SILVER_DIR, 'transactions_cleaned_part_*.parquet')
    trans_files = sorted(glob.glob(trans_pattern))
    print(f" -> Tự động phát hiện {len(trans_files)} khối dữ liệu transactions.")
    
    cols_trans = [
        'transaction_id', 'date', 'user_id', 'card_id', 'amount', 'use_chip', 
        'mcc', 'errors', 'merchant_state', 'merchant_id', 'is_fraud'
    ]
    
    temp_check = pd.read_parquet(trans_files[0])
    
    lat_col = 'merch_lat' if 'merch_lat' in temp_check.columns else ('merchant_lat' if 'merchant_lat' in temp_check.columns else None)
    lon_col = 'merch_long' if 'merch_long' in temp_check.columns else ('merchant_lon' if 'merchant_lon' in temp_check.columns else None)
    
    if lat_col and lon_col:
        cols_trans.extend([lat_col, lon_col])
        
    chunks = []
    for i, file_path in enumerate(trans_files):
        chunk = pd.read_parquet(file_path, columns=cols_trans)
        if lat_col and lat_col != 'merchant_lat':
            chunk.rename(columns={lat_col: 'merchant_lat'}, inplace=True)
        if lon_col and lon_col != 'merchant_lon':
            chunk.rename(columns={lon_col: 'merchant_lon'}, inplace=True)
            
        chunk = reduce_mem_usage(chunk)
        chunks.append(chunk)
        
    df_transactions = pd.concat(chunks, ignore_index=True)
    print(f" -> Tổng số dòng Transactions đã nạp vào RAM: {len(df_transactions):,}")
    return df_users, df_cards, df_transactions

# ==========================================
# LỚP FEATURE ENGINEERING CHUẨN NGHIỆP VỤ
# ==========================================
class RealDataFeatureEngineer:
    def __init__(self, df_users, df_cards, df_transactions):
        self.df_users = df_users
        self.df_cards = df_cards
        self.df_transactions = df_transactions

    def _join_data(self):
        df = self.df_transactions.merge(self.df_cards, on=['card_id', 'user_id'], how='left')
        del self.df_transactions
        gc.collect()
        df = df.merge(self.df_users, on='user_id', how='left')
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

        print("[Bước 2.4] Sắp xếp lại theo Điểm bán để tính Merchant Risk...")
        df = df.sort_values(['merchant_id', 'date']).reset_index(drop=True)
        df = df.set_index('date') 
        df['merchant_error_count_24h'] = df.groupby('merchant_id')['is_error_txn'].rolling('24h', closed='left').sum().reset_index(level=0, drop=True).values
        df['merchant_error_count_24h'] = df['merchant_error_count_24h'].fillna(0).astype(np.int16)
        df['flag_hot_merchant'] = (df['merchant_error_count_24h'] > 5).astype(np.int8)

        df = df.reset_index()
        df.drop(columns=['is_error_txn'], inplace=True)
        gc.collect()

        print("[Bước 2.5] Hoàn thiện các đặc trưng bảo mật và ĐỊA LÝ PROXY...")
        df['flag_night_txn'] = df['date'].dt.hour.between(0, 5).astype(np.int8)
        df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(np.int8)

        df['amount_zscore'] = np.where(df['amount_std_7d'] > 0, (df['abs_amount'] - df['amount_mean_7d']) / df['amount_std_7d'], 0).astype(np.float32)
        df['flag_amount_spike'] = (df['amount_zscore'] > 3.0).astype(np.int8)

        if 'credit_limit' in df.columns:
            df['flag_near_credit_limit'] = (df['abs_amount'] > (0.85 * df['credit_limit'])).astype(np.int8)
        else:
            df['flag_near_credit_limit'] = np.int8(0)

        df['flag_round_amount'] = (df['abs_amount'].isin([1, 5, 10, 20, 50, 100]) & (df['flag_is_refund'] == 0)).astype(np.int8)
        
        # Sửa lại cú pháp kiểm tra list an toàn
        mcc_risk_list = ['4829', '6011', '7995', '5912', '6051', 4829, 6011, 7995, 5912, 6051]
        df['flag_high_mcc_risk'] = df['mcc'].isin(mcc_risk_list).astype(np.int8)

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

        df['is_new_merchant'] = (~df.duplicated(subset=['card_id', 'merchant_id'])).astype(np.int8)

        if {'total_debt', 'yearly_income', 'credit_score'}.issubset(df.columns):
            df['flag_debt_pressure'] = ((df['total_debt'] / df['yearly_income'] > 0.8) & (df['credit_score'] < 620)).astype(np.int8)
        else:
            df['flag_debt_pressure'] = np.int8(0)

        # 🎯 TRÁI TIM CỦA FEATURE ENGINEERING: XỬ LÝ PROXY ĐỊA LÝ KHI THIẾU TỌA ĐỘ MERCHANT
        state_centroids = {
            'CA': (36.7783, -119.4179), 'NY': (40.7128, -74.0060), 'TX': (31.9686, -99.9018),
            'FL': (27.6648, -81.5158), 'IL': (40.6331, -89.3985), 'PA': (41.2033, -77.1945),
            'OH': (40.4173, -82.9071), 'GA': (32.1656, -82.9001), 'NC': (35.7596, -79.0193),
            'MI': (44.3148, -85.6024), 'NJ': (40.0583, -74.4057), 'VA': (37.4316, -78.6569)
        }

        if 'merchant_state' in df.columns and 'user_lat' in df.columns:
            print(" -> 🎯 Kích hoạt thuật toán suy luận khoảng cách Proxy qua Centroid của Bang...")
            df['merchant_proxy_lat'] = df['merchant_state'].map(lambda x: state_centroids.get(x, (37.0902, -95.7129))[0]).astype(np.float32)
            df['merchant_proxy_lon'] = df['merchant_state'].map(lambda x: state_centroids.get(x, (37.0902, -95.7129))[1]).astype(np.float32)
            
            df['distance_km'] = self._haversine_distance(df['user_lat'], df['user_lon'], df['merchant_proxy_lat'], df['merchant_proxy_lon'])
            df['distance_km'] = df['distance_km'].fillna(-1).astype(np.float32)
            df['flag_far_from_home'] = (df['distance_km'] > 500).astype(np.int8)

            time_hours = (df['time_since_last_txn'] / 3600.0) + 0.0001
            df['travel_speed_kmh'] = (df['distance_km'] / time_hours).astype(np.float32)
            df['flag_impossible_travel'] = ((df['travel_speed_kmh'] > 1000) & (df['distance_km'] > 200)).astype(np.int8)
            
            df.drop(columns=['merchant_proxy_lat', 'merchant_proxy_lon'], inplace=True)
        else:
            df['distance_km'] = np.float32(-1)
            df['flag_far_from_home'] = np.int8(0)
            df['travel_speed_kmh'] = np.float32(-1)
            df['flag_impossible_travel'] = np.int8(0)

        gc.collect()
        print("=> Hoàn tất tính toán các đặc trưng (Features)!")
        return df

# ==========================================
# CHƯƠNG TRÌNH CHÍNH CHUẨN MLOPS
# ==========================================
if __name__ == "__main__":
    print("=== PIPELINE TẠO ĐẶC TRƯNG TỪ TẦNG SILVER VÀO TẦNG GOLD ===")

    # 1. Đọc 100% dữ liệu gốc (13 triệu dòng)
    df_users, df_cards, df_transactions = load_silver_parquet_data()

    # 2. Xây dựng Features trên dữ liệu gốc (Đã bao gồm Proxy Địa lý)
    pipeline = RealDataFeatureEngineer(df_users, df_cards, df_transactions)
    final_df = pipeline.build_features()

    del df_users, df_cards, df_transactions
    gc.collect()
    
    # 3. CHIA TẬP NGHIÊM NGẶT (Giữ nguyên phân phối thực tế 1:1230)
    print("\n[Bước 3] Đang tiến hành chia tập Train/Test nguyên bản...")
    X = final_df.drop(columns=['is_fraud'])
    y = final_df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    df_train = X_train.copy()
    df_train['is_fraud'] = y_train
    
    df_test = X_test.copy()
    df_test['is_fraud'] = y_test
    
    del final_df, X, y, X_train, X_test, y_train, y_test
    gc.collect()

    # 4. CHỈ DOWNSAMPLE TẬP TRAIN (Áp dụng đúng tham số DOWNSAMPLE_RATIO = 123)
    print(f"\n[Bước 4] Thực hiện Negative Downsampling (Tỷ lệ 1:{DOWNSAMPLE_RATIO}) duy nhất trên tập huấn luyện...")
    fraud_train = df_train[df_train['is_fraud'] == 1]
    legit_train_full = df_train[df_train['is_fraud'] == 0]
    
    n_sample_to_keep = len(fraud_train) * DOWNSAMPLE_RATIO
    if len(legit_train_full) > n_sample_to_keep:
        legit_train_downsampled = legit_train_full.sample(n=n_sample_to_keep, random_state=42)
    else:
        legit_train_downsampled = legit_train_full
        
    df_train_downsampled = pd.concat([fraud_train, legit_train_downsampled]).sample(frac=1, random_state=42)

    # 5. LƯU RA 2 FILE RIÊNG BIỆT
    print(f"\n[Bước 5] Đang lưu tệp dữ liệu ra tầng 3_gold...")
    train_path = os.path.join(GOLD_DIR, "train_gold_downsampled.parquet")
    test_path = os.path.join(GOLD_DIR, "test_gold_original.parquet")
    
    df_train_downsampled.to_parquet(train_path, index=False, engine='pyarrow')
    df_test.to_parquet(test_path, index=False, engine='pyarrow')
    
    print(f" -> 🚀 XONG! Kho đặc trưng Gold đã kích hoạt thành công Proxy Địa lý và chia tập chuẩn!")
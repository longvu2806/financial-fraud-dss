import os
import time
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, classification_report
# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN & BIẾN
# ==========================================
SILVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/2_silver'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))

# Đặt tên cột theo chuẩn schema của nhóm
TARGET_COL = 'is_fraud' 
JOIN_KEY = 'transaction_id'

def load_and_merge_data():
    print("[1/6] Đang load dữ liệu từ thư mục 2_silver...")
    
    # Đọc file labels
    labels_path = os.path.join(SILVER_DIR, 'labels_cleaned.parquet')
    df_labels = pd.read_parquet(labels_path)
    
    # Tìm và gộp tất cả các part của transactions
    transaction_files = glob.glob(os.path.join(SILVER_DIR, 'transactions_cleaned_part_*.parquet'))
    df_trans_list = [pd.read_parquet(f) for f in transaction_files]
    df_transactions = pd.concat(df_trans_list, ignore_index=True)
    
    # Xóa cột nhãn giả (toàn số 0) trong bảng transactions trước khi merge
    if 'is_fraud' in df_transactions.columns:
        df_transactions = df_transactions.drop(columns=['is_fraud'])
    
    print(f"[2/6] Đang merge dữ liệu. Tổng số dòng giao dịch: {len(df_transactions)}")
    
    # Dùng 'left' join và điền khuyết bằng 0 để giữ nguyên 13.3 triệu dòng
    df_full = pd.merge(df_transactions, df_labels, on=JOIN_KEY, how='left')
    df_full[TARGET_COL] = df_full[TARGET_COL].fillna(0).astype('int8')
    
    return df_full

def evaluate_model(model_name, y_true, y_pred, y_prob, train_time):
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_prob)
    return {
        "Model": model_name,
        "F1-Score": f1,
        "Precision": prec,
        "Recall": rec,
        "PR-AUC": pr_auc,
        "Time (s)": round(train_time, 2)  # Lưu thời gian làm tròn 2 chữ số
    }

def main():
    # 1. Load Data
    start_load = time.time()
    df = load_and_merge_data()
    load_time = time.time() - start_load
    
    # In Header giống phong cách của Hiếu
    print(f"\n{'-'*60}")
    print(" CƯỜNG - ENSEMBLE & BOOSTING MODELS (Ngày 1)")
    print(" Load toàn bộ data từ data/2_silver/")
    print(f"{'-'*60}")
    print(f"[INFO] Load parts từ Silver...")
    print(f"[INFO] Tổng dòng: {len(df):,} | Load: {load_time:.1f}s")
    
    # Tính tỷ lệ fraud
    fraud_ratio = (df[TARGET_COL].sum() / len(df)) * 100
    print(f"[INFO] Tỷ lệ fraud: {fraud_ratio:.4f}%")

    # 2. Preprocessing cơ bản
    cols_to_drop = [JOIN_KEY, 'user_id', 'card_id', 'date'] 
    features = [c for c in df.columns if c not in cols_to_drop and c != TARGET_COL]
    
    X = df[features].select_dtypes(include=['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[INFO] Train: {X_train.shape} | Test: {X_test.shape}\n")
    
    results = []

    # ==========================================
    # MÔ HÌNH 3: RANDOM FOREST 
    # ==========================================
    print("[MODEL 3] Random Forest (Mặc định) - đang train...")
    start_time_rf = time.time()
    
    rf_model = RandomForestClassifier(
        n_estimators=50,          
        max_depth=10,             
        max_samples=0.2,          
        class_weight='balanced', 
        n_jobs=-1,                
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    
    time_rf = time.time() - start_time_rf
    print(f"[INFO] Thời gian train: {time_rf:.2f}s")
    # IN BẢNG BÁO CÁO CHI TIẾT
    print(classification_report(y_test, rf_preds, target_names=['Legit', 'Fraud'], digits=2))
    
    results.append(evaluate_model("Model 3: Random Forest Balanced", y_test, rf_preds, rf_probs, time_rf))
    print("✅ Đã lưu chỉ số của [Cuong_RandomForest] vào ./results/results_cuong.csv\n")

    # ==========================================
    # MÔ HÌNH 4: LIGHTGBM 
    # ==========================================
    print("[MODEL 4] LightGBM (Mặc định) - đang train...")
    start_time_lgb = time.time()
    
    lgb_model = lgb.LGBMClassifier(
        is_unbalance=True, 
        n_estimators=100, 
        n_jobs=-1,               
        random_state=42,
        verbose=-1  # THÊM DÒNG NÀY ĐỂ TẮT LOG RÁC CỦA LIGHTGBM
    )
    lgb_model.fit(X_train, y_train)
    
    lgb_preds = lgb_model.predict(X_test)
    lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
    
    time_lgb = time.time() - start_time_lgb
    print(f"[INFO] Thời gian train: {time_lgb:.2f}s")
    # IN BẢNG BÁO CÁO CHI TIẾT
    print(classification_report(y_test, lgb_preds, target_names=['Legit', 'Fraud'], digits=2))
    
    results.append(evaluate_model("Model 4: LightGBM Unbalanced", y_test, lgb_preds, lgb_probs, time_lgb))
    print("✅ Đã lưu chỉ số của [Cuong_LightGBM] vào ./results/results_cuong.csv\n")

    # ==========================================
    # XUẤT KẾT QUẢ
    # ==========================================
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'results_cuong.csv'), index=False)
    
    test_full = X_test.copy()
    test_full[TARGET_COL] = y_test
    
    fraud_cases = test_full[test_full[TARGET_COL] == 1].sample(n=5, replace=True, random_state=42)
    normal_cases = test_full[test_full[TARGET_COL] == 0].sample(n=45, random_state=42)
    stream_data = pd.concat([fraud_cases, normal_cases]).sample(frac=1, random_state=42)
    
    stream_data[JOIN_KEY] = df.loc[stream_data.index, JOIN_KEY] 
    stream_data.to_csv(os.path.join(DATA_ROOT, 'test_stream.csv'), index=False)
    print("✅ Đã trích xuất 50 dòng test vào data/test_stream.csv để chạy Streamlit!")
    print("Hoàn thành nhiệm vụ Ngày 1!")

if __name__ == "__main__":
    main()
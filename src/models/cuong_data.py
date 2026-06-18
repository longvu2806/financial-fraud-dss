import os
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, classification_report

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN TẦNG GOLD (2 FILES)
# ==========================================
GOLD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/3_gold'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))

TRAIN_FILE = os.path.join(GOLD_DIR, 'train_gold_downsampled.parquet')
TEST_FILE = os.path.join(GOLD_DIR, 'test_gold_original.parquet')

TARGET_COL = 'is_fraud' 
JOIN_KEY = 'transaction_id'

def evaluate_model(model_name, y_true, y_pred, y_prob, train_time):
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob)
    return {
        "Model": model_name,
        "F1-Score": f1,
        "Precision": prec,
        "Recall": rec,
        "PR-AUC": pr_auc,
        "Time (s)": round(train_time, 2)
    }

def main():
    print(f"\n{'-'*60}")
    print(" CƯỜNG - ENSEMBLE & BOOSTING MODELS")
    print(" Load dữ liệu Train/Test đã chia sẵn từ tầng GOLD")
    print(f"{'-'*60}")

    # ==========================================
    # 1. ĐỌC TRỰC TIẾP 2 FILE RIÊNG BIỆT
    # ==========================================
    start_load = time.time()
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
        raise FileNotFoundError("Thiếu file Parquet! Hãy đảm bảo script fill_data_to_gold.py đã chạy xong.")
        
    df_train = pd.read_parquet(TRAIN_FILE)
    df_test = pd.read_parquet(TEST_FILE)
    load_time = time.time() - start_load
    
    print(f"[INFO] Load thành công trong {load_time:.1f}s")
    print(f" -> Train (Đã nén 1:123): {df_train.shape[0]:,} dòng")
    print(f" -> Test (Nguyên bản): {df_test.shape[0]:,} dòng")

    # ==========================================
    # 2. CHUẨN BỊ FEATURES CHO MÔ HÌNH (Chỉ lấy số)
    # ==========================================
    cols_to_drop = [JOIN_KEY, 'user_id', 'card_id', 'merchant_id', 'date', 'acct_open_date', TARGET_COL]
    
    # Ép kiểu dữ liệu an toàn cho LightGBM và Random Forest (Loại bỏ các cột chữ)
    X_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns]).select_dtypes(include=[np.number])
    y_train = df_train[TARGET_COL]
    
    X_test = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns]).select_dtypes(include=[np.number])
    y_test = df_test[TARGET_COL]
    
    # Đồng bộ hóa các cột (Đề phòng Train/Test lệch cột do quá trình preprocessing)
    X_train, X_test = X_train.align(X_test, join='inner', axis=1)
    
    print(f"[INFO] Đang đưa {X_train.shape[1]} features số học vào huấn luyện...\n")
    results = []

    # ==========================================
    # 3. MÔ HÌNH 3: RANDOM FOREST 
    # ==========================================
    print("[MODEL 3] Random Forest (class_weight='balanced') - đang train...")
    start_time_rf = time.time()
    
    rf_model = RandomForestClassifier(
        n_estimators=100,          
        max_depth=12,             
        #class_weight='balanced', 
        n_jobs=-1,                
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    
    time_rf = time.time() - start_time_rf
    print(f"[INFO] Thời gian train: {time_rf:.2f}s")
    print(classification_report(y_test, rf_preds, target_names=['Legit', 'Fraud'], digits=2, zero_division=0))
    results.append(evaluate_model("Model 3: Random Forest", y_test, rf_preds, rf_probs, time_rf))

    # ==========================================
    # 4. MÔ HÌNH 4: LIGHTGBM 
    # ==========================================
    print("\n[MODEL 4] LightGBM (is_unbalance=True) - đang train...")
    start_time_lgb = time.time()
    
    lgb_model = lgb.LGBMClassifier(
        #is_unbalance=True, 
        n_estimators=150, 
        n_jobs=-1,               
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    lgb_preds = lgb_model.predict(X_test)
    lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
    
    time_lgb = time.time() - start_time_lgb
    print(f"[INFO] Thời gian train: {time_lgb:.2f}s")
    print(classification_report(y_test, lgb_preds, target_names=['Legit', 'Fraud'], digits=2, zero_division=0))
    results.append(evaluate_model("Model 4: LightGBM", y_test, lgb_preds, lgb_probs, time_lgb))

    # ==========================================
    # 5. XUẤT KẾT QUẢ & LUỒNG STREAMING
    # ==========================================
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, 'results_cuong.csv'), index=False)
    print("\n✅ Đã lưu: results_cuong.csv")
    
    # 🎯 FIX LỖI 37 CỘT: Trích xuất trực tiếp từ df_test nguyên bản để giữ 100% tất cả các cột
    print("[INFO] Đang tạo file streaming chứa đầy đủ các cột địa lý & chuỗi...")
    fraud_cases = df_test[df_test[TARGET_COL] == 1].sample(n=5, replace=True, random_state=42)
    normal_cases = df_test[df_test[TARGET_COL] == 0].sample(n=45, random_state=42)
    stream_data = pd.concat([fraud_cases, normal_cases]).sample(frac=1, random_state=42)
    
    stream_data.to_csv(os.path.join(DATA_ROOT, 'test_stream.csv'), index=False)
    print(f"✅ Đã xuất: test_stream.csv (Số lượng cột thực tế: {stream_data.shape[1]})")

if __name__ == "__main__":
    main()
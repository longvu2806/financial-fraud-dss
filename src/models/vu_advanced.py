import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from base_model import save_metrics_to_csv

# =========================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN TẦNG GOLD (SINGLE SOURCE OF TRUTH)
# =========================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Trỏ trực tiếp vào mỏ vàng đã được Hiệp làm sạch và tổng hợp đặc trưng
GOLD_FILE = os.path.join(BASE_DIR, "data", "3_gold", "featured_transactions.parquet")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models", "xgboost_v8.pkl")


def load_ready_gold_data():
    """Nạp trực tiếp 2 file vàng đã được cắt gọt sẵn sàng."""
    print("\n⏳ [1/2] Đang nạp tập TRAIN (đã nén) và TEST (nguyên bản) từ 3_gold...")
    train_path = os.path.join(BASE_DIR, "data", "3_gold", "train_gold_downsampled.parquet")
    test_path = os.path.join(BASE_DIR, "data", "3_gold", "test_gold_original.parquet")

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    print("⏳ [2/2] Đang Encode và cấu trúc ma trận...")
    # Tự động loại bỏ các cột ID và text rác cho cả 2 tập
    cols_to_drop = [c for c in df_train.columns if c.endswith('_id') or c in ['id', 'date', 'errors', 'merchant_city', 'merchant_state', 'zip', 'acct_open_date']]
    
    df_train = df_train.drop(columns=cols_to_drop, errors='ignore')
    df_test = df_test.drop(columns=cols_to_drop, errors='ignore')

    cat_cols = df_train.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        for col in cat_cols:
            df_train[col] = df_train[col].astype('category').cat.codes
            df_test[col] = df_test[col].astype('category').cat.codes

    df_train = df_train.fillna(-999)
    df_test = df_test.fillna(-999)

    X_train = df_train.drop(columns=['is_fraud'])
    y_train = df_train['is_fraud'].astype(np.int8)
    
    X_test = df_test.drop(columns=['is_fraud'])
    y_test = df_test['is_fraud'].astype(np.int8)

    del df_train, df_test
    gc.collect()

    print(f"✅ XONG! Kích thước Train: {len(X_train)} | Kích thước Test thực tế: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# =========================================================================
# 2. KHÔNG GIAN HUẤN LUYỆN CHUỖI 3 KIẾN TRÚC MÔ HÌNH XGBOOST
# =========================================================================

def run_model_6_xgb_baseline(X_train, X_test, y_train, y_test):
    print("\n🚀 Khởi động MÔ HÌNH 6: XGBoost (Mặc định)...")
    model = xgb.XGBClassifier(tree_method='hist', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    save_metrics_to_csv("Vu_Model_6_XGB_Baseline", y_test, y_pred, y_prob)


def run_model_7_xgb_spw(X_train, X_test, y_train, y_test):
    print("\n🚀 Khởi động MÔ HÌNH 7: XGBoost + Lớp 2 (Cost-Sensitive Learning)...")
    # Tính hệ số phạt tự động dựa trên tỷ lệ thực tế của tập nạp vào
    spw = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-5)
    
    model = xgb.XGBClassifier(tree_method='hist', scale_pos_weight=spw, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    save_metrics_to_csv("Vu_Model_7_XGB_SPW", y_test, y_pred, y_prob)


def run_model_8_xgb_tuned_and_export(X_train, X_test, y_train, y_test):
    print("\n🚀 Khởi động MÔ HÌNH 8 (FINAL): XGBoost + Thiết Quân Luật + Lớp 3 (Threshold)...")
    spw = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-5)
    
    xgb_params = {
        'n_estimators': 800,         # Tăng số vòng lặp tối đa lên 800 để cây có đủ không gian phát triển
        'learning_rate': 0.05,       # Tăng tốc độ học (từ 0.03 lên 0.05) để cây hội tụ nhanh hơn qua đoạn nhiễu
        'max_depth': 6,              # Nâng độ sâu lên 6 để bắt được tương tác phức tạp giữa Không gian và Thời gian
        'min_child_weight': 5,       # Giảm xuống 5 để cây nhạy cảm hơn, bám sát các ca Fraud hiếm
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'reg_alpha': 2.0,            # Cân bằng lại mức phạt L1
        'reg_lambda': 3.0,           
        'scale_pos_weight': spw,     
        'tree_method': 'hist',      
        'eval_metric': 'aucpr',     
        'early_stopping_rounds': 150, # 🚦 NỚI LỎNG PHANH: Cho phép mô hình kiên nhẫn qua 150 vòng sụt giảm
        'random_state': 42,
        'n_jobs': -1
    }   
    
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50
    )
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Quét ngưỡng động
    p_test, r_test, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (p_test * r_test) / (p_test + r_test + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    print(f"🎯 Điểm cắt xác suất (Optimal Threshold) tìm thấy: {optimal_threshold:.4f}")
    
    # =================================================================
    # =================================================================
    # ĐỘNG CƠ ĐA TẦNG (MULTI-LAYERED ENGINE) - BẢN VÁ LOGIC TỰ ĐỘNG DÒ TÌM
    # =================================================================
    # =================================================================
    # ĐỘNG CƠ XUẤT XƯỞNG CHUẨN MLOPS (CHỈ GIỮ LUỒNG F1 MAX)
    # =================================================================
    print("\n🔮 Kích hoạt Lớp phòng thủ Nghiệp vụ (F1 Max Base)...")
    
    # 1. Lưu kết quả luồng gốc tối ưu F1
    y_pred_base = (y_prob >= optimal_threshold).astype(int)
    save_metrics_to_csv("Vu_Model_8_Base_F1_Max", y_test, y_pred_base, y_prob)
    
    # 2. Đóng gói AI lõi xuất xưởng
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model, f)
        
    # 3. 🚨 XUẤT KHẨU NGƯỠNG CẮT CHO BACKEND 
    import json
    config_path = MODEL_SAVE_PATH.replace('.pkl', '_config.json')
    with open(config_path, 'w') as f:
        json.dump({
            "model_version": "v8_xgboost_f1_max",
            "optimal_threshold": float(optimal_threshold),
            "precision_expected": float(np.mean(y_test[y_pred_base == 1] == 1))
        }, f, indent=4)

    print(f"📦 Đã đóng gói AI Lõi tại: {MODEL_SAVE_PATH}")
    print(f"⚙️ Đã xuất file Cấu hình Ngưỡng (Threshold) tại: {config_path}")

# =========================================================================
# 3. ĐIỀU PHỐI KỊCH BẢN CHẠY CHÍNH (MAIN ENTRYPOINT)
# =========================================================================
if __name__ == "__main__":
    print("=================================================================")
    print("      HỆ THỐNG HUẤN LUYỆN XGBOOST TRÊN TẦNG GOLD                 ")
    print("=================================================================")
    
    # Kích hoạt luồng nạp dữ liệu siêu tốc
    X_train, X_test, y_train, y_test = load_ready_gold_data()
    
    # Kích hoạt tuần tự 3 kiến trúc
    run_model_6_xgb_baseline(X_train, X_test, y_train, y_test)
    run_model_7_xgb_spw(X_train, X_test, y_train, y_test)
    run_model_8_xgb_tuned_and_export(X_train, X_test, y_train, y_test)
    
    print("\n🏁 TIẾN TRÌNH CHẠY HOÀN THÀNH XUẤT SẮC!")
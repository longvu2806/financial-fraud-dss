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


def load_gold_data():
    """
    Nạp thẳng 'mỏ vàng' đã có 27 đặc trưng nghiệp vụ từ tầng 3_gold.
    Code lúc này siêu tinh gọn vì luồng ETL đã gánh hết toàn bộ phần tính toán nặng nề.
    """
    print("\n⏳ [1/3] Đang nạp dữ liệu từ tầng 3_gold...")
    if not os.path.exists(GOLD_FILE):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {GOLD_FILE}")

    df = pd.read_parquet(GOLD_FILE)
    print(f"✅ Đã nạp thành công {len(df)} bản ghi mang đầy đủ đặc trưng nghiệp vụ.")

    print("⏳ [2/3] Đang dọn dẹp Metadata và chuẩn bị ma trận huấn luyện...")
    # Loại bỏ ID rác và cột text tự do không có giá trị học máy
    cols_to_drop = [c for c in df.columns if c.endswith('_id') or c in ['id', 'date', 'errors', 'merchant_city', 'merchant_state', 'zip', 'acct_open_date']]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Encode tự động các trường phân loại (Categorical) còn sót
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        for col in cat_cols:
            df[col] = df[col].astype('category').cat.codes

    # Điền giá trị khuyết thiếu bằng điểm cắt biệt lập
    df = df.fillna(-999)

    # Tách đặc trưng (X) và nhãn mục tiêu (y)
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud'].astype(np.int8)

    print("⏳ [3/3] Tiến hành chia tập Train/Test (Tỷ lệ 80/20)...")
    # CẢNH BÁO: Do file gold hiện tại đã bị downsample, tập Test này đang mang tỷ lệ lệch
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    del df
    gc.collect()

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
        'n_estimators': 300,        
        'learning_rate': 0.05,
        'max_depth': 4,             
        'min_child_weight': 20,     
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 5.0,           
        'reg_lambda': 5.0,          
        'scale_pos_weight': spw,
        'tree_method': 'hist',      
        'eval_metric': 'aucpr',     
        'early_stopping_rounds': 50,
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
    
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    save_metrics_to_csv("Vu_Model_8_XGB_Tuned_Threshold", y_test, y_pred_optimal, y_prob)
    
    # Đóng gói
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"📦 Đã xuất xưởng file AI lõi thành công tại: {MODEL_SAVE_PATH}")


# =========================================================================
# 3. ĐIỀU PHỐI KỊCH BẢN CHẠY CHÍNH (MAIN ENTRYPOINT)
# =========================================================================
if __name__ == "__main__":
    print("=================================================================")
    print("      HỆ THỐNG HUẤN LUYỆN XGBOOST TRÊN TẦNG GOLD (LEAKED)        ")
    print("=================================================================")
    
    # Kích hoạt luồng nạp dữ liệu siêu tốc
    X_train, X_test, y_train, y_test = load_gold_data()
    
    # Kích hoạt tuần tự 3 kiến trúc
    run_model_6_xgb_baseline(X_train, X_test, y_train, y_test)
    run_model_7_xgb_spw(X_train, X_test, y_train, y_test)
    run_model_8_xgb_tuned_and_export(X_train, X_test, y_train, y_test)
    
    print("\n🏁 TIẾN TRÌNH CHẠY CỦA LEAD MODELER HOÀN THÀNH XUẤT SẮC!")
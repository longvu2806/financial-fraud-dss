import os
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_recall_curve, auc, classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # Hỗ trợ Scikit-learn >= 1.6 cho CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator 
except ImportError:
    FrozenEstimator = None

# ========================================================
# CẤU HÌNH ĐƯỜNG DẪN DỰ ÁN
# ========================================================
DATA_PATH = r"D:\DSS_CK\financial-fraud-dss\data\featured_transactions_data1.csv"
RESULTS_DIR = r"D:\DSS_CK\financial-fraud-dss\results"

# ========================================================
# HÀM LƯU KẾT QUẢ THEO CHUẨN (Từ base_model.py)
# ========================================================
def save_metrics_to_csv(model_name, y_true, y_pred, y_prob):
    """
    Hàm chuẩn hóa đầu ra giúp cả 4 thành viên ghi nhận kết quả 8 mô hình giống hệt nhau.
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    # Tính toán các chỉ số theo chuẩn sklearn
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob)
    
    df_metrics = pd.DataFrame([{
        'Model_Name': model_name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1_Score': round(f1, 4),
        'PR_AUC': round(pr_auc, 4)
    }])
    
    # Tên mô hình có tiền tố (ví dụ: Vu_XGBoost_v8)
    user_prefix = model_name.split('_')[0].lower()
    user_log = f"results_{user_prefix}.csv"
    file_path = os.path.join(RESULTS_DIR, user_log)
    
    # Ghi nối tiếp vào file
    if os.path.exists(file_path):
        df_metrics.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df_metrics.to_csv(file_path, index=False)
        
    print(f"✅ Đã lưu chỉ số của [{model_name}] vào {file_path}")

# ========================================================
# BƯỚC 1 & 2: NẠP VÀ CHUẨN BỊ DỮ LIỆU
# ========================================================
def step1_load_normal(file_path):
    print("--- BƯỚC 1: NẠP DỮ LIỆU ĐÃ FEATURE ENGINEERING ---")
    df = pd.read_csv(file_path)
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    print(f" -> TỔNG CỘNG nạp vào RAM: {len(df):,} dòng.")
    return df

def step2_prepare_data_by_time(df):
    print("\n--- BƯỚC 2: CHUẨN BỊ DỮ LIỆU (TRAIN - VALIDATION - TEST) ---")
    cat_cols = [col for col in df.select_dtypes(include=['object', 'string']).columns if col != 'set_type']
    
    if len(cat_cols) > 0:
        for col in cat_cols:
            df[col] = df[col].astype('category').cat.codes

    train_mask = df['set_type'] == 'train'
    test_mask = df['set_type'] == 'test'

    y_train = df.loc[train_mask, 'is_fraud'].astype(np.int8)
    y_test_full = df.loc[test_mask, 'is_fraud'].astype(np.int8)

    cols_to_drop = [c for c in df.columns if c.endswith('_id') or c in ['id', 'transaction_id', 'date', 'acct_open_date', 'set_type']]
    
    X_train = df.loc[train_mask].drop(columns=cols_to_drop + ['is_fraud'], errors='ignore')
    X_test_full = df.loc[test_mask].drop(columns=cols_to_drop + ['is_fraud'], errors='ignore')

    split_val_idx = int(len(X_test_full) * 0.5)
    X_val, y_val = X_test_full.iloc[:split_val_idx], y_test_full.iloc[:split_val_idx]
    X_test, y_test = X_test_full.iloc[split_val_idx:], y_test_full.iloc[split_val_idx:]

    print(f" -> Train: {len(X_train):,} dòng | Gian lận: {sum(y_train == 1)}")
    print(f" -> Val:   {len(X_val):,} dòng | Gian lận: {sum(y_val == 1)}")
    print(f" -> Test:  {len(X_test):,} dòng | Gian lận: {sum(y_test == 1)}")

    del df, X_test_full, y_test_full; gc.collect()
    return X_train, X_val, X_test, y_train, y_val, y_test

# ========================================================
# BƯỚC 3: HUẤN LUYỆN XGBOOST + HIỆU CHỈNH XÁC SUẤT
# ========================================================
def step3_train_calibrated_xgboost(X_train, X_val, y_train, y_val):
    print("\n--- BƯỚC 3: HUẤN LUYỆN XGBOOST + HIỆU CHỈNH XÁC SUẤT ---")
    my_scale_weight = 3.0 
    print(f" -> Cố định scale_pos_weight ở mức: {my_scale_weight}")

    xgb_params = {
        'n_estimators': 800,
        'learning_rate': 0.05,
        'max_depth': 4,
        'min_child_weight': 50,
        'gamma': 1.0,
        'subsample': 0.8,
        'colsample_bytree': 0.4,
        'reg_alpha': 5.0,
        'reg_lambda': 5.0,
        'scale_pos_weight': my_scale_weight,
        'tree_method': 'hist',
        'eval_metric': 'aucpr',
        'early_stopping_rounds': 50,
        'random_state': 42,
        'n_jobs': -1
    }

    base_model = xgb.XGBClassifier(**xgb_params)
    print(" -> Bắt đầu huấn luyện mô hình nền với Early Stopping...")
    base_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50
    )

    print(" -> Áp dụng Platt Scaling trên tập Validation để nắn xác suất...")
    if FrozenEstimator is not None:
        calibrated_model = CalibratedClassifierCV(
            estimator=FrozenEstimator(base_model),
            method='sigmoid'
        )
    else:
        calibrated_model = CalibratedClassifierCV(
            estimator=base_model,
            method='sigmoid',
            cv='prefit'
        )

    calibrated_model.fit(X_val, y_val)
    print(" -> Hiệu chỉnh hoàn tất!")
    return calibrated_model

# ========================================================
# BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH VÀ GHI LOG (CÓ TÍCH HỢP BASE_MODEL)
# ========================================================
def step4_evaluate(model, model_name, X_train, X_val, X_test, y_train, y_val, y_test):
    print("\n--- BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH ---")

    # 1. Dự đoán xác suất cho các tập
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # 2. TÌM NGƯỠNG TRÊN TẬP VALIDATION (QUAN TRỌNG)
    p_val, r_val, thresholds_val = precision_recall_curve(y_val, y_val_proba)
    f1_scores_val = 2 * (p_val * r_val) / (p_val + r_val + 1e-10)
    optimal_idx = np.argmax(f1_scores_val)
    optimal_threshold = thresholds_val[optimal_idx] if optimal_idx < len(thresholds_val) else 0.5

    print(f"\n=> Ngưỡng cảnh báo tối ưu (Tìm từ tập Validation): {optimal_threshold:.4f}")

    # 3. ĐÁNH GIÁ TRÊN TẬP TEST BẰNG NGƯỠNG ĐÃ TÌM ĐƯỢC
    p_test, r_test, _ = precision_recall_curve(y_test, y_test_proba)
    pr_auc_test = auc(r_test, p_test)
    print(f" -> [ĐIỂM SỐ] PR-AUC (Tập Test) : {pr_auc_test:.4f}")

    y_pred_optimal = (y_test_proba >= optimal_threshold).astype(int)

    # Báo cáo phân loại
    print("\nBÁO CÁO PHÂN LOẠI (TRÊN TẬP TEST ĐỘC LẬP):")
    print(classification_report(y_test, y_pred_optimal))

    # ==========================================
    # LƯU KẾT QUẢ RA FILE THEO CHUẨN BASE_MODEL
    # ==========================================
    save_metrics_to_csv(model_name, y_test, y_pred_optimal, y_test_proba)

    # ==========================================
    # ĐỒ HỌA TRỰC QUAN
    # ==========================================
    # Vẽ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_optimal)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (Actual)')
    plt.title(f'Confusion Matrix (Threshold = {optimal_threshold:.2f})')
    plt.show()

    # Xử lý an toàn cho Feature Importance
    try:
        if hasattr(model, 'calibrated_classifiers_'):
            inner_model = model.calibrated_classifiers_[0].estimator
            if hasattr(inner_model, 'estimator'):
                importance = inner_model.estimator.feature_importances_
            else:
                importance = inner_model.feature_importances_
        else:
            importance = model.feature_importances_

        feat_imp = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})
        feat_imp = feat_imp.sort_values(by='Importance', ascending=False).head(20)

        plt.figure(figsize=(10, 6))
        plt.barh(feat_imp['Feature'][::-1], feat_imp['Importance'][::-1], color='teal')
        plt.title("Top 20 Đặc trưng quyết định Gian lận (XGBoost)")
        plt.xlabel("Trọng số quan trọng")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"\n[Cảnh báo] Không thể vẽ biểu đồ Feature Importance: {e}")

# ========================================================
# MAIN TRÌNH ĐIỀU KHIỂN
# ========================================================
if __name__ == "__main__":
    df_raw = step1_load_normal(DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = step2_prepare_data_by_time(df_raw)
    
    model_calibrated = step3_train_calibrated_xgboost(X_train, X_val, y_train, y_val)
    
    # Đánh giá và lưu tự động vào results_vu.csv
    # Lưu ý: model_name bắt đầu bằng "Vu_" để hệ thống regex đúng thư mục.
    step4_evaluate(
        model=model_calibrated, 
        model_name="Vu_XGBoost_Tuned_v8", 
        X_train=X_train, X_val=X_val, X_test=X_test, 
        y_train=y_train, y_val=y_val, y_test=y_test
    )
import os
import json
import pickle
import pandas as pd

# ==========================================
# 1. KHỞI TẠO HỆ THỐNG (Chỉ chạy 1 lần khi bật App)
# ==========================================
MODEL_PATH = r"D:\DSS_CK\financial-fraud-dss\saved_models\xgboost_v8.pkl"
CONFIG_PATH = r"D:\DSS_CK\financial-fraud-dss\saved_models\xgboost_v8_config.json" # Đường dẫn tới file JSON của bạn

print("-> Đang khởi động hệ thống lõi (Backend DSS)...")

# Nạp bộ não XGBoost
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Nạp cấu hình điểm chuẩn
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# Rút trích đúng cái ngưỡng Optimal Threshold từ JSON
OPTIMAL_THRESHOLD = config["optimal_threshold"]
EXPECTED_PRECISION = config["precision_expected"]

print(f"[OK] Đã nạp mô hình: {config['model_version']}")
print(f"[OK] Ngưỡng cảnh báo đỏ: {OPTIMAL_THRESHOLD:.4f} (Độ chính xác kỳ vọng: {EXPECTED_PRECISION*100:.1f}%)")

# ==========================================
# 2. HÀM DỰ ĐOÁN THỜI GIAN THỰC (Dùng cho giao diện Streamlit)
# ==========================================
def detect_fraud_realtime(new_transaction_df):
    """
    Hàm này nhận vào 1 dòng dữ liệu giao dịch mới (đã qua lớp Feature Engineering)
    và trả về quyết định Khóa thẻ hay Cho phép qua.
    """
    
    # 1. Mô hình dự đoán xác suất rủi ro
    # Lấy cột số 1 (xác suất là class 1 - Gian lận)
    risk_probability = model.predict_proba(new_transaction_df)[:, 1]
    
    # 2. Đưa ra phán quyết dựa trên cấu hình JSON
    # Nếu xác suất >= 0.9895 -> Báo động đỏ (Trả về 1)
    # Ngược lại -> Giao dịch an toàn (Trả về 0)
    is_fraud = (risk_probability >= OPTIMAL_THRESHOLD).astype(int)
    
    return is_fraud[0], risk_probability[0]

# ==========================================
# KỊCH BẢN TEST THỬ (MÔ PHỎNG LUỒNG STREAMING CỦA HIẾU)
# ==========================================
if __name__ == "__main__":
    # Giả lập: Lấy ngẫu nhiên 1 dòng dữ liệu từ file test_stream.csv của Cường
    print("\n--- MÔ PHỎNG CÓ GIAO DỊCH MỚI QUẸT THẺ ---")
    
    # Đọc 1 dòng dữ liệu (Nhớ bỏ cột is_fraud vì thực tế làm gì biết trước nhãn)
    sample_data = pd.read_csv(r"D:\DSS_CK\financial-fraud-dss\results\test_stream.csv").drop(columns=['is_fraud'], errors='ignore').iloc[[0]]
    
    # Đưa vào hàm lõi để quét
    decision, prob = detect_fraud_realtime(sample_data)
    
    # In ra màn hình cho Điều tra viên xem
    if decision == 1:
        print(f"🚨 CẢNH BÁO ĐỎ: Phát hiện gian lận! (Độ rủi ro: {prob:.4f} >= {OPTIMAL_THRESHOLD:.4f})")
        print("-> Lệnh hệ thống: KHÓA THẺ NGAY LẬP TỨC.")
    else:
        print(f"✅ GIAO DỊCH AN TOÀN. (Độ rủi ro: {prob:.4f} < {OPTIMAL_THRESHOLD:.4f})")
        print("-> Lệnh hệ thống: PHÊ DUYỆT GIAO DỊCH.")
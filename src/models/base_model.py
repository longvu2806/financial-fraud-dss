import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score

RESULTS_DIR = "./results"

def save_metrics_to_csv(model_name, y_true, y_pred, y_prob):
    """
    Hàm chuẩn hóa đầu ra giúp cả 4 thành viên ghi nhận kết quả 8 mô hình giống hệt nhau.
    """
    # 1. Đảm bảo thư mục kết quả tồn tại
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    # 2. Tính toán các chỉ số theo chuẩn sklearn
    # Sử dụng average_precision_score cho PR-AUC vì lớp gian lận cực đoan (~0.8%)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob)
    
    # 3. Tạo DataFrame cho mô hình hiện tại
    df_metrics = pd.DataFrame([{
        'Model_Name': model_name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1_Score': round(f1, 4),
        'PR_AUC': round(pr_auc, 4)
    }])
    
    # 4. Định dạng tên file log dựa theo tên người code để tuyệt đối không bị xung đột Git khi push
    # Tên mô hình truyền vào nên có tiền tố tên người, ví dụ: "Vu_XGBoost_v8"
    user_prefix = model_name.split('_')[0].lower()
    user_log = f"results_{user_prefix}.csv"
    file_path = os.path.join(RESULTS_DIR, user_log)
    
    # Ghi nối tiếp (append) vào file của từng người
    if os.path.exists(file_path):
        df_metrics.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df_metrics.to_csv(file_path, index=False)
        
    print(f"✅ Đã lưu chỉ số của [{model_name}] vào {file_path}")
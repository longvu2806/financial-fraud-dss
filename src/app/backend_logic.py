"""
backend_logic.py
-----------------
Phụ trách: Vũ (Backend Logic & Tích hợp)

Chứa các hàm "hậu trường" cho Streamlit DSS:
  - load_model(): nạp xgboost_v8.pkl lên RAM. Nếu file chưa tồn tại
    (Vũ chưa train xong), tự động dùng MockFraudModel để cả team
    vẫn chạy được giao diện song song, không ai bị block.
  - get_feature_columns(): danh sách cột feature đúng thứ tự đưa vào model.
  - predict_proba_for_df(): chạy model trên một DataFrame, trả về xác suất gian lận.
  - calculate_roi(): logic toán học cho thanh trượt Threshold — đổi ngưỡng
    (vd 0.5 -> 0.9757) thì số tiền cứu được / số ca khóa oan thay đổi thế nào.

Khi Vũ train xong xgboost_v8.pkl và đặt vào saved_models/, code KHÔNG cần
sửa gì cả — load_model() sẽ tự động ưu tiên dùng model thật.
"""

import os
import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Cấu hình đường dẫn
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "xgboost_v8.pkl")

# Các cột KHÔNG phải feature (id, nhãn thật) -> loại ra trước khi đưa vào model
NON_FEATURE_COLUMNS = ["transaction_id", "is_fraud"]


# ---------------------------------------------------------------------------
# Mock model — dùng tạm khi Vũ chưa train xong xgboost_v8.pkl
# ---------------------------------------------------------------------------
class MockFraudModel:
    """
    Model giả lập có cùng interface với XGBoost (.predict_proba)
    để toàn bộ giao diện Streamlit chạy được ngay, không phải chờ
    model thật. Logic: kết hợp vài cờ rủi ro có sẵn trong feature
    (flag_amount_spike, flag_hot_merchant, flag_chip_bypass, v.v.)
    cộng thêm nhiễu ngẫu nhiên có seed cố định -> demo ổn định,
    F1/PR-AUC không có ý nghĩa thật, CHỈ để demo giao diện.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.is_mock = True

    def predict_proba(self, X: pd.DataFrame):
        risk_cols = [
            "flag_amount_spike", "flag_hot_merchant", "flag_chip_bypass",
            "flag_dark_web_card", "flag_burst_1h", "flag_rapid_sequence",
            "flag_near_credit_limit", "flag_high_mcc_risk", "flag_state_hop",
            "flag_new_card", "flag_pin_stale", "flag_debt_pressure",
        ]
        present = [c for c in risk_cols if c in X.columns]

        if present:
            risk_score = X[present].fillna(0).sum(axis=1) / max(len(present), 1)
        else:
            risk_score = pd.Series(np.zeros(len(X)), index=X.index)

        noise = self.rng.normal(loc=0.0, scale=0.08, size=len(X))
        raw = risk_score.to_numpy() * 1.6 + noise
        proba_fraud = 1 / (1 + np.exp(-(raw * 6 - 1.5)))  # ép về [0,1] dạng sigmoid
        proba_fraud = np.clip(proba_fraud, 0.001, 0.999)

        return np.column_stack([1 - proba_fraud, proba_fraud])


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_model(model_path: str = MODEL_PATH):
    """
    Nạp model XGBoost đã train (xgboost_v8.pkl) lên RAM.
    Nếu file không tồn tại -> trả về MockFraudModel kèm cờ is_mock=True
    để giao diện hiển thị cảnh báo "đang dùng dữ liệu giả lập".
    """
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        model.is_mock = False
        return model

    return MockFraudModel()


def get_feature_columns(df: pd.DataFrame) -> list:
    """Trả về danh sách cột feature thật (loại transaction_id, is_fraud)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hoá DataFrame trước khi đưa vào model:
      - chỉ giữ cột feature
      - XGBoost/LightGBM chịu được NaN nên KHÔNG fillna ở đây để giữ
        đúng hành vi model thật; nhưng MockFraudModel tự fillna(0) nội bộ.
    """
    feature_cols = get_feature_columns(df)
    return df[feature_cols]


def predict_proba_for_df(model, df: pd.DataFrame) -> np.ndarray:
    """
    Chạy model trên DataFrame, trả về mảng xác suất gian lận (lớp 1)
    cho từng dòng.
    """
    X = prepare_features(df)
    proba = model.predict_proba(X)
    return proba[:, 1]


# ---------------------------------------------------------------------------
# Logic Threshold / ROI cho Tab Giám đốc
# ---------------------------------------------------------------------------
def calculate_roi(
    df: pd.DataFrame,
    fraud_proba: np.ndarray,
    threshold: float,
    review_cost_per_case: float = 5.0,
) -> dict:
    """
    Tính toán tác động tài chính khi đổi ngưỡng (threshold) phân loại gian lận.

    Tham số:
        df: DataFrame gốc, phải có cột 'amount' và 'is_fraud' (nhãn thật).
        fraud_proba: xác suất gian lận model dự đoán cho từng dòng (cùng thứ tự với df).
        threshold: ngưỡng quyết định (0.0 - 1.0). >= threshold -> bị chặn/điều tra.
        review_cost_per_case: chi phí vận hành ước tính cho mỗi ca bị gắn cờ điều tra
            (nhân sự, thời gian xử lý), dùng để tính chi phí ròng.

    Trả về dict gồm:
        flagged_count: tổng số ca bị gắn cờ tại threshold này
        true_positive / false_positive / false_negative / true_negative
        money_saved: tổng số tiền gian lận thực sự bị chặn đứng (TP)
        money_missed: tổng số tiền gian lận bị bỏ lọt (FN)
        innocent_blocked: số ca khách hàng tốt bị khóa oan (FP)
        review_cost: chi phí vận hành để xử lý các ca bị gắn cờ
        net_benefit: money_saved - innocent_blocked_cost - review_cost (lợi ích ròng)
        precision / recall / f1: chỉ số tại threshold hiện tại
    """
    amounts = df["amount"].to_numpy() if "amount" in df.columns else np.zeros(len(df))
    y_true = df["is_fraud"].to_numpy() if "is_fraud" in df.columns else np.zeros(len(df))

    y_pred = (fraud_proba >= threshold).astype(int)

    tp_mask = (y_pred == 1) & (y_true == 1)
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)
    tn_mask = (y_pred == 0) & (y_true == 0)

    tp, fp, fn, tn = tp_mask.sum(), fp_mask.sum(), fn_mask.sum(), tn_mask.sum()

    money_saved = amounts[tp_mask].sum() if tp > 0 else 0.0
    money_missed = amounts[fn_mask].sum() if fn > 0 else 0.0

    flagged_count = int(tp + fp)
    review_cost = flagged_count * review_cost_per_case

    # Giả định: khách bị khóa oan gây thiệt hại "mềm" cho ngân hàng
    # (mất lòng tin / chi phí xin lỗi), ước tính bằng % giá trị giao dịch của họ.
    innocent_friction_cost = amounts[fp_mask].sum() * 0.02 if fp > 0 else 0.0

    net_benefit = money_saved - innocent_friction_cost - review_cost

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "flagged_count": flagged_count,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "money_saved": float(money_saved),
        "money_missed": float(money_missed),
        "innocent_blocked": int(fp),
        "review_cost": float(review_cost),
        "net_benefit": float(net_benefit),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def roi_curve(df: pd.DataFrame, fraud_proba: np.ndarray, n_points: int = 50) -> pd.DataFrame:
    """
    Quét qua nhiều threshold (0.01 -> 0.99) để vẽ biểu đồ trade-off
    giữa Tiền cứu được và Số ca khóa oan (dùng cho Tab Giám đốc của Hiệp).
    """
    thresholds = np.linspace(0.01, 0.99, n_points)
    rows = []
    for t in thresholds:
        r = calculate_roi(df, fraud_proba, t)
        r["threshold"] = float(t)
        rows.append(r)
    return pd.DataFrame(rows)
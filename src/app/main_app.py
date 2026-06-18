"""
main_app.py
------------
File chạy chính của giao diện DSS (Hiếu & Cường ghép vào đây).

Chạy bằng:
    streamlit run src/app/main_app.py

Trách nhiệm:
  - Set page config (tên app, icon, layout wide).
  - Load model (qua backend_logic, tự fallback sang mock nếu chưa có .pkl).
  - Load test_stream.csv (qua đường dẫn tương đối tới data/).
  - Hiển thị trạng thái kết nối DB (Cường phụ trách phần psycopg2 thật,
    ở đây gọi qua db_status() để main_app không cần biết chi tiết).
  - Dựng 2 tab: "📊 Giám đốc" (tab_director) và "🕵️ Điều tra viên" (tab_investigator).

Dark mode được set qua .streamlit/config.toml (xem file cùng cấp),
không set bằng code ở đây.
"""

import os
import sys
import streamlit as st
import pandas as pd

# Cho phép import các module cùng thư mục (backend_logic, tab_director, tab_investigator)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend_logic import load_model
from tab_director import render_director_tab
from tab_investigator import render_investigator_tab

try:
    from db_status import get_db_status
except ImportError:
    # Cường chưa code xong phần psycopg2 -> dùng hàm giả lập tạm
    def get_db_status():
        return {"connected": False, "row_count": None, "message": "Module db_status chưa sẵn sàng."}


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "test_stream.csv")


st.set_page_config(
    page_title="Financial Fraud DSS",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_stream_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main():
    st.title("🛡️ Financial Fraud Detection — DSS")
    st.caption("Hệ thống hỗ trợ ra quyết định phát hiện gian lận tài chính")

    # ---------------------------------------------------------------
    # Header: trạng thái DB (việc của Cường)
    # ---------------------------------------------------------------
    db_status = get_db_status()
    if db_status.get("connected"):
        st.caption(f"🟢 Trạng thái DB: Đã kết nối **{db_status['row_count']:,} dòng**")
    else:
        st.caption(f"🔴 Trạng thái DB: Chưa kết nối ({db_status.get('message', '')})")

    # ---------------------------------------------------------------
    # Load model + dữ liệu stream
    # ---------------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        st.error(
            f"Không tìm thấy file dữ liệu tại `{DATA_PATH}`. "
            "Hãy đảm bảo test_stream.csv đã được Cường đặt vào thư mục data/."
        )
        st.stop()

    model = get_model()
    df = get_stream_data(DATA_PATH)

    # ---------------------------------------------------------------
    # 2 Tab chính
    # ---------------------------------------------------------------
    tab1, tab2 = st.tabs(["📊 Giám đốc", "🕵️ Điều tra viên"])

    with tab1:
        from backend_logic import predict_proba_for_df
        fraud_proba = predict_proba_for_df(model, df)
        render_director_tab(df, fraud_proba, model=model)

    with tab2:
        speed = st.slider(
            "Tốc độ luồng (giao dịch / giây)",
            min_value=0.5, max_value=10.0, value=2.0, step=0.5,
            key="stream_speed",
        )
        render_investigator_tab(df, model, speed=speed)


if __name__ == "__main__":
    main()
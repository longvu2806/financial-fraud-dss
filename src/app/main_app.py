# main_app.py nâng cấp
import os
import sys
import streamlit as st
import pandas as pd

# Cho phép import các module từ thư mục hiện tại và thư mục con tabs/
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from backend_logic import load_model, predict_proba_for_df
from tab_director import render_director_tab
from tab_investigator import render_investigator_tab

# --- IMPORT 4 TAB INSIGHT MỚI CỦA HIỆP ---
from tab_executive import render_tab_1
from tab_behavior import render_tab_2
from tab_customer import render_tab_3
from tab_geography import render_tab_4

try:
    from db_status import get_db_status
except ImportError:
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
    st.title("🛡️ Financial Fraud Detection — Decision Support System")
    
    # ---------------------------------------------------------------
    # Header: trạng thái DB (Cường phụ trách)
    # ---------------------------------------------------------------
    db_status = get_db_status()
    if db_status.get("connected"):
        st.success(f"🟢 Trạng thái DB: Đã kết nối **{db_status['row_count']:,} dòng**")
    else:
        st.warning(f"🔴 Trạng thái DB: Chưa kết nối ({db_status.get('message', '')})")

    # ---------------------------------------------------------------
    # Load model + dữ liệu stream
    # ---------------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        st.error(f"Không tìm thấy file dữ liệu tại `{DATA_PATH}`.")
        st.stop()

    model = get_model()
    df = get_stream_data(DATA_PATH)

    # ---------------------------------------------------------------
    # PHÂN CHIA 2 TAB CHÍNH CỦA HỆ THỐNG
    # ---------------------------------------------------------------
    tab_management, tab_live = st.tabs(["📊 Phân tích chiến lược (Giám đốc)", "🕵️ Giám sát trực tiếp (Điều tra viên)"])

    # --- TAB 1: DÀNH CHO GIÁM ĐỐC (CHỨA 4 INSIGHTS CHUYÊN SÂU) ---
    with tab_management:
        st.write("### Bảng điều khiển phân tích rủi ro hệ thống")
        
        # Tích hợp 4 Tab Insight của Hiệp vào làm Sub-tabs
        sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
            "📈 KPI Tổng quan", 
            "🕵️ Hành vi gian lận", 
            "👥 Hồ sơ khách hàng", 
            "🌍 Bản đồ & Ngành hàng",
            "🎯 Giả lập ROI (Vũ)"
        ])
        
        with sub_tab1:
            render_tab_1() # Tab 1 chuyên sâu
        with sub_tab2:
            render_tab_2() # Tab 2 chuyên sâu
        with sub_tab3:
            render_tab_3() # Tab 3 chuyên sâu
        with sub_tab4:
            render_tab_4() # Tab 4 chuyên sâu
        with sub_tab5:
            # Đây là phần logic cũ của Vũ để sếp kéo thanh trượt Threshold
            fraud_proba = predict_proba_for_df(model, df)
            render_director_tab(df, fraud_proba, model=model)

    # --- TAB 2: DÀNH CHO ĐIỀU TRA VIÊN (LIVE STREAMING) ---
    with tab_live:
        col_ctrl1, col_ctrl2 = st.columns([1, 3])
        with col_ctrl1:
            speed = st.slider(
                "Tốc độ luồng (giao dịch / giây)",
                min_value=0.5, max_value=5.0, value=1.0, step=0.5,
                key="stream_speed",
            )
        render_investigator_tab(df, model, speed=speed)

if __name__ == "__main__":
    main()
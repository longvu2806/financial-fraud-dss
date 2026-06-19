import os
import sys
import streamlit as st
import pandas as pd

# Cho phép import các module từ thư mục hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# --- CODE CỦA ĐỒNG ĐỘI (Giữ nguyên tuyệt đối để tránh Git Conflict) ---
from backend_logic import load_model, predict_proba_for_df
from tab_director import render_director_tab
from tab_investigator import render_investigator_tab

try:
    from db_status import get_db_status
except ImportError:
    def get_db_status():
        return {"connected": False, "row_count": None, "message": "Module db_status chưa sẵn sàng."}

# --- IMPORT CÁC TAB CỦA HIỆP ---
from tab_executive import render_tab_1
from tab_behavior import render_tab_2
from tab_customer import render_tab_3
from tab_geography import render_tab_4
from tab_spending import render_tab_5
from tab_segmentation import render_tab_6
from tab_financial import render_tab_7
from tab_operations import render_tab_8

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
    # Header: Trạng thái DB (Của Cường)
    # ---------------------------------------------------------------
    db_status = get_db_status()
    if db_status.get("connected"):
        st.success(f"🟢 Trạng thái DB: Đã kết nối **{db_status['row_count']:,} dòng**")
    else:
        st.warning(f"🔴 Trạng thái DB: Chưa kết nối ({db_status.get('message', '')})")

    # ---------------------------------------------------------------
    # Load model + dữ liệu stream (Của Vũ)
    # ---------------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        st.error(f"Không tìm thấy file dữ liệu tại `{DATA_PATH}`.")
        st.stop()

    model = get_model()
    df = get_stream_data(DATA_PATH)

    # ---------------------------------------------------------------
    # PHÂN CHIA 3 TAB CHÍNH CỦA HỆ THỐNG
    # ---------------------------------------------------------------
    tab_management, tab_business, tab_live = st.tabs([
        "🛡️ Quản trị Rủi ro (Risk)", 
        "💼 Phát triển Kinh doanh (Business)", 
        "🕵️ Giám sát Live"
    ])

    # --- TAB CHÍNH 1: QUẢN TRỊ RỦI RO (HIỆP & VŨ) ---
    with tab_management:
        st.write("### Bảng điều khiển phân tích rủi ro hệ thống")
        
        sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
            "📈 KPI Tổng quan", 
            "🕵️ Hành vi gian lận", 
            "👥 Hồ sơ khách hàng", 
            "🌍 Bản đồ & Ngành hàng",
            "🎯 Giả lập ROI (Vũ)"
        ])
        
        with sub_tab1: render_tab_1() 
        with sub_tab2: render_tab_2() 
        with sub_tab3: render_tab_3() 
        with sub_tab4: render_tab_4() 
        with sub_tab5:
            # Logic của Vũ được giữ nguyên vẹn
            fraud_proba = predict_proba_for_df(model, df)
            render_director_tab(df, fraud_proba, model=model)

    # --- TAB CHÍNH 2: PHÁT TRIỂN KINH DOANH (HIỆP) ---
    with tab_business:
        st.write("### Bảng điều khiển phân tích và tăng trưởng kinh doanh")
        
        # Gọi 4 Tab mới mà chúng ta vừa tạo
        sub_b1, sub_b2, sub_b3, sub_b4 = st.tabs([
            "📈 Xu hướng Chi tiêu", 
            "🎯 Phân khúc Khách", 
            "💎 Sức khỏe Tài chính", 
            "⚙️ Trải nghiệm (Lỗi)"
        ])
        with sub_b1: render_tab_5()
        with sub_b2: render_tab_6()
        with sub_b3: render_tab_7()
        with sub_b4: render_tab_8()

    # --- TAB CHÍNH 3: DÀNH CHO ĐIỀU TRA VIÊN (HIẾU) ---
    with tab_live:
        col_ctrl1, col_ctrl2 = st.columns([1, 3])
        with col_ctrl1:
            speed = st.slider(
                "Tốc độ luồng (giao dịch / giây)",
                min_value=0.5, max_value=5.0, value=1.0, step=0.5,
                key="stream_speed",
            )
        # Logic live streaming của Hiếu được giữ nguyên vẹn
        render_investigator_tab(df, model, speed=speed)

if __name__ == "__main__":
    main()
import streamlit as st
import plotly.express as px
import pandas as pd
from backend_data import FraudDashboardBackend

def render_tab_3():
    backend = FraudDashboardBackend()
    
    st.header("👥 Hồ Sơ Khách Hàng & Rủi Ro Thẻ")
    st.info("Phân tích mối liên hệ giữa hồ sơ tài chính, thông tin thẻ và nguy cơ bị tấn công.")

    col1, col2 = st.columns([1, 2])
    
    # ---------------------------------------------------------
    # 1. BIỂU ĐỒ 100% STACKED BAR: TÁC ĐỘNG DARK WEB
    # ---------------------------------------------------------
    with col1:
        st.subheader("🌐 Cảnh báo Dark Web")
        df_dw_raw = backend.fetch_darkweb_impact()
        
        if not df_dw_raw.empty:
            df_dw_raw['is_fraud'] = df_dw_raw['is_fraud'].astype(int)
            df_dw_raw['Fraud_Label'] = df_dw_raw['is_fraud'].map({0: 'Hợp pháp', 1: 'Gian lận'})
            
            df_dw_raw['DarkWeb_Label'] = df_dw_raw['card_on_dark_web'].astype(str).str.lower().map({
                'no': 'Thẻ An toàn', 
                'yes': 'Bị lộ thông tin'
            }).fillna('Không xác định')
            
            # BƯỚC KHẮC PHỤC LỖI TẠI ĐÂY: Loại bỏ 'barnorm' ra khỏi px.bar
            fig_dw = px.bar(
                df_dw_raw, 
                x="DarkWeb_Label", 
                y="count", 
                color="Fraud_Label", 
                title="Tỉ lệ Gian lận trên nhóm thẻ",
                barmode="stack", # CHỈ ĐỂ STACK Ở ĐÂY
                labels={'DarkWeb_Label': 'Trạng thái bảo mật', 'count': 'Tỉ lệ %', 'Fraud_Label': 'Kết quả'},
                color_discrete_map={"Hợp pháp": "#1f77b4", "Gian lận": "#d62728"}
            )
            
            # SỬ DỤNG LỆNH UPDATE_LAYOUT ĐỂ BIẾN THÀNH 100% MÀ KHÔNG BỊ LỖI
            fig_dw.update_layout(barnorm='percent', yaxis_ticksuffix=" %")
            
            # Bỏ use_container_width đi để không bị vướng cảnh báo
            st.plotly_chart(fig_dw)
        else:
            st.warning("⚠️ Không có dữ liệu Dark Web để hiển thị.")
        
    # ---------------------------------------------------------
    # 2. BIỂU ĐỒ SCATTER: ÁP LỰC TÀI CHÍNH VS GIAN LẬN
    # ---------------------------------------------------------
    with col2:
        st.subheader("🎯 Phân tích First-Party Fraud")
        df_scatter = backend.fetch_debt_vs_credit_sample(limit=5000)
        
        if not df_scatter.empty:
            df_scatter['Status'] = df_scatter['is_fraud'].map({0: 'Hợp pháp', 1: 'Gian lận'})
            
            fig_scatter = px.scatter(
                df_scatter, 
                x="credit_score", 
                y="debt_ratio", 
                color="Status",
                title="Mối liên hệ: Áp lực nợ vs Điểm tín dụng",
                labels={'credit_score': 'Điểm tín dụng', 'debt_ratio': 'Hệ số nợ (Nợ/Thu nhập)'},
                opacity=0.5,
                color_discrete_map={"Hợp pháp": "lightgreen", "Gian lận": "red"}
            )
            
            fig_scatter.add_hline(y=0.8, line_dash="dash", line_color="orange", annotation_text="Vùng rủi ro nợ cao (>80%)")
            st.plotly_chart(fig_scatter)
        else:
            st.warning("⚠️ Không có dữ liệu hồ sơ tài chính.")
        
    # ---------------------------------------------------------
    # 3. BIỂU ĐỒ HISTOGRAM: ĐỘ TUỔI NẠN NHÂN
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("👴 Phân tích Nhân khẩu học: Độ tuổi nạn nhân")
    df_age = backend.fetch_victim_age_dist()
    
    if not df_age.empty:
        df_age = df_age[df_age['user_age'] > 0]
        
        fig_hist = px.histogram(
            df_age, 
            x="user_age", 
            nbins=40, 
            title="Phân phối độ tuổi của các khách hàng bị tấn công (is_fraud = 1)",
            labels={'user_age': 'Tuổi của chủ thẻ'},
            color_discrete_sequence=['#9467bd'], 
            marginal="box" 
        )
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist)
        
        avg_age = df_age['user_age'].mean()
        median_age = df_age['user_age'].median()
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Tuổi trung bình nạn nhân", f"{avg_age:.1f} tuổi")
        m_col2.metric("Tuổi trung vị", f"{int(median_age)} tuổi")
        m_col3.info("💡 **Insight:** Tội phạm thường tập trung vào nhóm tuổi có tích lũy tài chính cao hoặc nhóm người cao tuổi ít am hiểu công nghệ bảo mật.")
    else:
        st.warning("⚠️ Chưa ghi nhận dữ liệu gian lận để phân tích độ tuổi.")
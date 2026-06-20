import streamlit as st
import plotly.express as px
from backend_business import BusinessDashboardBackend

def render_tab_8():
    st.header("⚙️ Phân Tích Lỗi & Trải Nghiệm Khách Hàng")
    backend = BusinessDashboardBackend()
    
    col1, col2 = st.columns(2)
    with col1:
        df_err = backend.fetch_error_breakdown()
        fig1 = px.bar(df_err, x='error_count', y='errors', orientation='h', title="Phân rã Nguyên nhân Lỗi quẹt thẻ")
        fig1.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig1, width='stretch')
        
    with col2:
        df_decline = backend.fetch_decline_rate()
        fig2 = px.bar(df_decline, x='use_chip', y='decline_rate', color='use_chip', title="Tỷ lệ Giao dịch Thất bại (Decline Rate %)")
        st.plotly_chart(fig2, width='stretch')
        
    st.info("💡 **Gợi ý Vận hành:** Nếu 'Insufficient Balance' xuất hiện nhiều, phòng tín dụng nên chạy chiến dịch Upsell nâng hạn mức thẻ. Nếu 'Online' có Decline Rate cao, cần rà soát lại cổng Payment Gateway.")
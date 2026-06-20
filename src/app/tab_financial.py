import streamlit as st
import plotly.express as px
from backend_business import BusinessDashboardBackend

def render_tab_7():
    st.header("💎 Sức khỏe Tài chính Khách hàng & LTV")
    backend = BusinessDashboardBackend()
    
    col1, col2 = st.columns(2)
    with col1:
        df_credit = backend.fetch_credit_score_dist()
        fig1 = px.histogram(df_credit, x="credit_score", nbins=30, title="Phân phối Điểm Tín Dụng (Credit Score)", color_discrete_sequence=['indigo'])
        st.plotly_chart(fig1, width='stretch')
        
    with col2:
        df_util = backend.fetch_utilization_ratio()
        df_util = df_util[df_util['utilization_pct'] < 200] # Lọc nhiễu
        fig2 = px.box(df_util, x='card_type', y='utilization_pct', title="Tỷ lệ sử dụng Hạn mức thẻ (%)", color='card_type')
        st.plotly_chart(fig2, width='stretch')

    col3, col4 = st.columns(2)
    with col3:
        df_debt = backend.fetch_debt_distribution()
        fig3 = px.box(df_debt, y="total_debt", title="Phân tán Cục nợ khách hàng đang gánh ($)", color_discrete_sequence=['brown'])
        st.plotly_chart(fig3, width='stretch')

    with col4:
        df_card = backend.fetch_card_performance()
        fig4 = px.bar(df_card, x='card_brand', y='total_volume', color='card_type', barmode='group', title="Doanh số Quẹt thẻ: Visa vs Mastercard")
        st.plotly_chart(fig4, width='stretch')
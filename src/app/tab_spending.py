import streamlit as st
import plotly.express as px
from backend_business import BusinessDashboardBackend

def render_tab_5():
    st.header("📈 Phân Tích Xu Hướng Chi Tiêu (Spending Trends)")
    backend = BusinessDashboardBackend()
    
    col1, col2 = st.columns(2)
    with col1:
        df_trend = backend.fetch_spending_trends()
        fig1 = px.line(df_trend, x='month', y='total_spent', title="Dòng tiền chi tiêu theo tháng", markers=True)
        st.plotly_chart(fig1, width='stretch')
        
    with col2:
        df_cat = backend.fetch_top_categories()
        fig2 = px.bar(df_cat, x='total_spent', y='merchant_category', orientation='h', title="Top Ngành hàng Đốt tiền")
        fig2.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig2, width='stretch')

    col3, col4 = st.columns(2)
    with col3:
        df_heat = backend.fetch_spending_heatmap()
        day_map = {0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'}
        df_heat['day_name'] = df_heat['day_of_week'].map(day_map)
        df_pivot = df_heat.pivot(index='day_name', columns='tx_hour', values='total_spent').reindex(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        fig3 = px.imshow(df_pivot, title="Bản đồ nhiệt: Lịch trình Cà thẻ (Theo Doanh số)", color_continuous_scale="Greens")
        st.plotly_chart(fig3, width='stretch')

    with col4:
        df_pay = backend.fetch_payment_method_trend()
        fig4 = px.bar(df_pay, x='year', y='total_spent', color='use_chip', title="Xu hướng Phương thức quẹt thẻ", barmode='group')
        st.plotly_chart(fig4, width='stretch')
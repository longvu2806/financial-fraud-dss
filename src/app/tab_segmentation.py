import streamlit as st
import plotly.express as px
from backend_business import BusinessDashboardBackend

def render_tab_6():
    st.header("🎯 Phân Khúc Khách Hàng (Customer Segmentation)")
    backend = BusinessDashboardBackend()
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        df_gen = backend.fetch_generation_spending()
        if not df_gen.empty:
            fig1 = px.pie(df_gen, values='total_spent', names='generation', 
                          title="Cơ cấu Doanh thu theo Thế hệ", hole=0.4)
            st.plotly_chart(fig1, width='stretch')
        else:
            st.warning("⚠️ Chưa có dữ liệu doanh thu.")
            
    with col2:
        df_scatter = backend.fetch_income_vs_spending()
        if not df_scatter.empty:
            fig2 = px.scatter(df_scatter, x='yearly_income', y='avg_yearly_spend', opacity=0.5,
                              title="Thu nhập vs. Mức chi tiêu thẻ", color_discrete_sequence=['#17becf'])
            st.plotly_chart(fig2, width='stretch')
        else:
            st.warning("⚠️ Chưa có dữ liệu thu nhập.")

    st.markdown("---")
    col3, col4 = st.columns([1.5, 1])
    
    with col3:
        df_behavior = backend.fetch_gen_category_behavior()
        if not df_behavior.empty:
            # 1. SỬA LỖI TẠI ĐÂY: Bỏ barnorm ra khỏi px.bar
            fig3 = px.bar(df_behavior, x='generation', y='total_spent', color='merchant_category', 
                          title="Hành vi tiêu dùng theo Thế hệ", barmode="stack")
            
            # 2. ĐƯA LÊN ĐÂY: Dùng update_layout để ép thành 100% an toàn
            fig3.update_layout(barnorm='percent', yaxis_ticksuffix=" %")
            
            st.plotly_chart(fig3, width='stretch')
        else:
            st.warning("⚠️ Chưa có dữ liệu hành vi ngành hàng.")
            
    with col4:
        df_state = backend.fetch_spending_by_state()
        if not df_state.empty:
            fig4 = px.choropleth(df_state, locations='state', locationmode="USA-states", color='total_spent',
                                 scope="usa", hover_data=['users_count'], title="Mật độ Chi tiêu theo Bang")
            st.plotly_chart(fig4, width='stretch')
        else:
            st.warning("⚠️ Chưa có dữ liệu địa lý.")
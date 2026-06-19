import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from backend_data import FraudDashboardBackend

def render_tab_4():
    backend = FraudDashboardBackend()
    st.header("🌍 Phân Tích Địa Lý & Ngành Hàng (MCC)")
    st.info("Phân tích luồng di chuyển của tội phạm và các nhóm ngành hàng mục tiêu.")

    col1, col2 = st.columns([1.5, 1])
    
    # 1. Map Visualization (Dữ liệu thật theo Bang)
    with col1:
        df_map = backend.fetch_fraud_map_data()
        
        # Thêm điều kiện chống lỗi rỗng
        if not df_map.empty:
            # Vẽ bản đồ nhiệt theo vùng
            fig_map = px.choropleth(
                df_map,
                locations='merchant_state', 
                locationmode="USA-states", 
                color='count',
                scope="usa",
                hover_data=['avg_amt'],
                title="📍 Bản đồ Nhiệt các Bang có số vụ Gian lận cao nhất",
                color_continuous_scale="Reds"
            )
            fig_map.update_layout(height=450, margin={"r":0,"t":40,"l":0,"b":0})
            
            # SỬA LỖI CẢNH BÁO STREAMLIT TẠI ĐÂY
            st.plotly_chart(fig_map, width='stretch')
        else:
            st.warning("⚠️ Chưa có dữ liệu địa lý.")

    # 2. Bar Chart: Top 10 MCC (Dữ liệu thật từ JOIN dim_mcc)
    with col2:
        df_mcc = backend.fetch_top_mcc_risk()
        
        if not df_mcc.empty:
            fig_mcc = px.bar(
                df_mcc, 
                x='fraud_count', 
                y='merchant_category', # Tên ngành hàng thật
                orientation='h',
                title="🛒 Top Ngành Hàng Bị Lợi Dụng", 
                color='fraud_count', 
                color_continuous_scale="Reds"
            )
            # Sắp xếp lại cho ngành cao nhất nằm trên cùng
            fig_mcc.update_layout(yaxis={'categoryorder':'total ascending'})
            
            # SỬA LỖI CẢNH BÁO STREAMLIT TẠI ĐÂY
            st.plotly_chart(fig_mcc, width='stretch')
        else:
            st.warning("⚠️ Chưa có dữ liệu ngành hàng.")

    # 3. Sankey Diagram (Dữ liệu thật luồng di chuyển)
    st.markdown("---")
    st.subheader("✈️ Phân tích State Hop: Dòng chảy giao dịch khác bang")
    st.write("Thể hiện luồng tiền: Từ bang cư trú của khách hàng -> Đến bang nơi tội phạm quẹt thẻ.")
    
    df_sankey = backend.fetch_state_hop_flow()

    if not df_sankey.empty:
        # LOGIC ĐẶC BIỆT: Chuyển đổi tên Bang thành ID số cho Sankey
        all_nodes = list(pd.concat([df_sankey['user_state'], df_sankey['merchant_state']]).unique())
        node_map = {name: i for i, name in enumerate(all_nodes)}
        
        fig_sankey = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15,
              thickness = 20,
              line = dict(color = "black", width = 0.5),
              label = all_nodes,
              color = "indianred"
            ),
            link = dict(
              source = df_sankey['user_state'].map(node_map), 
              target = df_sankey['merchant_state'].map(node_map), 
              value =  df_sankey['value']
            ))])
        
        fig_sankey.update_layout(font_size=12, height=500)
        
        # SỬA LỖI CẢNH BÁO STREAMLIT TẠI ĐÂY
        st.plotly_chart(fig_sankey, width='stretch')
    else:
        st.warning("⚠️ Chưa ghi nhận dữ liệu giao dịch khác bang (State Hop).")
import streamlit as st
import plotly.express as px
from backend_data import FraudDashboardBackend 
from backend_logic import calculate_roi # Để tính toán logic thanh trượt của Vũ

def render_tab_1():
    # Khởi tạo backend
    backend = FraudDashboardBackend()
    
    st.header("📊 Báo Cáo Giám Sát Tổng Quan")
    st.info("Dữ liệu được tổng hợp trực tiếp từ PostgreSQL (13.3 triệu giao dịch)")

    # 1. Lấy KPI thật từ Database
    kpis = backend.fetch_executive_kpis()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng Số Giao Dịch", f"{kpis['total_tx']:,}")
    col2.metric("Tổng GT Giao Dịch ($)", f"${kpis['total_amt']:,.0f}")
    
    # Tỷ lệ gian lận tính trên số vụ (Thêm điều kiện chống lỗi chia cho 0)
    fraud_rate = (kpis['fraud_count'] / kpis['total_tx']) * 100 if kpis['total_tx'] > 0 else 0
    col3.metric("Tỉ lệ Gian Lận (%)", f"{fraud_rate:.3f}%")
    col4.metric("Thiệt Hại Thực Tế ($)", f"${kpis['fraud_loss']:,.0f}")

    st.markdown("---")
    
    # 2. Thanh trượt Threshold (Theo Roadmap Vũ & Hiệp)
    st.subheader("🎯 Giả lập Ngưỡng quyết định & Hiệu quả kinh tế (ROI)")
    st.write("Kéo thanh trượt để xem nếu thay đổi độ nhạy của AI thì ngân hàng tiết kiệm được bao nhiêu tiền.")
    
    # Ở đây sếp chọn ngưỡng tối ưu mà chúng ta tìm được là 0.9757
    threshold = st.slider("Điều chỉnh Ngưỡng chặn gian lận (Threshold):", 
                          min_value=0.0, max_value=1.0, value=0.9757, step=0.01)
    
    # Hiển thị kết quả giả lập (Chờ Vũ truyền biến df_test và y_proba vào để mở comment)
    # ROI_data = calculate_roi(df_test, y_proba, threshold)
    # st.success(f"Dự kiến cứu được: ${ROI_data['money_saved']:,.0f} | Khóa nhầm: {ROI_data['innocent_blocked']} khách hàng")

    st.markdown("---")

    col_chart1, col_chart2 = st.columns([2, 1])
    
    # 3. Biểu đồ đường xu hướng (SỬA LỖI CẢNH BÁO STREAMLIT TẠI ĐÂY)
    with col_chart1:
        df_trend = backend.fetch_fraud_trend()
        if not df_trend.empty:
            fig_trend = px.line(df_trend, x='period', y='fraud_amount', markers=True, 
                                title="📉 Xu hướng thiệt hại do gian lận theo thời gian",
                                labels={'period': 'Thời gian', 'fraud_amount': 'Thiệt hại ($)'})
            fig_trend.update_traces(line_color='red', line_width=3)
            # Dùng width='stretch' thay cho use_container_width=True
            st.plotly_chart(fig_trend, width='stretch')
        else:
            st.warning("Chưa có dữ liệu xu hướng.")
        
    # 4. Biểu đồ Donut Loại thẻ (SỬA LỖI CẢNH BÁO STREAMLIT TẠI ĐÂY)
    with col_chart2:
        df_card = backend.fetch_card_type_distribution()
        if not df_card.empty:
            fig_donut = px.pie(df_card, values='fraud_count', names='card_type', hole=0.5,
                               title="💳 Tỉ lệ gian lận theo Loại thẻ",
                               color_discrete_sequence=px.colors.sequential.RdBu)
            fig_donut.update_traces(textinfo='percent+label')
            # Dùng width='stretch' thay cho use_container_width=True
            st.plotly_chart(fig_donut, width='stretch')
        else:
            st.warning("Chưa có dữ liệu loại thẻ.")
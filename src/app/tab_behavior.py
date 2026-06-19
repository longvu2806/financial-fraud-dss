import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

# Import backend thật
from backend_data import FraudDashboardBackend

def render_tab_2():
    backend = FraudDashboardBackend()
    
    st.header("🕵️ Phân Tích Hành Vi Gian Lận (Fraud Pattern)")
    st.info("Khám phá các quy luật tấn công dựa trên thời gian, số tiền và phương thức giao dịch.")

    col1, col2 = st.columns(2)
    
    # 1. Heatmap - Thời gian rủi ro cao từ Database
    with col1:
        df_heat = backend.fetch_heatmap_data()
        
        if not df_heat.empty:
            # [SỬA LỖI LOGIC]: Trong PostgreSQL, DOW trả về 0 = Chủ Nhật, 1 = Thứ 2...
            day_map = {0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'}
            df_heat['day_name'] = df_heat['tx_day_of_week'].map(day_map)
            
            # Pivot dữ liệu để tạo ma trận cho Heatmap
            df_pivot = df_heat.pivot(index='day_name', columns='tx_hour', values='fraud_count')
            # Sắp xếp lại thứ tự các ngày trong tuần cho chuẩn lịch làm việc
            df_pivot = df_pivot.reindex(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            
            fig_heat = px.imshow(
                df_pivot, 
                labels=dict(x="Giờ trong ngày", y="Ngày trong tuần", color="Số vụ gian lận"),
                x=df_pivot.columns,
                y=df_pivot.index,
                title="🔥 Bản đồ nhiệt: Thời điểm tội phạm hoạt động mạnh nhất",
                color_continuous_scale="YlOrRd"
            )
            
            # [SỬA CẢNH BÁO STREAMLIT]
            st.plotly_chart(fig_heat, width='stretch')
        else:
            st.warning("⚠️ Không có dữ liệu thời gian.")
        
    # 2. Boxplot - Phân phối số tiền thực tế
    with col2:
        df_box = backend.fetch_amount_distribution_sample(limit=10000)
        
        if not df_box.empty:
            fig_box = px.box(
                df_box, 
                x="status", 
                y="amount", 
                color="status",
                title="💰 So sánh giá trị giao dịch (Log scale)",
                log_y=True, # Dùng log scale vì tiền giao dịch chênh lệch rất lớn
                labels={'status': 'Trạng thái', 'amount': 'Số tiền ($)'},
                color_discrete_map={"Hợp pháp": "green", "Gian lận": "red"}
            )
            
            # [SỬA CẢNH BÁO STREAMLIT]
            st.plotly_chart(fig_box, width='stretch')
            st.caption("Ghi chú: Trục Y sử dụng thang Logarit để quan sát rõ các giao dịch giá trị nhỏ.")
        else:
            st.warning("⚠️ Không có dữ liệu giao dịch.")

    st.markdown("---")
    col3, col4 = st.columns(2)
    
    # 3. Bar Chart - Tỷ lệ rủi ro theo Phương thức quẹt thẻ
    with col3:
        df_chip = backend.fetch_chip_usage_stats()
        
        if not df_chip.empty:
            fig_chip = px.bar(
                df_chip, 
                x='use_chip', 
                y='fraud_rate', 
                title="🏧 Tỉ lệ rủi ro theo Phương thức thanh toán",
                labels={'use_chip': 'Phương thức', 'fraud_rate': 'Tỉ lệ Gian lận (%)'},
                color='use_chip',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            # [SỬA CẢNH BÁO STREAMLIT]
            st.plotly_chart(fig_chip, width='stretch')
        else:
            st.warning("⚠️ Không có dữ liệu phương thức thanh toán.")
        
    # 4. Bar Chart - Top mã lỗi khi gian lận
    with col4:
        df_err = backend.fetch_error_analysis()
        
        if not df_err.empty:
            fig_err = px.bar(
                df_err, 
                x='count', 
                y='errors', 
                orientation='h', 
                title="⚠️ Top Mã lỗi xuất hiện trong các vụ gian lận",
                labels={'count': 'Số lần xuất hiện', 'errors': 'Loại lỗi'},
                color='count',
                color_continuous_scale="Oranges"
            )
            # Sắp xếp mã lỗi nhiều nhất lên trên
            fig_err.update_layout(yaxis={'categoryorder':'total ascending'})
            
            # [SỬA CẢNH BÁO STREAMLIT]
            st.plotly_chart(fig_err, width='stretch')
        else:
            st.write("✅ Không phát hiện lỗi kỹ thuật đáng kể trong các ca gian lận.")

    # Thêm insight nghiệp vụ cuối trang
    with st.expander("💡 Giải thích nghiệp vụ (Data Scientist Insight)"):
        st.write("""
        - **Heatmap:** Giúp nhận diện 'khung giờ vàng' của tội phạm. Thông thường là từ 1h - 4h sáng khi chủ thẻ ít kiểm tra điện thoại.
        - **Boxplot:** Nếu nến của 'Gian lận' ngắn và nằm thấp, tội phạm đang dùng chiêu bài 'thử thẻ' (Micro-tests). Nếu có nhiều điểm rời rạc ở trên cao, chúng đang cố 'rút cạn' tài khoản.
        - **Payment Method:** Giao dịch 'Online' và 'Swipe' (Thẻ từ) thường có tỷ lệ lừa đảo cao vượt trội so với 'Chip' (Thẻ nhúng) do tính bảo mật thấp hơn.
        """)
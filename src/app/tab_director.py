"""
tab_director.py
-----------------
Phụ trách: Hiệp (biểu đồ) / Cường (ghép layout Tab Giám đốc)

UI cho Tab 1 - "Giám đốc": người ra quyết định kéo thanh trượt Threshold,
xem ngay tác động tài chính: tiền cứu được, số ca khóa oan, lợi ích ròng,
và biểu đồ trade-off Tiền vs Trải nghiệm khách hàng.

Dùng Plotly cho biểu đồ tương tác (hover, zoom) — đúng yêu cầu "dựng các
thẻ số liệu tài chính, biểu đồ Trade-off giữa Tiền và Trải nghiệm khách hàng".
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from backend_logic import calculate_roi, roi_curve

try:
    from db_status import get_db_summary
except ImportError:
    def get_db_summary():
        return {"connected": False, "row_count": None, "message": "Module db_status chưa sẵn sàng."}


def render_director_tab(df: pd.DataFrame, fraud_proba, model=None):
    """
    Render toàn bộ nội dung Tab Giám đốc.

    Tham số:
        df: DataFrame test_stream (đã có cột 'amount', 'is_fraud').
        fraud_proba: mảng xác suất gian lận model dự đoán, cùng thứ tự với df.
        model: model đã load (dùng để lấy optimal_threshold làm giá trị mặc
            định cho thanh trượt, do Vũ quét bằng precision_recall_curve
            và xuất kèm trong xgboost_v8_config.json).
    """
    st.subheader("📊 Bảng điều khiển Giám đốc")
    st.caption(
        "Kéo thanh trượt để thay đổi ngưỡng (threshold) phân loại gian lận. "
        "Ngưỡng thấp -> bắt nhiều gian lận hơn nhưng khóa oan nhiều khách tốt hơn."
    )

    # -----------------------------------------------------------------
    # Bối cảnh toàn hệ thống — số liệu tổng hợp từ DB (toàn bộ dữ liệu
    # thật, không phải chỉ mẫu test_stream.csv đang demo bên dưới).
    # -----------------------------------------------------------------
    db_summary = get_db_summary()
    if db_summary.get("connected") and db_summary.get("row_count"):
        sample_size = len(df)
        db_rows = db_summary["row_count"]
        coverage_pct = (sample_size / db_rows * 100) if db_rows else 0

        with st.container(border=True):
            st.caption("🗄️ Bối cảnh toàn hệ thống (toàn bộ dữ liệu trong Database)")
            dcol1, dcol2, dcol3, dcol4 = st.columns(4)
            dcol1.metric("Tổng giao dịch (DB)", f"{db_rows:,}")

            fraud_count = db_summary.get("fraud_count")
            if fraud_count is not None:
                fraud_rate = (fraud_count / db_rows * 100) if db_rows else 0
                dcol2.metric("Tổng ca gian lận", f"{fraud_count:,}", f"{fraud_rate:.2f}% tổng")

            fraud_amount = db_summary.get("fraud_amount")
            if fraud_amount is not None:
                dcol3.metric("Tổng tiền gian lận", f"${fraud_amount:,.0f}")

            dcol4.metric(
                "Độ phủ mẫu demo",
                f"{coverage_pct:.3f}%",
                help=f"Mẫu test_stream.csv đang dùng để demo chỉ chiếm {sample_size:,}/{db_rows:,} dòng của toàn DB.",
            )
        st.caption(
            "ℹ️ Các thẻ số liệu và biểu đồ bên dưới được tính trên mẫu "
            f"**test_stream.csv** ({sample_size:,} dòng), không phải toàn bộ DB."
        )

    default_threshold = getattr(model, "optimal_threshold", 0.5)
    if getattr(model, "model_version", None):
        st.caption(
            f"🎯 Model: `{model.model_version}` — ngưỡng tối ưu (F1 max) được Vũ "
            f"tìm ra khi train: **{default_threshold:.4f}**"
        )

    # -----------------------------------------------------------------
    # Thanh trượt Threshold
    # -----------------------------------------------------------------
    threshold = st.slider(
        "Ngưỡng quyết định (Threshold)",
        min_value=0.01,
        max_value=0.99,
        value=float(np.clip(default_threshold, 0.01, 0.99)),
        step=0.0001,
        format="%.4f",
        help="Giao dịch có xác suất gian lận >= ngưỡng này sẽ bị chặn / gắn cờ điều tra.",
    )

    result = calculate_roi(df, fraud_proba, threshold)

    # -----------------------------------------------------------------
    # Thẻ số liệu tài chính
    # -----------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "💰 Tiền cứu được",
        f"${result['money_saved']:,.0f}",
        help="Tổng giá trị các giao dịch gian lận thực sự bị chặn đứng (True Positive).",
    )
    col2.metric(
        "🚫 Khách bị khóa oan",
        f"{result['innocent_blocked']:,} ca",
        help="Số khách hàng tốt bị gắn cờ nhầm (False Positive) -> ảnh hưởng trải nghiệm.",
    )
    col3.metric(
        "⚠️ Gian lận bị bỏ lọt",
        f"${result['money_missed']:,.0f}",
        help="Tổng giá trị gian lận không bị phát hiện (False Negative).",
    )
    col4.metric(
        "📈 Lợi ích ròng",
        f"${result['net_benefit']:,.0f}",
        help="Tiền cứu được - chi phí khách bị khóa oan - chi phí vận hành.",
    )

    col5, col6, col7 = st.columns(3)
    col5.metric("Precision", f"{result['precision']*100:.1f}%")
    col6.metric("Recall", f"{result['recall']*100:.1f}%")
    col7.metric("F1-score", f"{result['f1']*100:.1f}%")

    st.divider()

    # -----------------------------------------------------------------
    # Biểu đồ Trade-off: Tiền cứu được vs Khách bị khóa oan theo threshold
    # -----------------------------------------------------------------
    st.markdown("#### Trade-off: Tiền cứu được vs Trải nghiệm khách hàng")

    curve_df = roi_curve(df, fraud_proba, n_points=60)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve_df["threshold"], y=curve_df["money_saved"],
        name="💰 Tiền cứu được ($)", mode="lines",
        line=dict(color="#2ECC71", width=3),
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=curve_df["threshold"], y=curve_df["innocent_blocked"],
        name="🚫 Khách bị khóa oan (ca)", mode="lines",
        line=dict(color="#E74C3C", width=3, dash="dot"),
        yaxis="y2",
    ))
    # Đánh dấu vị trí threshold hiện tại đang chọn
    fig.add_vline(x=threshold, line_width=1.5, line_dash="dash", line_color="#F1C40F")

    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Threshold"),
        yaxis=dict(title="Tiền cứu được ($)", color="#2ECC71"),
        yaxis2=dict(title="Số ca khóa oan", overlaying="y", side="right", color="#E74C3C"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------
    # Biểu đồ phụ: Precision / Recall theo threshold
    # -----------------------------------------------------------------
    with st.expander("Xem thêm: Precision / Recall theo Threshold"):
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=curve_df["threshold"], y=curve_df["precision"],
            name="Precision", line=dict(color="#3498DB", width=2),
        ))
        fig2.add_trace(go.Scatter(
            x=curve_df["threshold"], y=curve_df["recall"],
            name="Recall", line=dict(color="#9B59B6", width=2),
        ))
        fig2.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(title="Threshold"),
            yaxis=dict(title="Score", range=[0, 1]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)
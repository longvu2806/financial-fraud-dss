"""
tab_investigator.py
---------------------
Phụ trách: Hiếu (Frontend - Tab 2)

UI cho Tab 2 - "Điều tra viên": mô phỏng luồng giao dịch chạy real-time
(đọc test_stream.csv mà Cường tạo), mỗi giao dịch chấm điểm xác suất
gian lận bằng model rồi hiển thị cảnh báo màu:
    - Xanh (st.success)  -> an toàn
    - Vàng (st.warning)  -> nghi vấn, cần xem thêm
    - Đỏ   (st.error)    -> gian lận khả năng cao, chặn ngay

Dùng st.empty() để tạo vùng nội dung "nhảy số" liên tục mô phỏng dòng
giao dịch đổ về theo thời gian thực.
"""

import time
import streamlit as st
import pandas as pd

from backend_logic import predict_proba_for_df


# Ngưỡng phân loại 3 mức cảnh báo (độc lập với threshold của Tab Giám đốc,
# vì ở đây mục đích là phân loại trực quan Xanh/Vàng/Đỏ cho điều tra viên,
# không phải quyết định chặn/không chặn giao dịch).
THRESHOLD_WARNING = 0.4   # >= mức này -> Vàng (nghi vấn)
THRESHOLD_DANGER = 0.75   # >= mức này -> Đỏ (gian lận khả năng cao)


def _render_transaction_card(row: pd.Series, proba: float):
    """Hiển thị 1 giao dịch dưới dạng cảnh báo màu phù hợp."""
    txn_id = row.get("transaction_id", "N/A")
    amount = row.get("amount", 0.0)
    location = f"({row.get('user_lat', '?')}, {row.get('user_lon', '?')})"

    info_line = (
        f"**Mã GD:** `{txn_id}` &nbsp;|&nbsp; "
        f"**Số tiền:** ${amount:,.2f} &nbsp;|&nbsp; "
        f"**Vị trí:** {location} &nbsp;|&nbsp; "
        f"**Xác suất gian lận:** {proba*100:.2f}%"
    )

    if proba >= THRESHOLD_DANGER:
        st.error(f"🔴 **GIAN LẬN KHẢ NĂNG CAO** — {info_line}")
    elif proba >= THRESHOLD_WARNING:
        st.warning(f"🟡 **NGHI VẤN, CẦN KIỂM TRA** — {info_line}")
    else:
        st.success(f"🟢 **GIAO DỊCH AN TOÀN** — {info_line}")


def render_investigator_tab(df: pd.DataFrame, model, speed: float = 1.0, max_rows: int = 50):
    """
    Render toàn bộ nội dung Tab Điều tra viên.

    Tham số:
        df: DataFrame test_stream (50 dòng do Cường tạo).
        model: model đã load từ backend_logic.load_model().
        speed: số giao dịch hiển thị mỗi giây (điều khiển bằng slider).
        max_rows: số dòng tối đa lấy ra mô phỏng (mặc định 50, theo đúng
            kích thước test_stream.csv mà Cường sinh ra).
    """
    st.subheader("🕵️ Luồng giao dịch thời gian thực")

    if getattr(model, "is_mock", False):
        st.info(
            "⚠️ Đang dùng **mô hình giả lập (mock)** vì xgboost_v8.pkl chưa có sẵn. "
            "Khi Vũ train xong và lưu vào saved_models/, tab này sẽ tự động "
            "dùng model thật mà không cần sửa code."
        )

    stream_df = df.head(max_rows).reset_index(drop=True)

    # Chấm điểm trước toàn bộ (rẻ hơn việc gọi model từng dòng trong loop)
    fraud_proba = predict_proba_for_df(model, stream_df)

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_b:
        run = st.toggle("▶️ Bắt đầu luồng", value=False, key="investigator_run")
    with col_c:
        if st.button("🔄 Reset"):
            st.session_state["investigator_run"] = False
            st.rerun()

    progress_placeholder = st.empty()
    stream_placeholder = st.empty()
    summary_placeholder = st.empty()

    if not run:
        with progress_placeholder.container():
            st.caption("Bấm **Bắt đầu luồng** để mô phỏng giao dịch đổ về theo thời gian thực.")
        return

    # ------------------------------------------------------------------
    # Mô phỏng luồng giao dịch "nhảy số" theo thời gian thực
    # ------------------------------------------------------------------
    seen_rows = []
    n = len(stream_df)

    for i in range(n):
        row = stream_df.iloc[i]
        proba = fraud_proba[i]
        seen_rows.append((row, proba))

        progress_placeholder.progress((i + 1) / n, text=f"Đang xử lý giao dịch {i+1}/{n}")

        # Vùng "nhảy số" liên tục: chỉ hiện N giao dịch gần nhất để không
        # làm tràn màn hình, giống màn hình giám sát thực tế.
        with stream_placeholder.container():
            recent = seen_rows[-8:][::-1]  # mới nhất lên trên
            for r, p in recent:
                _render_transaction_card(r, p)

        # Bảng tổng kết cập nhật liên tục theo số giao dịch đã quét
        flagged = sum(1 for _, p in seen_rows if p >= THRESHOLD_DANGER)
        warned = sum(1 for _, p in seen_rows if THRESHOLD_WARNING <= p < THRESHOLD_DANGER)
        with summary_placeholder.container():
            s1, s2, s3 = st.columns(3)
            s1.metric("Đã quét", f"{len(seen_rows)}/{n}")
            s2.metric("🔴 Cảnh báo Đỏ", flagged)
            s3.metric("🟡 Cảnh báo Vàng", warned)

        time.sleep(max(0.05, 1.0 / speed))

    progress_placeholder.success("✅ Đã quét hết luồng giao dịch test_stream.csv")
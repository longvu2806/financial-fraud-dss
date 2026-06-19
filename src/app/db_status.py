"""
db_status.py
-------------
Phụ trách: Cường (Database Connect & Báo cáo)

Hàm get_db_status() trả về dict trạng thái kết nối PostgreSQL để main_app.py
hiển thị lên đầu trang dạng: "Trạng thái DB: Đã kết nối 13,305,915 dòng".

Cấu hình kết nối — theo thứ tự ưu tiên:
  1. Streamlit Secrets (khuyên dùng khi deploy):
         [postgres]
         host = "..."
         port = 5432
         dbname = "..."
         user = "..."
         password = "..."
     Đặt trong .streamlit/secrets.toml (local) hoặc Streamlit Cloud Secrets.

  2. Biến môi trường (khuyên dùng khi chạy Docker / CI):
         FRAUD_DB_HOST, FRAUD_DB_PORT, FRAUD_DB_NAME, FRAUD_DB_USER, FRAUD_DB_PASSWORD

  3. Fallback mặc định: localhost:5432 / frauddb / postgres (chỉ cho dev local).

Nếu không kết nối được (chưa cấu hình, DB chưa chạy, sai credentials),
hàm trả về connected=False kèm message mô tả lỗi — KHÔNG crash app.
"""

import os
import logging
from functools import lru_cache
from typing import Optional

# Load .env tự động (python-dotenv) — ưu tiên trước khi đọc os.environ
try:
    from dotenv import load_dotenv
    # Tìm .env từ thư mục gốc project (2 cấp trên src/app/)
    _env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".env",
    )
    load_dotenv(_env_path, override=False)  # override=False: biến môi trường hệ thống thắng
except ImportError:
    pass  # dotenv chưa cài -> bỏ qua, vẫn dùng os.environ bình thường

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Đọc cấu hình kết nối
# ---------------------------------------------------------------------------

def _get_db_config() -> dict:
    """
    Đọc thông tin kết nối PostgreSQL theo thứ tự ưu tiên:
    Streamlit Secrets > biến môi trường > giá trị mặc định.
    """
    # --- Thử Streamlit Secrets trước ---
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "postgres" in st.secrets:
            cfg = st.secrets["postgres"]
            return {
                "host":     cfg.get("host",     "localhost"),
                "port":     int(cfg.get("port", 5432)),
                "dbname":   cfg.get("dbname",   "frauddb"),
                "user":     cfg.get("user",     "postgres"),
                "password": cfg.get("password", ""),
            }
    except Exception:
        pass  # streamlit chưa khởi động hoặc secrets chưa được cấu hình

    # --- Biến môi trường ---
    return {
        "host":     os.environ.get("DB_HOST",     os.environ.get("FRAUD_DB_HOST",     "localhost")),
        "port":     int(os.environ.get("DB_PORT", os.environ.get("FRAUD_DB_PORT", "5432"))),
        "dbname":   os.environ.get("DB_NAME",     os.environ.get("FRAUD_DB_NAME",     "caixabank_db")),
        "user":     os.environ.get("DB_USER",     os.environ.get("FRAUD_DB_USER",     "postgres")),
        "password": os.environ.get("DB_PASSWORD", os.environ.get("FRAUD_DB_PASSWORD", "")),
    }


# ---------------------------------------------------------------------------
# Kết nối & truy vấn (có cache 60 giây để tránh query liên tục mỗi rerender)
# ---------------------------------------------------------------------------

def _fetch_row_count(cfg: dict) -> int:
    """
    Mở kết nối tới PostgreSQL, đếm số dòng trong bảng transactions.
    Đóng kết nối ngay sau khi xong (không dùng connection pool vì chỉ
    cần đọc một lần mỗi lần app load / refresh).
    """
    import psycopg2  # import ở đây để app không crash nếu psycopg2 chưa cài

    conn = psycopg2.connect(
        host=cfg["host"],
        port=cfg["port"],
        dbname=cfg["dbname"],
        user=cfg["user"],
        password=cfg["password"],
        connect_timeout=5,          # không treo app quá 5 giây nếu DB chậm
        options="-c statement_timeout=8000",  # tối đa 8 giây cho câu COUNT
    )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM fact_transactions;")
            row_count: int = cur.fetchone()[0]
        return row_count
    finally:
        conn.close()


def _load_row_count_cached() -> Optional[int]:
    """
    Wrapper dùng lru_cache để tránh gọi DB liên tục trong cùng một
    phiên chạy Python. Cache sẽ reset khi app restart.

    Nếu muốn TTL thực sự (ví dụ 60 giây), thay thế bằng
    streamlit @st.cache_data(ttl=60) gọi trực tiếp từ main_app.py.
    """
    cfg = _get_db_config()
    return _fetch_row_count(cfg)


# lru_cache với maxsize=1: chỉ giữ 1 kết quả, tự vô hiệu khi app restart.
_cached_fetch = lru_cache(maxsize=1)(_load_row_count_cached)


# ---------------------------------------------------------------------------
# Public API — hàm duy nhất main_app.py cần gọi
# ---------------------------------------------------------------------------

def get_db_status() -> dict:
    """
    Trả về dict:
        connected (bool)  : True nếu kết nối & đếm thành công.
        row_count (int|None): Số dòng trong bảng transactions, hoặc None nếu lỗi.
        message (str)     : Trống nếu thành công, mô tả lỗi nếu có.

    Không raise exception — mọi lỗi đều được bắt và trả về connected=False.
    """
    try:
        # Thử import psycopg2 trước để báo lỗi rõ ràng hơn
        import psycopg2  # noqa: F401
    except ImportError:
        return {
            "connected": False,
            "row_count": None,
            "message": (
                "Thư viện psycopg2 chưa được cài. "
                "Chạy: pip install psycopg2-binary"
            ),
        }

    try:
        row_count = _cached_fetch()
        return {
            "connected": True,
            "row_count": row_count,
            "message": "",
        }

    except Exception as exc:
        logger.warning("Không thể kết nối PostgreSQL: %s", exc)

        # Phân loại lỗi để hiển thị thông báo hữu ích hơn cho Cường debug
        msg = str(exc)
        if "could not connect" in msg or "Connection refused" in msg:
            hint = f"PostgreSQL chưa chạy hoặc sai host/port. Chi tiết: {msg}"
        elif "password authentication" in msg or "authentication failed" in msg:
            hint = f"Sai username/password. Chi tiết: {msg}"
        elif "database" in msg and "does not exist" in msg:
            hint = f"Database không tồn tại. Chi tiết: {msg}"
        elif "statement timeout" in msg or "connect_timeout" in msg:
            hint = f"DB phản hồi quá chậm (timeout). Chi tiết: {msg}"
        else:
            hint = msg

        return {
            "connected": False,
            "row_count": None,
            "message": hint,
        }


# ---------------------------------------------------------------------------
# Tiện ích bổ sung cho Cường (tuỳ chọn dùng trong báo cáo / logging)
# ---------------------------------------------------------------------------

def get_db_summary() -> dict:
    """
    Mở rộng của get_db_status(): bổ sung thêm số liệu tổng hợp
    (số giao dịch gian lận, tổng tiền) nếu kết nối thành công.
    Dùng cho phần báo cáo tổng quan trong main_app nếu cần.
    """
    base = get_db_status()
    if not base["connected"]:
        return base

    cfg = _get_db_config()
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=cfg["host"], port=cfg["port"], dbname=cfg["dbname"],
            user=cfg["user"], password=cfg["password"],
            connect_timeout=5, options="-c statement_timeout=10000",
        )
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        COUNT(*)                                    AS total_rows,
                        SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_count,
                        SUM(amount)                                 AS total_amount,
                        SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) AS fraud_amount
                    FROM fact_transactions;
                """)
                row = cur.fetchone()
        finally:
            conn.close()

        base.update({
            "fraud_count":   int(row[1] or 0),
            "total_amount":  float(row[2] or 0),
            "fraud_amount":  float(row[3] or 0),
        })
    except Exception as exc:
        logger.warning("get_db_summary() lỗi khi query tổng hợp: %s", exc)
        # Không ghi đè trạng thái connected — kết nối cơ bản vẫn OK

    return base
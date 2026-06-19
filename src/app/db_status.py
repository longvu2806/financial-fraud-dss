"""
db_status.py
-------------
Phụ trách: Cường (Database Connect & Báo cáo)

Hàm get_db_status() trả về dict trạng thái kết nối PostgreSQL để main_app.py
hiển thị lên đầu trang dạng: "Trạng thái DB: Đã kết nối 13,305,915 dòng".
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)

# ==========================================
# 🎯 ĐỊNH VỊ THƯ MỤC GỐC ĐỂ IMPORT CONFIG
# ==========================================
# Đảm bảo Python nhận diện được thư mục gốc của dự án (financial-fraud-dss)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Lùi về các cấp thư mục cho đến khi chạm root chứa thư mục 'config'
root_dir = current_dir
while not os.path.exists(os.path.join(root_dir, 'config')) and root_dir != os.path.dirname(root_dir):
    root_dir = os.path.dirname(root_dir)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import cấu hình Database
try:
    from config.database_config import DBConfig
except ImportError:
    raise ImportError("❌ Không tìm thấy file config/database_config.py. Vui lòng kiểm tra lại cấu trúc thư mục.")


# ==========================================
# 🎯 HÀM LẤY TRẠNG THÁI DATABASE
# ==========================================
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
        import psycopg2
    except ImportError:
        return {
            "connected": False,
            "row_count": None,
            "message": "Thư viện psycopg2 chưa được cài. Chạy: pip install psycopg2-binary"
        }

    try:
        # Sử dụng thông số kết nối từ file config của nhóm
        conn = psycopg2.connect(
            host=DBConfig.HOST,
            port=DBConfig.PORT,
            dbname=DBConfig.NAME,
            user=DBConfig.USER,
            password=DBConfig.PASSWORD,
            connect_timeout=5,                  # Không treo app quá 5s
            options="-c statement_timeout=8000" # Tối đa 8s cho câu COUNT
        )
        
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM fact_transactions;")
            row_count = cur.fetchone()[0]
        
        conn.close()

        return {
            "connected": True,
            "row_count": row_count,
            "message": ""
        }

    except psycopg2.OperationalError as exc:
        msg = str(exc).strip()
        # Phân loại lỗi chi tiết (Kế thừa từ bản GitHub)
        if "could not connect" in msg or "Connection refused" in msg:
            hint = f"PostgreSQL chưa chạy hoặc sai host/port. Chi tiết: {msg}"
        elif "password authentication" in msg or "authentication failed" in msg:
            hint = f"Sai username/password. Chi tiết: {msg}"
        elif "database" in msg and "does not exist" in msg:
            hint = f"Database không tồn tại. Chi tiết: {msg}"
        elif "statement timeout" in msg or "connect_timeout" in msg:
            hint = f"DB phản hồi quá chậm (timeout). Chi tiết: {msg}"
        else:
            hint = f"Lỗi kết nối PostgreSQL: {msg}"

        return {
            "connected": False,
            "row_count": None,
            "message": hint,
        }
    except Exception as exc:
        return {
            "connected": False,
            "row_count": None,
            "message": f"Lỗi truy vấn dữ liệu: {str(exc).strip()}",
        }


# ==========================================
# 🎯 TIỆN ÍCH BỔ SUNG (Dành cho Dashboard/Báo cáo)
# ==========================================
def get_db_summary() -> dict:
    """
    Mở rộng của get_db_status(): bổ sung thêm số liệu tổng hợp
    (số giao dịch gian lận, tổng tiền) nếu kết nối thành công.
    Dùng cho phần báo cáo tổng quan trong main_app.
    """
    base = get_db_status()
    if not base["connected"]:
        return base

    try:
        import psycopg2

        conn = psycopg2.connect(
            host=DBConfig.HOST, port=DBConfig.PORT, dbname=DBConfig.NAME,
            user=DBConfig.USER, password=DBConfig.PASSWORD,
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

    return base


# ==========================================
# 🧪 CHẠY THỬ NGHIỆM ĐỘC LẬP
# ==========================================
if __name__ == "__main__":
    print("⏳ Đang kiểm tra kết nối Database...")
    status = get_db_status()
    
    if status["connected"]:
        print("✅ KẾT NỐI THÀNH CÔNG!")
        print(f"📊 Trạng thái DB: Đã kết nối {status['row_count']:,} dòng.")
    else:
        print("❌ KẾT NỐI THẤT BẠI!")
        print(f"⚠️ Chi tiết lỗi: {status['message']}")
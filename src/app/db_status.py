"""
db_status.py
-------------
Phụ trách: Cường (Database Connect & Báo cáo)

Hàm get_db_status() trả về dict trạng thái kết nối PostgreSQL để main_app.py
hiển thị lên đầu trang dạng: "Trạng thái DB: Đã kết nối 13,305,915 dòng".

TODO (Cường điền logic thật ở đây):
    import psycopg2
    conn = psycopg2.connect(
        host="...", port=5432, dbname="...", user="...", password="...",
    )
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM transactions;")
    row_count = cur.fetchone()[0]
    conn.close()

Hiện tại đây là bản giả lập (mock) để main_app.py chạy được ngay
trong lúc Cường chưa hoàn thiện kết nối DB thật.
"""


def get_db_status() -> dict:
    """
    Trả về dict:
        connected: bool
        row_count: int hoặc None
        message: str (mô tả lỗi nếu có, hoặc trống nếu thành công)
    """
    try:
        # ---- THAY PHẦN NÀY BẰNG KẾT NỐI POSTGRES THẬT CỦA CƯỜNG ----
        raise NotImplementedError("Kết nối Postgres thật chưa được cấu hình.")

    except Exception as e:
        return {
            "connected": False,
            "row_count": None,
            "message": str(e),
        }
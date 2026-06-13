# config/database_config.py
import os
from dotenv import load_dotenv

# Tìm và nạp các biến môi trường từ file .env ở thư mục gốc
load_dotenv()

class DBConfig:
    USER = os.getenv("DB_USER", "postgres")
    PASSWORD = os.getenv("DB_PASSWORD", "password")
    HOST = os.getenv("DB_HOST", "localhost")
    PORT = os.getenv("DB_PORT", "5432")
    NAME = os.getenv("DB_NAME", "caixabank_db")
    
    @classmethod
    def get_connection_url(cls):
        """Trả về chuỗi URL kết nối chuẩn cho SQLAlchemy / Psycopg2"""
        return f"postgresql://{cls.USER}:{cls.PASSWORD}@{cls.HOST}:{cls.PORT}/{cls.NAME}"

# Chạy thử nghiệm độc lập để kiểm tra cấu hình
if __name__ == "__main__":
    print("🔗 URL Kết nối CSDL của bạn là:")
    print(DBConfig.get_connection_url())
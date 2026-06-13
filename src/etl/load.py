# src/etl/load.py
import psycopg2
import io
import os
import sys
from sqlalchemy import create_engine

# Đoạn code này giúp Python tìm thấy thư mục config ở ngoài root khi chạy file local
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.database_config import DBConfig

class PostgresLoader:
    def __init__(self):
        # 🎯 LẤY CẤU HÌNH TỰ ĐỘNG TỪ THƯ MỤC CONFIG TẠI ĐÂY
        self.db_url = DBConfig.get_connection_url()
        self.conn = psycopg2.connect(self.db_url)
        self.engine = create_engine(self.db_url)
        
    def execute_schema(self, schema_path=None):
        """Khởi tạo cấu trúc các bảng trong cơ sở dữ liệu"""
        if schema_path is None:
            # Tự động tìm đường dẫn đến file schema.sql từ vị trí file load.py
            current_dir = os.path.dirname(os.path.abspath(__file__))
            schema_path = os.path.abspath(os.path.join(current_dir, "../db/schema.sql"))
            
        print(f"🛠️ Đang đọc file cấu trúc: {schema_path}")
        print("🛠️ Đang khởi tạo cấu trúc bảng CSDL Caixabank...")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        with self.conn.cursor() as cursor:
            cursor.execute(schema_sql)
        self.conn.commit()
        print("✔️ Khởi tạo cấu trúc CSDL PostgreSQL thành công!")

    def fast_load_dataframe(self, df, table_name):
        """Hàm nạp siêu tốc sử dụng cơ chế Hashing Memory Copy của Psycopg2"""
        if df.empty:
            return
        print(f"📥 Đang nạp {len(df)} dòng vào bảng '{table_name}'...")
        output = io.StringIO()
        df.to_csv(output, sep='\t', header=False, index=False, na_rep='\\N')
        output.seek(0)
        
        with self.conn.cursor() as cursor:
            try:
                cursor.copy_from(output, table_name, null='\\N', sep='\t')
                self.conn.commit()
                print(f"✅ Nạp bảng '{table_name}' thành công!")
            except Exception as e:
                self.conn.rollback()
                print(f"❌ THẤT BẠI khi nạp bảng {table_name}: {e}")
                raise e

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    loader = PostgresLoader()
    loader.execute_schema()
    loader.close()
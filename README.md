#1 Kích hoạt môi trường ảo

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
python -m venv .venv
.\.venv\Scripts\Activate.ps1

#2 cài thư viện

pip install -r requirements.txt

#3 Quy trình đẩy code
+ Thêm nhánh mới: git checkout -b branch_name
+ Chuyển qua các nhánh: git checkout branch_name
- Quy trình đẩy code
+ B1: git add .
+ B2: git status ( để kiểm tra trạng thái, có thể bỏ qua)
+ B3: git commit -m "nội dung đã chỉnh sửa"
+ B4: git push origin -u branch_name

#4
Tạo 1 file tên .env ở thư mục gốc ( cùng cấp với csac thư mục như data,notebooks,..)
Sau đó copy và sửa tt DATABASE của mình
# Thông tin kết nối PostgreSQL Local của bạn
DB_USER=postgres
DB_PASSWORD=password_cua_ban
DB_HOST=localhost
DB_PORT=5432
DB_NAME=caixabank_db
#5 Cài dotenv
pip install dotenv
#6 MỞ POSTGRESQL tạo database caixabank_db
<<<<<<< HEAD
#7 chay run_pipeline.py (lưu ý phải chuyền về thư mục gốc) -> python src/etl/run_pipeline.py
=======
#7 chay run_pipeline.py (lưu ý phải chuyền về thư mục gốc) -> python src/etl/run_pipeline.py

# cài thư viện chạy model của Vũ
pip install pyarrow xgboost pandas scikit-learn

# chạy app
streamlit run src/app/main_app.py 
>>>>>>> f9fc0df ( hiếu xong sơ bộ app)

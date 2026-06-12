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

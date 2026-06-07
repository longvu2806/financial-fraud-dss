#1 Kích hoạt môi trường ảo

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
python -m venv .venv
.\.venv\Scripts\Activate.ps1

#2 cài thư viện

pip install -r requirements.txt
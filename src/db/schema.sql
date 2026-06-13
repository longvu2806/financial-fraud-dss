-- =========================================================================
-- 🏛️ CAIXABANK CENTRAL DATA WAREHOUSE SCHEMA (SILVER/GOLD LAYER)
-- =========================================================================

-- 1. Tạo bảng danh mục mã MCC (Merchant Category Codes)
CREATE TABLE IF NOT EXISTS dim_mcc (
    mcc VARCHAR(10) PRIMARY KEY,
    merchant_category VARCHAR(255) NOT NULL
);

-- 2. Tạo bảng hồ sơ tĩnh của Khách hàng (Users Profiles)
CREATE TABLE IF NOT EXISTS dim_users (
    user_id BIGINT PRIMARY KEY,
    user_age INT,
    retirement_age INT,
    birth_year INT,
    birth_month INT,
    gender VARCHAR(10),
    address TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    per_capita_income DOUBLE PRECISION, -- Lưu số thực, đã làm sạch dấu $
    yearly_income DOUBLE PRECISION,     -- Lưu số thực, đã làm sạch dấu $
    total_debt DOUBLE PRECISION,        -- Lưu số thực, đã làm sạch dấu $
    credit_score INT,
    num_credit_cards INT
);

-- 3. Tạo bảng hồ sơ quản lý Thẻ (Cards Profiles)
CREATE TABLE IF NOT EXISTS dim_cards (
    card_id BIGINT PRIMARY KEY,
    user_id BIGINT,
    card_brand VARCHAR(50),             -- Đã chuẩn hóa chữ thường
    card_type VARCHAR(50),              -- Đã chuẩn hóa chữ thường
    card_number BIGINT,
    expires VARCHAR(10),
    cvv INT,
    has_chip VARCHAR(10),
    num_cards_issued INT,
    credit_limit DOUBLE PRECISION,      -- Lưu số thực, đã làm sạch dấu $
    acct_open_date VARCHAR(20),
    year_pin_last_changed INT,
    card_on_dark_web VARCHAR(10)
);

-- 4. Tạo bảng sự kiện Giao dịch chính (Fact Transactions)
CREATE TABLE IF NOT EXISTS fact_transactions (
    transaction_id BIGINT PRIMARY KEY,  -- Ánh xạ từ cột 'id' thô ban đầu
    date TIMESTAMP,                     -- Ép kiểu dữ liệu thời gian hệ thống chuẩn
    user_id BIGINT,
    card_id BIGINT,
    amount DOUBLE PRECISION,            -- Đã dọn dấu $ và áp mốc cắt cụt Outlier 874.20
    use_chip VARCHAR(50),               -- Đã xử lý chuỗi chữ thường
    merchant_id BIGINT,
    merchant_city VARCHAR(255),
    merchant_state VARCHAR(50),         -- Đã điền khuyết thiếu 'UNKNOWN'
    zip VARCHAR(20),                    -- Đã xử lý rác .0 và đưa về chuỗi văn bản
    mcc VARCHAR(10),
    errors VARCHAR(255),                -- Đã điền khuyết thiếu 'no_error'
    
    -- ⏱️ 3 Cột Kỹ nghệ đặc trưng (Features) phân rã từ Timestamp phục vụ Mô hình AI
    tx_hour SMALLINT,                   -- Giờ giao dịch từ 0 -> 23
    tx_day_of_week SMALLINT,            -- Ngày trong tuần từ 0 -> 6 (0 là Thứ Hai)
    is_night_tx SMALLINT                -- Biến nhị phân (1 nếu giao dịch từ 1h-5h sáng, ngược lại là 0)
);

-- 5. Tạo bảng nhãn kiểm định phục vụ Mô hình Học Máy (Fraud Labels)
CREATE TABLE IF NOT EXISTS fact_fraud_labels (
    transaction_id BIGINT PRIMARY KEY,
    is_fraud SMALLINT                   -- Đã ánh xạ thành số thực tế (0: An toàn, 1: Gian lận)
);

-- =========================================================================
-- ⚡ TỐI ƯU HÓA HẠ TẦNG: KHỞI TẠO MA TRẬN INDEX CHO TOÀN BỘ CSDL
-- =========================================================================
-- Các chỉ mục Index này giúp tối ưu hóa tốc độ chạy các câu lệnh JOIN 
-- giữa bảng Fact khổng lồ với các bảng Dim khi trích xuất dữ liệu huấn luyện AI.

CREATE INDEX IF NOT EXISTS idx_trans_user ON fact_transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_trans_card ON fact_transactions(card_id);
CREATE INDEX IF NOT EXISTS idx_trans_mcc ON fact_transactions(mcc);
CREATE INDEX IF NOT EXISTS idx_trans_date ON fact_transactions(date);
CREATE INDEX IF NOT EXISTS idx_trans_is_night ON fact_transactions(is_night_tx);
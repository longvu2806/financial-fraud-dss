import pandas as pd
from sqlalchemy import create_engine
from config.database_config import DBConfig

class BusinessDashboardBackend:
    def __init__(self):
        self.engine = create_engine(DBConfig.get_connection_url())

    # ==========================================
    # DASHBOARD 1: SPENDING TRENDS
    # ==========================================
    def fetch_spending_trends(self):
        query = """
        SELECT DATE_TRUNC('month', date) as month, SUM(amount) as total_spent
        FROM fact_transactions WHERE is_fraud = 0
        GROUP BY 1 ORDER BY 1;
        """
        return pd.read_sql(query, self.engine)

    def fetch_top_categories(self):
        query = """
        SELECT m.merchant_category, SUM(t.amount) as total_spent
        FROM fact_transactions t JOIN dim_mcc m ON t.mcc = m.mcc
        WHERE t.is_fraud = 0 GROUP BY 1 ORDER BY 2 DESC LIMIT 10;
        """
        return pd.read_sql(query, self.engine)

    def fetch_spending_heatmap(self):
        query = """
        SELECT EXTRACT(DOW FROM date) as day_of_week, tx_hour, SUM(amount) as total_spent
        FROM fact_transactions WHERE is_fraud = 0 GROUP BY 1, 2;
        """
        return pd.read_sql(query, self.engine)

    def fetch_payment_method_trend(self):
        query = """
        SELECT EXTRACT(YEAR FROM date) as year, use_chip, SUM(amount) as total_spent
        FROM fact_transactions WHERE is_fraud = 0 GROUP BY 1, 2 ORDER BY 1, 2;
        """
        return pd.read_sql(query, self.engine)

    # ==========================================
    # DASHBOARD 2: CUSTOMER SEGMENTATION
    # ==========================================
    def fetch_income_vs_spending(self, limit=5000):
        # Lấy mẫu User để vẽ Scatter
        query = f"""
        SELECT u.yearly_income, SUM(t.amount) as avg_yearly_spend
        FROM fact_transactions t JOIN dim_users u ON t.user_id = u.user_id
        WHERE t.is_fraud = 0 
        GROUP BY u.user_id, u.yearly_income 
        HAVING SUM(t.amount) > 0 LIMIT {limit};
        """
        return pd.read_sql(query, self.engine)

    def fetch_generation_spending(self):
        query = """
        SELECT 
            CASE 
                WHEN u.user_age <= 27 THEN 'Gen Z'
                WHEN u.user_age <= 43 THEN 'Millennials'
                WHEN u.user_age <= 59 THEN 'Gen X'
                ELSE 'Boomers' END as generation,
            SUM(t.amount) as total_spent
        FROM fact_transactions t JOIN dim_users u ON t.user_id = u.user_id
        WHERE t.is_fraud = 0 GROUP BY 1;
        """
        return pd.read_sql(query, self.engine)

    def fetch_gen_category_behavior(self):
        query = """
        SELECT 
            CASE 
                WHEN u.user_age <= 27 THEN 'Gen Z' WHEN u.user_age <= 43 THEN 'Millennials'
                WHEN u.user_age <= 59 THEN 'Gen X' ELSE 'Boomers' END as generation,
            m.merchant_category, SUM(t.amount) as total_spent
        FROM fact_transactions t 
        JOIN dim_users u ON t.user_id = u.user_id
        JOIN dim_mcc m ON t.mcc = m.mcc
        WHERE t.is_fraud = 0 GROUP BY 1, 2
        """
        df = pd.read_sql(query, self.engine)
        # Chỉ lấy Top 5 ngành hàng cho biểu đồ sạch đẹp
        top_cats = df.groupby('merchant_category')['total_spent'].sum().nlargest(5).index
        return df[df['merchant_category'].isin(top_cats)]

    def fetch_spending_by_state(self):
        query = """
        SELECT u.address as state, COUNT(DISTINCT u.user_id) as users_count, SUM(t.amount) as total_spent
        FROM fact_transactions t JOIN dim_users u ON t.user_id = u.user_id
        WHERE t.is_fraud = 0 GROUP BY 1;
        """
        return pd.read_sql(query, self.engine)

    # ==========================================
    # DASHBOARD 3: FINANCIAL HEALTH
    # ==========================================
    def fetch_credit_score_dist(self):
        query = "SELECT credit_score FROM dim_users;"
        return pd.read_sql(query, self.engine)

    def fetch_utilization_ratio(self, limit=5000):
        query = f"""
        SELECT c.card_type, (SUM(t.amount) / NULLIF(MAX(c.credit_limit), 0)) * 100 as utilization_pct
        FROM fact_transactions t JOIN dim_cards c ON t.card_id = c.card_id
        WHERE t.is_fraud = 0 GROUP BY t.card_id, c.card_type LIMIT {limit};
        """
        return pd.read_sql(query, self.engine)

    def fetch_debt_distribution(self):
        query = "SELECT total_debt FROM dim_users WHERE total_debt > 0;"
        return pd.read_sql(query, self.engine)

    def fetch_card_performance(self):
        query = """
        SELECT c.card_brand, c.card_type, SUM(t.amount) as total_volume
        FROM fact_transactions t JOIN dim_cards c ON t.card_id = c.card_id
        WHERE t.is_fraud = 0 GROUP BY 1, 2;
        """
        return pd.read_sql(query, self.engine)

    # ==========================================
    # DASHBOARD 4: OPERATIONAL FRICTION
    # ==========================================
    # =========================================================================
    # TAB 4: OPERATIONAL FRICTION (LỖI VÀ TRẢI NGHIỆM)
    # =========================================================================
    def fetch_error_breakdown(self):
        """Top mã lỗi xuất hiện khi giao dịch hợp pháp"""
        # THÊM 'no_error' VÀO DANH SÁCH LOẠI TRỪ
        query = """
        SELECT errors, COUNT(*) as error_count
        FROM fact_transactions 
        WHERE is_fraud = 0 
          AND errors IS NOT NULL 
          AND LOWER(TRIM(errors::text)) NOT IN ('', 'nan', 'none', 'null', '0', 'no_error')
        GROUP BY 1 
        ORDER BY 2 DESC 
        LIMIT 10;
        """
        return pd.read_sql(query, self.engine)

    def fetch_decline_rate(self):
        """Tỷ lệ giao dịch thất bại theo phương thức quẹt thẻ"""
        # THÊM 'no_error' VÀO DANH SÁCH LOẠI TRỪ
        query = """
        SELECT use_chip, 
               COUNT(*) as total_tx,
               SUM(CASE WHEN errors IS NOT NULL 
                         AND LOWER(TRIM(errors::text)) NOT IN ('', 'nan', 'none', 'null', '0', 'no_error') 
                    THEN 1 ELSE 0 END) as failed_tx
        FROM fact_transactions 
        WHERE is_fraud = 0 
        GROUP BY 1;
        """
        df = pd.read_sql(query, self.engine)
        
        # Tính phần trăm và tránh lỗi chia cho 0
        df['decline_rate'] = (df['failed_tx'] / df['total_tx'].replace(0, 1)) * 100
        return df
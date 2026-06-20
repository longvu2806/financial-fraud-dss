import pandas as pd
from sqlalchemy import create_engine
from config.database_config import DBConfig

class FraudDashboardBackend:
    def __init__(self):
        # SỬ DỤNG SQLALCHEMY THEO ĐÚNG KHUYẾN CÁO CỦA PANDAS
        # DBConfig.get_connection_url() sẽ trả về dạng "postgresql://user:pass@host:port/dbname"
        self.engine = create_engine(DBConfig.get_connection_url())

    # =========================================================================
    # TAB 1: EXECUTIVE SUMMARY (GIÁM ĐỐC)
    # =========================================================================
    def fetch_executive_kpis(self):
        query = """
        SELECT 
            COUNT(*) as total_tx,
            SUM(amount) as total_amt,
            SUM(is_fraud) as fraud_count,
            SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) as fraud_loss
        FROM fact_transactions;
        """
        # engine sẽ tự động mở/đóng kết nối mượt mà
        df = pd.read_sql(query, self.engine)
        return df.iloc[0].to_dict()

    def fetch_fraud_trend(self, timeframe='month'):
        query = f"""
        SELECT 
            DATE_TRUNC('{timeframe}', date) as period, 
            COUNT(*) as total_count,
            SUM(is_fraud) as fraud_count,
            SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) as fraud_amount
        FROM fact_transactions
        GROUP BY 1 ORDER BY period;
        """
        return pd.read_sql(query, self.engine)

    def fetch_card_type_distribution(self):
        query = """
        SELECT c.card_type, SUM(t.is_fraud) as fraud_count
        FROM fact_transactions t
        JOIN dim_cards c ON t.card_id = c.card_id
        GROUP BY c.card_type;
        """
        return pd.read_sql(query, self.engine)

    # =========================================================================
    # TAB 2: FRAUD PATTERN (HÀNH VI)
    # =========================================================================
    def fetch_heatmap_data(self):
        query = """
        SELECT 
            EXTRACT(DOW FROM date) as tx_day_of_week, 
            EXTRACT(HOUR FROM date) as tx_hour, 
            SUM(is_fraud) as fraud_count
        FROM fact_transactions
        GROUP BY 1, 2;
        """
        return pd.read_sql(query, self.engine)

    def fetch_amount_distribution_sample(self, limit=10000):
        query = f"""
        (SELECT 'Hợp pháp' as status, amount FROM fact_transactions WHERE is_fraud = 0 LIMIT {limit})
        UNION ALL
        (SELECT 'Gian lận' as status, amount FROM fact_transactions WHERE is_fraud = 1 LIMIT {limit})
        """
        return pd.read_sql(query, self.engine)

    def fetch_chip_usage_stats(self):
        query = """
        SELECT use_chip, AVG(is_fraud)*100 as fraud_rate
        FROM fact_transactions
        GROUP BY use_chip;
        """
        return pd.read_sql(query, self.engine)

    def fetch_error_analysis(self):
        query = """
        SELECT errors, COUNT(*) as count
        FROM fact_transactions
        WHERE is_fraud = 1 AND errors IS NOT NULL
        GROUP BY errors ORDER BY count DESC LIMIT 10;
        """
        return pd.read_sql(query, self.engine)

    # =========================================================================
    # TAB 3: CUSTOMER & CARD RISK (KHÁCH HÀNG)
    # =========================================================================
    def fetch_darkweb_impact(self):
        query = """
        SELECT c.card_on_dark_web, t.is_fraud, COUNT(*) as count
        FROM fact_transactions t
        JOIN dim_cards c ON t.card_id = c.card_id
        GROUP BY 1, 2;
        """
        return pd.read_sql(query, self.engine)

    def fetch_debt_vs_credit_sample(self, limit=5000):
        query = f"""
        SELECT u.credit_score, (u.total_debt / NULLIF(u.yearly_income, 0)) as debt_ratio, t.is_fraud
        FROM fact_transactions t
        JOIN dim_users u ON t.user_id = u.user_id
        WHERE t.transaction_id IN (SELECT transaction_id FROM fact_transactions TABLESAMPLE SYSTEM (0.1))
        LIMIT {limit};
        """
        return pd.read_sql(query, self.engine)

    def fetch_victim_age_dist(self):
        query = """
        SELECT u.user_age
        FROM fact_transactions t
        JOIN dim_users u ON t.user_id = u.user_id
        WHERE t.is_fraud = 1;
        """
        return pd.read_sql(query, self.engine)

    # =========================================================================
    # TAB 4: GEOGRAPHIC & MERCHANT (ĐỊA LÝ)
    # =========================================================================
    def fetch_fraud_map_data(self):
        query = """
        SELECT merchant_state, COUNT(*) as count, AVG(amount) as avg_amt
        FROM fact_transactions
        WHERE is_fraud = 1
        GROUP BY merchant_state;
        """
        return pd.read_sql(query, self.engine)

    def fetch_top_mcc_risk(self):
        query = """
        SELECT m.merchant_category, COUNT(t.is_fraud) as fraud_count
        FROM fact_transactions t
        JOIN dim_mcc m ON t.mcc = m.mcc
        WHERE t.is_fraud = 1
        GROUP BY 1 ORDER BY fraud_count DESC LIMIT 10;
        """
        return pd.read_sql(query, self.engine)

    def fetch_state_hop_flow(self):
        query = """
        SELECT u.address as user_state, t.merchant_state, COUNT(*) as value
        FROM fact_transactions t
        JOIN dim_users u ON t.user_id = u.user_id
        WHERE t.is_fraud = 1 AND u.address != t.merchant_state
        GROUP BY 1, 2 ORDER BY value DESC LIMIT 20;
        """
        return pd.read_sql(query, self.engine)
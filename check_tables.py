import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
cur = conn.cursor()
cur.execute("""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = 'fact_transactions'
    ORDER BY ordinal_position;
""")
for col, dtype in cur.fetchall():
    print(f"{col:40s} {dtype}")
conn.close()
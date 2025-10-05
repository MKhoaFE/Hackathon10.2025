from sqlalchemy import create_engine, text

DB_CONNECTION_STRING = (
    "mssql+pyodbc://@HC-C-004RC\\SQLEXPRESS/TestAIRag"
    "?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
)

engine = create_engine(DB_CONNECTION_STRING)

try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("Database connection OK:", result.scalar())
except Exception as e:
    print("Connection failed:", e)

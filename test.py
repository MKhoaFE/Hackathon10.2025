from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

# ==============================
# 1. Cấu hình kết nối DB
# ==============================
DB_CONNECTION_STRING = "mssql+pyodbc://SGPCLU239-AG1-L.apac.bosch.com/DB_SOVH_BI_MS_SQL?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
os.environ["http_proxy"] = "http://127.0.0.1:3128"
os.environ["https_proxy"] = "http://127.0.0.1:3128"
engine = create_engine(DB_CONNECTION_STRING)

# ==============================
# 2. Load model embedding
# ==============================
print("Đang load model intfloat/multilingual-e5-base ...")
model = SentenceTransformer("intfloat/multilingual-e5-base")

# ==============================
# 3. Lấy dữ liệu từ bảng
# ==============================
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT TOP (1010) movie_id, title, overview
        FROM dbo.HKT_Movies
        WHERE embedding IS NULL
    """))
    movies = result.fetchall()

print(f"Đã lấy {len(movies)} bộ phim cần tạo embedding.")

# ==============================
# 4. Tính embedding cho từng phim
# ==============================
updates = []
for movie_id, title, overview in movies:
    text_input = f"{title}. {overview or ''}"
    embedding = model.encode(text_input, normalize_embeddings=True)
    embedding_json = json.dumps(embedding.tolist())
    updates.append((embedding_json, movie_id))

print(f"Đã tạo embedding cho {len(updates)} bản ghi.")

# ==============================
# 5. Ghi embedding trở lại DB
# ==============================
with engine.begin() as conn:
    for emb, mid in updates:
        conn.execute(
            text("UPDATE dbo.HKT_Movies SET embedding = :emb WHERE movie_id = :mid"),
            {"emb": emb, "mid": mid}
        )

print("✅ Đã cập nhật embedding vào cột [embedding] trong HKT_Movies thành công.")

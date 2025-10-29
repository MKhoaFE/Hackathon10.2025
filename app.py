from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import requests
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import re
import faiss

# setup proxy (nếu cần)
os.environ["http_proxy"] = "http://127.0.0.1:3128"
os.environ["https_proxy"] = "http://127.0.0.1:3128"

# ------------------------
# Flask setup
# ------------------------
app = Flask(__name__)
CORS(app)
load_dotenv()



# ------------------------
# Database setup
# ------------------------
engine = None
Session = None
try:
    engine = create_engine(DB_CONNECTION_STRING)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM dbo.HKT_Movies"))
        print(" DB connected successfully. Total movies:", list(result)[0][0])
except Exception as e:
    print(" DB connection failed:", e)

Base = declarative_base()

# ------------------------
# Embedding model
# ------------------------
# NOTE: Bạn đã chọn "intfloat/multilingual-e5-base"
embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
EMBED_DIM = embedding_model.get_sentence_embedding_dimension()  # ví dụ 768

# ------------------------
# Movie model
# ------------------------
class HKT_Movies(Base):
    __tablename__ = 'HKT_Movies'

    movie_id = Column(Integer, primary_key=True)
    title = Column(String(500))
    original_title = Column(String(500))
    overview = Column(Text)
    release_date = Column(String(50))
    runtime = Column(Integer)
    status = Column(String(100))
    original_language = Column(String(10))
    vote_average = Column(Float)
    vote_count = Column(Integer)
    popularity = Column(Float)
    budget = Column(Float)
    revenue = Column(Float)
    imdb_id = Column(String(50))
    embedding = Column(Text)  # JSON string

class HKT_Movie_Genres(Base):
    __tablename__ = 'HKT_Movie_Genres'

    id = Column(Integer, primary_key=True, autoincrement=True)
    movie_id = Column(Integer)
    genre_id = Column(Integer)
    embedding = Column(Text)

class HKT_Genres(Base):
    __tablename__ = 'HKT_Genres'

    genre_id = Column(Integer, primary_key=True)
    genre_name = Column(String(200))
    embedding = Column(Text)

class HKT_Recommendations(Base):
    __tablename__ = 'HKT_Recommendations'

    id = Column(Integer, primary_key=True, autoincrement=True)
    movie_id = Column(Integer)
    recommended_movie_id = Column(Integer)
    embedding = Column(Text)

if engine:
    Base.metadata.create_all(engine)

# ------------------------
# Helper functions (embedding / clean)
# ------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'<[^>]+>', '', text)            # loại bỏ HTML
    text = re.sub(r'[\n\r]+', ' ', text)          # replace newline bằng space
    # Giữ ký tự chữ số + chữ tiếng Việt + latin cơ bản
    text = re.sub(r'[^0-9a-z\s\àáâãèéêìíòóôõùúăđĩũơưạảầầ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)              # normalize whitespace
    return text.strip()

def get_embedding(text: str):
    """Trả về vector embedding dạng list (float). Luôn đúng EMBED_DIM."""
    text = clean_text(text)
    if not text:
        vec = np.zeros(EMBED_DIM, dtype='float32')
    else:
        vec = embedding_model.encode(text)
        vec = np.array(vec, dtype='float32')
        # ensure correct dim
        if vec.shape[0] != EMBED_DIM:
            # Nếu model trả dim khác (rất hiếm), làm padding/trunc
            new = np.zeros(EMBED_DIM, dtype='float32')
            new[:min(vec.shape[0], EMBED_DIM)] = vec[:EMBED_DIM]
            vec = new
    return vec.tolist()


# ------------------------
# FAISS indexes and builders
# ------------------------
faiss_index = None
movie_id_map = []

watch_faiss_index = None
watch_id_map = []

def reembed_all_movies(session):
    """Re-encode all movies using current embedding_model (use when model changed)."""
    movies = session.query(HKT_Movies).all()
    print(f"Re-embedding {len(movies)} movies with model dim={EMBED_DIM}")
    for m in movies:
        text = f"{m.title or ''} {m.overview or ''}".strip()
        if not text:
            vec = np.zeros(EMBED_DIM, dtype='float32')
        else:
            vec = np.array(embedding_model.encode(clean_text(text)), dtype='float32')
        m.embedding = json.dumps(vec.tolist())
    session.commit()
    print("Re-embedding movies done.")


def build_faiss_index(session):
    """Build FAISS index từ bảng HKT_Movies."""
    global faiss_index, movie_id_map
    movies = session.query(HKT_Movies).filter(HKT_Movies.embedding.isnot(None)).all()
    if not movies:
        faiss_index = None
        movie_id_map = []
        return
    
    embeddings = np.array([json.loads(m.embedding) for m in movies], dtype='float32')
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBED_DIM:
        print("Dim mismatch hoặc embedding lỗi — cần re-embedding.")
        for m in movies:
            text = f"{m.title or ''} {m.overview or ''}".strip()
            m.embedding = json.dumps(get_embedding(text))
        session.commit()
        embeddings = np.array([json.loads(m.embedding) for m in movies], dtype='float32')

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)

    movie_id_map = [m.movie_id for m in movies]
    print(f"FAISS index built with {len(movie_id_map)} HKT_Movies, dim={dim}")





# ------------------------
# Update missing embeddings (keeps existing, fills blanks)
# ------------------------
def update_missing_embeddings(session):
    movies = session.query(HKT_Movies).filter(
        (HKT_Movies.embedding.is_(None)) | 
        (HKT_Movies.embedding == "") | 
        (HKT_Movies.embedding == "null")
    ).all()
    print(f"Found {len(movies)} movies missing embeddings")

    for movie in movies:
        text_input = f"{movie.title or ''}. {movie.overview or ''}".strip()
        if not text_input:
            continue

        try:
            emb = embedding_model.encode(text_input, normalize_embeddings=True)
            emb = np.array(emb, dtype='float32')
            movie.embedding = json.dumps(emb.tolist())
        except Exception as e:
            print(f"⚠️ Embedding failed for movie_id={movie.movie_id}: {e}")
    
    try:
        session.commit()
        print("Embeddings updated successfully.")
    except Exception as e:
        session.rollback()
        print("Commit failed:", e)

def update_missing_genres_embeddings(session):
    """Tạo embedding cho các genre chưa có."""
    genres = session.query(HKT_Genres).filter(
        (HKT_Genres.embedding.is_(None)) |
        (HKT_Genres.embedding == "") |
        (HKT_Genres.embedding == "null")
    ).all()
    print(f"🎭 Found {len(genres)} genres missing embeddings")

    for g in genres:
        text_input = clean_text(g.genre_name or "")
        if not text_input:
            continue
        try:
            emb = embedding_model.encode(text_input, normalize_embeddings=True)
            emb = np.array(emb, dtype='float32')
            g.embedding = json.dumps(emb.tolist())
        except Exception as e:
            print(f"⚠️ Embedding failed for genre_id={g.genre_id}: {e}")

    try:
        session.commit()
        print("Genre embeddings updated successfully.")
    except Exception as e:
        session.rollback()
        print("Commit failed for genres:", e)


def update_missing_recommendation_embeddings(session):
    """Tạo embedding cho các recommendation chưa có."""
    recs = session.query(HKT_Recommendations).filter(
        (HKT_Recommendations.embedding.is_(None)) |
        (HKT_Recommendations.embedding == "") |
        (HKT_Recommendations.embedding == "null")
    ).all()
    print(f"🔗 Found {len(recs)} recommendations missing embeddings")

    for r in recs:
        text_input = f"Movie {r.movie_id} recommended {r.recommended_movie_id}"
        try:
            emb = embedding_model.encode(clean_text(text_input), normalize_embeddings=True)
            emb = np.array(emb, dtype='float32')
            r.embedding = json.dumps(emb.tolist())
        except Exception as e:
            print(f"⚠️ Embedding failed for recommendation_id={r.id}: {e}")

    try:
        session.commit()
        print("Recommendation embeddings updated successfully.")
    except Exception as e:
        session.rollback()
        print("Commit failed for recommendations:", e)


# ------------------------
# Build both indexes at startup (if DB present)
# ------------------------
if Session:
    s = Session()
    try:
        update_missing_embeddings(s)  # <-- thêm dòng này
        build_faiss_index(s)
        # build_watch_faiss_index(s)
    except Exception as e:
        print("Error building FAISS indexes at startup:", e)
    finally:
        s.close()
# ------------------------
# Retrieval helpers
# ------------------------

def retrieve_movies_with_embedding(session, user_message, limit=5):
    global faiss_index, movie_id_map
    if not faiss_index or not movie_id_map:
        return []
    
    if not user_message.strip():
        return session.query(HKT_Movies).order_by(HKT_Movies.popularity.desc()).limit(limit).all()
    
    user_vec = np.array(embedding_model.encode(user_message), dtype='float32').reshape(1, -1)
    if user_vec.shape[1] != faiss_index.d:
        build_faiss_index(session)
    faiss.normalize_L2(user_vec)
    D, I = faiss_index.search(user_vec, limit)
    
    top_ids = [movie_id_map[i] for i in I[0] if i < len(movie_id_map)]
    results = session.query(HKT_Movies).filter(HKT_Movies.movie_id.in_(top_ids)).all()
    results_sorted = sorted(results, key=lambda m: top_ids.index(m.movie_id))
    return results_sorted

def save_movie_to_db(session, movie_data):
    """
    Lưu hoặc cập nhật movie vào HKT_Movies với embedding.
    """
    try:
        # Kiểm tra movie đã tồn tại chưa
        existing = session.query(HKT_Movies).filter_by(movie_id=movie_data['id']).first()
        
        # Chuẩn bị text để tạo embedding
        title = movie_data.get('title') or ''
        overview = movie_data.get('overview') or ''
        text_for_embedding = f"{title} {overview}".strip()
        
        if not text_for_embedding:
            print(f"[save_movie_to_db] Movie {movie_data['id']} có title + overview trống, bỏ qua embedding.")
            embedding_str = None
        else:
            embedding_vec = get_embedding(text_for_embedding)
            if embedding_vec:
                embedding_str = json.dumps(embedding_vec)
            else:
                print(f"[save_movie_to_db] Không tạo được embedding cho movie {title}")
                embedding_str = None
        
        # Cập nhật hoặc tạo mới movie
        if existing:
            existing.title = title
            existing.original_title = movie_data.get('original_title')
            existing.overview = overview
            existing.release_date = movie_data.get('release_date')
            existing.vote_average = movie_data.get('vote_average')
            existing.vote_count = movie_data.get('vote_count')
            existing.popularity = movie_data.get('popularity')
            existing.original_language = movie_data.get('original_language')
            existing.embedding = embedding_str
        else:
            new_movie = HKT_Movies(
                movie_id=movie_data['id'],
                title=title,
                original_title=movie_data.get('original_title'),
                overview=overview,
                release_date=movie_data.get('release_date'),
                vote_average=movie_data.get('vote_average'),
                vote_count=movie_data.get('vote_count'),
                popularity=movie_data.get('popularity'),
                original_language=movie_data.get('original_language'),
                embedding=embedding_str
            )
            session.add(new_movie)
        
        # Commit session sau khi lưu
        session.commit()


    except Exception as e:
        session.rollback()  # rollback nếu lỗi


# ------------------------
# API endpoints
# ------------------------

@app.route('/api/admin/reembed', methods=['POST'])
def reembed_movies():
    if not Session:
        return jsonify({"error": "DB session not initialized"}), 500
    session = Session()
    try:
        movies = session.query(HKT_Movies).filter(
            (HKT_Movies.embedding.is_(None)) |
            (HKT_Movies.embedding == "") |
            (HKT_Movies.embedding == "null") |
            (HKT_Movies.embedding == "None") |
            (HKT_Movies.embedding.like("%null%")) |
            (HKT_Movies.embedding.like("%None%"))
        ).all()
        print(f"🔎 Found {len(movies)} movies missing embeddings")
        if len(movies) > 0:
            print("👀 Ví dụ 5 dòng đầu:")
            for m in movies[:5]:
                print(f"movie_id={m.movie_id}, title={m.title}, overview_len={len(m.overview or '')}")


        for movie in movies:
            text_input = f"{movie.title or ''}. {movie.overview or ''}".strip()
            if not text_input:
                continue

            try:
                emb = embedding_model.encode(text_input, normalize_embeddings=True)
                print(f"✅ Movie {movie.movie_id} embedded: dim={len(emb)}")

                emb = np.array(emb, dtype='float32')
                movie.embedding = json.dumps(emb.tolist())
            except Exception as e:
                print(f"⚠️ Embedding failed for movie_id={movie.movie_id}: {e}")

        session.commit()
        print("✅ Embeddings updated successfully.")
        return jsonify({"message": f"Re-embedded {len(movies)} movies successfully."})
    except Exception as e:
        session.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()


@app.route('/api/search', methods=['GET'])
def search_movies():
    query = request.args.get('query', '')
    page = request.args.get('page', 1)
    
    url = f'https://api.themoviedb.org/3/search/movie'
    params = {
        'api_key': TMDB_API_KEY,
        'query': query,
        'page': page,
        'language': 'vi-VN'
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if Session and 'results' in data:
        session = Session()
        for movie_data in data['results']:
            save_movie_to_db(session, movie_data)
        session.commit()
        session.close()
    
    return jsonify(data)

@app.route('/api/discover', methods=['GET'])
def discover_movies():
    genre = request.args.get('genre', '')
    page = request.args.get('page', 1)
    
    url = f'https://api.themoviedb.org/3/discover/movie'
    params = {
        'api_key': TMDB_API_KEY,
        'page': page,
        'language': 'vi-VN',
        'sort_by': 'popularity.desc'
    }
    
    if genre:
        params['with_genres'] = genre
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if Session and 'results' in data:
        session = Session()
        for movie_data in data['results']:
            save_movie_to_db(session, movie_data)
        session.commit()
        session.close()
    
    return jsonify(data)

@app.route('/api/genres', methods=['GET'])
def get_genres():
    url = f'https://api.themoviedb.org/3/genre/movie/list'
    params = {'api_key': TMDB_API_KEY, 'language': 'vi-VN'}
    response = requests.get(url, params=params)
    return jsonify(response.json())

@app.route('/api/movie/<int:movie_id>', methods=['GET'])
def get_movie_details(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}'
    params = {'api_key': TMDB_API_KEY, 'language': 'vi-VN'}
    response = requests.get(url, params=params)
    data = response.json()
    
    if Session and response.status_code == 200:
        session = Session()
        save_movie_to_db(session, data)
        session.commit()
        session.close()
    
    return jsonify(data)




# ------------------------
# Nhận thông tin người dùng từ frontend
# ------------------------

# ------------------------
# AI Chat (RAG) với local embedding
# ------------------------
@app.route('/api/chat', methods=['POST'])
def ai_chat():
    data = request.json
    user_message = data.get('message', '')
    conversation_history = data.get('history', [])

    if not OPENAI_API_KEY or not TMDB_API_KEY:
        return jsonify({'error': 'API keys not configured'}), 500
    
    retrieved_movies = []

    # Retrieval semantic từ DB
    if Session:
        session = Session()
        try:
            retrieved_movies = retrieve_movies_with_embedding(session, user_message, limit=5)
        finally:
            session.close()
    
    context_text = ""
    if retrieved_movies:
        context_items = []
        for m in retrieved_movies:
            try:
                genres = json.loads(m.genres) if m.genres else []
                if isinstance(genres, list):
                    genres_str = ", ".join([g.get("name", str(g)) if isinstance(g, dict) else str(g) for g in genres])
                else:
                    genres_str = str(genres)
            except:
                genres_str = ""
            
            context_items.append(
                f"[{m.imdb_id}] {m.title} ({m.release_date or 'N/A'}) - "
                f"⭐ {m.vote_average or 0}/10 | {genres_str} | {m.original_language}\n"
                f"Tóm tắt: {m.overview or 'Không có mô tả.'}\n"
            )
        context_text = "\n".join(context_items)
    else:
        context_text = "Không có phim nào khớp với truy vấn."

    system_prompt = f"""

You are a movie consultant.

Here is a list of movies found by the system that are related to the user description:

{context_text}

Your task:

- If the user description matches the movies in the list, select the most suitable movies.

- If the user just asks a simple, general movie question, then based on the user's personality (if the user's personality matches, otherwise, based on the user's prompt and generate movies for them) and select the appropriate movie.

- If the user asks about movies they have watched, access the watch_history table to get a list of movies they have watched

- If you just chat or ask random questions, answer briefly and friendly, DO NOT suggest movies.

Must always return valid JSON in one of the following 2 structures:

1 If you want to suggest a movie:
{{
"message": "Friendly answer in English (1-2 sentences).",
"suggest_movies": true,
"movies_ids": [imdb_id1, imdb_id2, ...],
"explanation": "Brief explanation of why you chose these movies."
}}

2 If no movies match:
{{
"message": "Friendly answer in English.",
"suggest_movies": false
}}

Output only JSON, no other text.
"""

    messages = [{'role': 'system', 'content': system_prompt}]
    messages.extend(conversation_history)
    messages.append({'role': 'user', 'content': user_message})

    # 3️⃣ Gọi OpenAI
    openai_url = 'https://api.openai.com/v1/chat/completions'
    headers = {'Authorization': f'Bearer {OPENAI_API_KEY}', 'Content-Type': 'application/json'}
    payload = {'model': 'gpt-4-turbo', 'messages': messages, 'temperature': 1}

    ai_response = requests.post(openai_url, headers=headers, json=payload)
    ai_data = ai_response.json()
    ai_message = ai_data['choices'][0]['message']['content'] if 'choices' in ai_data and len(ai_data['choices'])>0 else ai_data.get('error', {}).get('message', str(ai_data))

    # 4️⃣ Parse JSON an toàn
    def safe_json_load(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # fallback: tìm JSON trong text nếu AI trả thêm text khác
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    return {}
            return {}

    ai_json = safe_json_load(ai_message)
    suggest_movies = ai_json.get('suggest_movies', False)
    movies_ids = ai_json.get('movies_ids', [])

    # 5️⃣ Lấy thông tin phim từ TMDB nếu AI gợi ý
    movies = []
    if suggest_movies and movies_ids:
        for tmdb_id in movies_ids:
            resp = requests.get(
                f"https://api.themoviedb.org/3/movie/{tmdb_id}",
                params={'api_key': TMDB_API_KEY, 'language': 'vi-VN'}
            )
            if resp.status_code == 200:
                movies.append(resp.json())

    # 6️⃣ Lưu movies vào DB nếu cần
    if Session and movies:
        session = Session()
        try:
            for movie_data in movies:
                save_movie_to_db(session, movie_data)
            session.commit()
            # Build FAISS index sau khi cập nhật
            build_faiss_index(session)
        finally:
            session.close()

    # Sau khi có danh sách 'movies' (list phim đã lấy)
    if not movies and suggest_movies:
        # Nếu AI chỉ ra imdb_id (hoặc movie_id) mà không fetch từ TMDB
        if Session:
            session = Session()
            try:
                from sqlalchemy.orm import joinedload
                from sqlalchemy import select
                from sqlalchemy.sql import text

                # Join HKT_Movies với HKT_Images để lấy ảnh
                query = text("""
                    SELECT m.movie_id, m.title, m.overview, i.poster_url, i.backdrop_url
                    FROM dbo.HKT_Movies m
                    LEFT JOIN dbo.HKT_Images i ON m.movie_id = i.movie_id
                    WHERE m.movie_id IN :ids
                """)
                rows = session.execute(query, {"ids": tuple(movies_ids)}).fetchall()
                for r in rows:
                    movies.append({
                        "movie_id": r.movie_id,
                        "title": r.title,
                        "overview": r.overview,
                        "poster_url": r.poster_url,
                        "backdrop_url": r.backdrop_url
                    })
            finally:
                session.close()


    # 7️⃣ Trả về response
    return jsonify({
        'message': ai_json.get('message', ai_message),
        'suggest_movies': suggest_movies,
        'movies': movies,
        'explanation': ai_json.get('explanation', '')
    })

# ------------------------
# Health check
# ------------------------
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'tmdb_configured': bool(TMDB_API_KEY),
        'openai_configured': bool(OPENAI_API_KEY),
        'db_configured': bool(DB_CONNECTION_STRING)
    })

@app.route('/api/embed_missing', methods=['POST'])
def embed_missing_movies():
    """Tạo embedding cho các phim chưa có embedding."""
    if not Session:
        return jsonify({"error": "DB session not initialized"}), 500

    session = Session()
    try:
        # 1️⃣ Lấy danh sách phim chưa có embedding
        result = session.execute(text("""
            SELECT TOP (1000) movie_id, title, overview
            FROM dbo.HKT_Movies
            WHERE embedding IS NULL OR embedding = '' OR embedding = 'null'
        """))
        movies = result.fetchall()
        print(f"📦 Found {len(movies)} movies without embeddings")

        # 2️⃣ Load model embedding (có sẵn)
        model = embedding_model
        updates = []

        # 3️⃣ Tính embedding
        for movie_id, title, overview in movies:
            text_input = f"{title or ''}. {overview or ''}".strip()
            if not text_input:
                continue
            emb = model.encode(text_input, normalize_embeddings=True)
            emb_json = json.dumps(emb.tolist())
            updates.append((emb_json, movie_id))

        # 4️⃣ Ghi vào DB
        for emb, mid in updates:
            session.execute(
                text("UPDATE dbo.HKT_Movies SET embedding = :emb WHERE movie_id = :mid"),
                {"emb": emb, "mid": mid}
            )
        session.commit()

        print(f"✅ Updated {len(updates)} embeddings successfully.")
        return jsonify({"message": f"Updated {len(updates)} embeddings successfully."})

    except Exception as e:
        session.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()


# ------------------------
# Run app
# ------------------------
if __name__ == '__main__':
    # 🔹 Tự động cập nhật embedding nếu còn null
    if Session:
        s = Session()
        try:
            update_missing_embeddings(s)
            update_missing_genres_embeddings(s)       
            update_missing_recommendation_embeddings(s)
            build_faiss_index(s)
        except Exception as e:
            print("Error updating embeddings at startup:", e)
        finally:
            s.close()

    app.run(host='0.0.0.0', port=8080, debug=True)

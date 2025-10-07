from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import requests
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
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
# ------------------------
# Flask setup
# ------------------------
app = Flask(__name__)
CORS(app)
load_dotenv()

TMDB_API_KEY = os.getenv('TMDB_API_KEY')
OPENAI_API_KEY = ""
DB_CONNECTION_STRING = os.getenv('DB_CONNECTION_STRING')

print("OPENAI_API_KEY:", OPENAI_API_KEY)
print("TMDB_API_KEY:", TMDB_API_KEY)

# ------------------------
# Database setup
# ------------------------
engine = None
Session = None
if DB_CONNECTION_STRING and DB_CONNECTION_STRING.startswith('mssql'):
    try:
        engine = create_engine(DB_CONNECTION_STRING)
        Session = sessionmaker(bind=engine)
    except Exception as e:
        print(f"Database connection error: {e}")

Base = declarative_base()

# ------------------------
# Embedding model
# ------------------------
embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")


# ------------------------
# Movie model
# ------------------------
class Movie(Base):
    __tablename__ = 'movies'
    
    id = Column(Integer, primary_key=True)
    tmdb_id = Column(Integer, unique=True, nullable=False)
    title = Column(String(500), nullable=False)
    original_title = Column(String(500))
    overview = Column(Text)
    poster_path = Column(String(500))
    backdrop_path = Column(String(500))
    release_date = Column(String(50))
    vote_average = Column(Float)
    vote_count = Column(Integer)
    popularity = Column(Float)
    genres = Column(Text)
    original_language = Column(String(10))
    embedding = Column(Text)  # Lưu vector dạng JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

if engine:
    Base.metadata.create_all(engine)


# ------------------------
# Watch history model
# ------------------------
class WatchHistory(Base):
    __tablename__ = 'watch_history'
    
    id = Column(Integer, primary_key=True)
    movie_id = Column(Integer, nullable=False)  # tmdb_id của phim
    title = Column(String(500), nullable=False)
    watched_at = Column(DateTime, default=datetime.utcnow)
    note = Column(Text)  # ghi chú (tùy chọn)
    embedding = Column(Text)

# Tạo bảng (nếu chưa có)
if engine:
    Base.metadata.create_all(engine)

def save_watch_history(session, movie_id, title, note=""):
    text_for_embedding = f"{title}. {note or ''}"
    embedding_vec = get_embedding(text_for_embedding)
    embedding_str = json.dumps(embedding_vec)

    history = WatchHistory(
        movie_id=movie_id,
        title=title,
        note=note,
        embedding=embedding_str
    )
    session.add(history)
    session.commit()

watch_faiss_index = None
watch_id_map = []

def build_watch_faiss_index(session):
    global watch_faiss_index, watch_id_map
    watched = session.query(WatchHistory).filter(WatchHistory.embedding.isnot(None)).all()
    if not watched:
        watch_faiss_index = None
        watch_id_map = []
        return

    embeddings = np.array([json.loads(w.embedding) for w in watched], dtype='float32')
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]

    watch_faiss_index = faiss.IndexFlatIP(dim)
    watch_faiss_index.add(embeddings)
    watch_id_map = [w.id for w in watched]

    print(f"FAISS index built with {len(watch_id_map)} watch_history items")

session = Session()
build_watch_faiss_index(session)
session.close()


def retrieve_from_watch_history(session, user_message, limit=3):
    global watch_faiss_index, watch_id_map
    if not watch_faiss_index:
        return []

    user_vec = embedding_model.encode(user_message).reshape(1, -1).astype('float32')
    faiss.normalize_L2(user_vec)
    D, I = watch_faiss_index.search(user_vec, limit)

    ids = [watch_id_map[i] for i in I[0] if i < len(watch_id_map)]
    results = session.query(WatchHistory).filter(WatchHistory.id.in_(ids)).all()
    return results

# ------------------------
# Insert sample watch history (if empty)
# ------------------------
if Session:
    session = Session()
    existing = session.query(WatchHistory).count()
    if existing == 0:
        sample_movies = [
            {"movie_id": 603692, "title": "John Wick: Chapter 4", "note": "Phim hành động căng thẳng."},
            {"movie_id": 872585, "title": "Oppenheimer", "note": "Phim tiểu sử về nhà khoa học Mỹ."},
            {"movie_id": 569094, "title": "Spider-Man: Across the Spider-Verse", "note": "Phim hoạt hình siêu anh hùng tuyệt đẹp."}
        ]
        for m in sample_movies:
            session.add(WatchHistory(**m))
        session.commit()
        print("✅ Sample watch_history inserted.")
    session.close()
# ------------------------
# Helper functions
# ------------------------
faiss_index = None
movie_id_map = []  
def build_faiss_index(session):
    """Build FAISS index từ movies trong DB"""
    global faiss_index, movie_id_map
    movies = session.query(Movie).filter(Movie.embedding.isnot(None)).all()
    if not movies:
        faiss_index = None
        movie_id_map = []
        return
    
    embeddings = np.array([json.loads(m.embedding) for m in movies], dtype='float32')
    faiss.normalize_L2(embeddings)  # chuẩn hóa để inner product = cosine similarity
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)
    
    # Lưu mapping từ FAISS index -> tmdb_id
    movie_id_map = [m.tmdb_id for m in movies]
    print(f"FAISS index built with {len(movie_id_map)} movies")

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'<[^>]+>', '', text)            # loại bỏ HTML
    text = re.sub(r'[\n\r]+', ' ', text)          # replace newline bằng space
    text = re.sub(r'[^a-z0-9\sàáâãèéêìíòóôõùúăđĩũơưăạả...', ' ', text)  # giữ chữ + số + tiếng Việt
    text = re.sub(r'\s+', ' ', text)              # normalize whitespace
    return text

def get_embedding(text):
    """Trả về vector embedding dạng list"""
    text = clean_text(text)
    if text:
        vector = embedding_model.encode(text).tolist()
    else:
        vector = [0.0] * 384  # default nếu không có text
    return vector

def save_movie_to_db(session, movie_data):
    """Lưu phim vào database kèm embedding"""
    try:
        existing_movie = session.query(Movie).filter_by(tmdb_id=movie_data['id']).first()
        genres_str = json.dumps(movie_data.get('genres', [])) if 'genres' in movie_data else json.dumps(movie_data.get('genre_ids', []))
        
        # Tạo embedding từ title + overview
        text_for_embedding = (movie_data.get('title', '') or '') + " " + (movie_data.get('overview', '') or '')
        embedding_vec = get_embedding(text_for_embedding)
        embedding_str = json.dumps(embedding_vec)
        
        if existing_movie:
            existing_movie.title = movie_data.get('title', '')
            existing_movie.original_title = movie_data.get('original_title', '')
            existing_movie.overview = movie_data.get('overview', '')
            existing_movie.poster_path = movie_data.get('poster_path', '')
            existing_movie.backdrop_path = movie_data.get('backdrop_path', '')
            existing_movie.release_date = movie_data.get('release_date', '')
            existing_movie.vote_average = movie_data.get('vote_average', 0)
            existing_movie.vote_count = movie_data.get('vote_count', 0)
            existing_movie.popularity = movie_data.get('popularity', 0)
            existing_movie.genres = genres_str
            existing_movie.original_language = movie_data.get('original_language', '')
            existing_movie.embedding = embedding_str
        else:
            new_movie = Movie(
                tmdb_id=movie_data['id'],
                title=movie_data.get('title', ''),
                original_title=movie_data.get('original_title', ''),
                overview=movie_data.get('overview', ''),
                poster_path=movie_data.get('poster_path', ''),
                backdrop_path=movie_data.get('backdrop_path', ''),
                release_date=movie_data.get('release_date', ''),
                vote_average=movie_data.get('vote_average', 0),
                vote_count=movie_data.get('vote_count', 0),
                popularity=movie_data.get('popularity', 0),
                genres=genres_str,
                original_language=movie_data.get('original_language', ''),
                embedding=embedding_str
            )
            session.add(new_movie)
    except Exception as e:
        print(f"Error saving movie: {e}")



def retrieve_movies_with_embedding(session, user_message, limit=5):
    global faiss_index, movie_id_map
    if not faiss_index or not movie_id_map:
        return []  # chưa build index
    
    # Nếu message rỗng → trả về top phim hot (popularity cao)
    if not user_message.strip():
        top_movies = session.query(Movie).order_by(Movie.popularity.desc()).limit(limit).all()
        return top_movies
    
    # Embedding người dùng
    user_vec = embedding_model.encode(user_message).reshape(1, -1).astype('float32')
    faiss.normalize_L2(user_vec)
    
    # Tìm top k
    D, I = faiss_index.search(user_vec, limit)  # D = similarity, I = index FAISS
    top_tmdb_ids = [movie_id_map[i] for i in I[0] if i < len(movie_id_map)]
    
    # Lấy chi tiết phim từ DB
    top_movies = session.query(Movie).filter(Movie.tmdb_id.in_(top_tmdb_ids)).all()
    # Sắp xếp theo thứ tự FAISS trả về
    top_movies_sorted = sorted(top_movies, key=lambda m: top_tmdb_ids.index(m.tmdb_id))
    return top_movies_sorted

# ------------------------
# API endpoints
# ------------------------
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
@app.route('/api/user-info', methods=['POST'])
def receive_user_info():
    data = request.get_json()
    username = data.get('username')
    token = data.get('token')
    tinh_cach = data.get('tinh_cach')

    print(f"📩 Nhận user info: {username=}, {token=}, {tinh_cach=}")

    # Bạn có thể lưu vào biến toàn cục, DB, hoặc cache nếu cần
    # Ở đây ta chỉ trả lại xác nhận
    return jsonify({
        'status': 'received',
        'user': username,
        'tinh_cach': tinh_cach
    })

# ------------------------
# AI Chat (RAG) với local embedding
# ------------------------
@app.route('/api/chat', methods=['POST'])
def ai_chat():
    data = request.json
    user_message = data.get('message', '')
    conversation_history = data.get('history', [])
    user_name = data.get('user')
    user_tinh_cach = data.get('tinh_cach')
    print(f" AI chat từ user: {user_name}, tính cách: {user_tinh_cach}")

    if not OPENAI_API_KEY or not TMDB_API_KEY:
        return jsonify({'error': 'API keys not configured'}), 500
    
    if Session:
        session = Session()
        watched = session.query(WatchHistory).all()
        print("\n🎬 Watch history data:")
    for w in watched:
        print(f"- {w.title} (tmdb_id={w.movie_id}) | Ghi chú: {w.note}")
    watch_text = ""
    if Session:
        session = Session()
        watched = session.query(WatchHistory).all()
    if watched:
        watch_text = "\n".join([f"- {w.title} (Ghi chú: {w.note or ''})" for w in watched])
    session.close()
    # 1️⃣ Retrieval semantic từ DB
    retrieved_movies = []
    if Session:
        session = Session()
        retrieved_movies = retrieve_movies_with_embedding(session, user_message, limit=5)
        session.close()
    
    context_text = ""
    if retrieved_movies:
        context_items = []
        for m in retrieved_movies:
            try:
                genres = json.loads(m.genres) if m.genres else []
                if isinstance(genres, list):
                    genres_str = ", ".join([g.get("name", str(g)) for g in genres])
                else:
                    genres_str = str(genres)
            except:
                genres_str = ""
            
            context_items.append(
                f"[{m.tmdb_id}] {m.title} ({m.release_date or 'N/A'}) - "
                f"⭐ {m.vote_average or 0}/10 | {genres_str} | {m.original_language}\n"
                f"Tóm tắt: {m.overview or 'Không có mô tả.'}\n"
            )
        context_text = "\n".join(context_items)
    else:
        context_text = "Không có phim nào khớp với truy vấn."

    
    system_prompt = f"""
Người dùng hiện tại có thông tin sau:
- Tên: {user_name or "Ẩn danh"}
- Tính cách: {user_tinh_cach or "Không rõ"}

Danh sách phim người dùng đã xem:
{watch_text or "Chưa có phim nào trong lịch sử."}


Bạn là chuyên gia tư vấn phim.
Dưới đây là danh sách phim được hệ thống tìm thấy có liên quan đến mô tả người dùng:

{context_text}

Nhiệm vụ của bạn:
- Nếu mô tả người dùng tương ứng với phim trong danh sách, chọn ra những phim phù hợp nhất.
- Nếu người dùng chỉ hỏi câu hỏi về film đơn giản mang tính chung chung thì dựa vào tính cách của người dùng (nếu tính cách của người dùng hợp còn không thì dựa vào prompt của user và generate phim cho họ) và chọn ra phim phù hợp.
- Nếu người dùng hỏi về các phim đã xem bạn hãy truy cập bảng watch_history để lấy danh sách các phim đã xem
- Nếu chỉ trò chuyện hoặc hỏi linh tinh, trả lời ngắn gọn, thân thiện, KHÔNG gợi ý phim.

Phải luôn trả về JSON hợp lệ theo đúng 1 trong 2 cấu trúc sau:

1 Nếu bạn muốn gợi ý phim:
{{
    "message": "Câu trả lời thân thiện bằng tiếng Việt (1-2 câu).",
    "suggest_movies": true,
    "movies_ids": [tmdb_id1, tmdb_id2, ...],
    "explanation": "Giải thích ngắn gọn tại sao chọn những phim này."
}}

2 Nếu không có phim nào phù hợp:
{{
    "message": "Câu trả lời thân thiện bằng tiếng Việt.",
    "suggest_movies": false
}}

Chỉ xuất ra JSON, không kèm văn bản khác.
"""

    messages = [{'role': 'system', 'content': system_prompt}]
    messages.extend(conversation_history)
    messages.append({'role': 'user', 'content': user_message})

    # 3️⃣ Gọi OpenAI
    openai_url = 'https://api.openai.com/v1/chat/completions'
    headers = {'Authorization': f'Bearer {OPENAI_API_KEY}', 'Content-Type': 'application/json'}
    payload = {'model': 'gpt-3.5-turbo', 'messages': messages, 'temperature': 0.5}

    ai_response = requests.post(openai_url, headers=headers, json=payload)
    ai_data = ai_response.json()
    ai_message = ai_data['choices'][0]['message']['content']

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
        for movie_data in movies:
            save_movie_to_db(session, movie_data)
        session.commit()
        
        # Build FAISS index sau khi cập nhật
        build_faiss_index(session)
        
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

# ------------------------
# Run app
# ------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

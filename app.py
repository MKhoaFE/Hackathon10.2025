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
# ------------------------
# Flask setup
# ------------------------
app = Flask(__name__)
CORS(app)
load_dotenv()

TMDB_API_KEY = os.getenv('TMDB_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
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
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # local embedding

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
# Helper functions
# ------------------------
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
    """Retrieval semantic bằng embedding nhanh & chính xác hơn"""
    movies = session.query(Movie).filter(Movie.embedding.isnot(None)).all()
    if not movies:
        return []
    
    # Lấy embeddings từ DB
    embeddings = np.array([json.loads(m.embedding) for m in movies])
    embeddings = normalize(embeddings)  # Chuẩn hóa vector để cosine chính xác
    
    # Vector người dùng
    user_vec = embedding_model.encode(user_message).reshape(1, -1)
    user_vec = normalize(user_vec)
    
    # Tính cosine similarity hàng loạt
    sims = np.dot(embeddings, user_vec.T).flatten()
    
    # Lấy top phim có độ tương đồng cao nhất
    top_indices = np.argsort(sims)[::-1][:limit]
    top_movies = [movies[i] for i in top_indices]
    
    return top_movies


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

    
    # 2️⃣ Tạo system prompt
    system_prompt = f"""
Người dùng hiện tại có thông tin sau:
- Tên: {user_name or "Ẩn danh"}
- Tính cách: {user_tinh_cach or "Không rõ"}

Bạn là chuyên gia tư vấn phim.
Dưới đây là danh sách phim được hệ thống tìm thấy có liên quan đến mô tả người dùng:

{context_text}

Nhiệm vụ của bạn:
- Nếu mô tả người dùng tương ứng với phim trong danh sách, chọn ra những phim phù hợp nhất.
- Nếu người dùng chỉ hỏi câu hỏi về film đơn giản mang tính chung chung thì dựa vào tính cách của người dùng và chọn ra phim phù hợp.
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
    
    # 4️⃣ Lấy phim TMDB nếu AI gợi ý
    try:
        ai_json = json.loads(ai_message)
        suggest_movies = ai_json.get('suggest_movies', False)
        movies_ids = ai_json.get('movies_ids', [])
        
        movies = []
        if suggest_movies and movies_ids:
            for tmdb_id in movies_ids:
                resp = requests.get(f"https://api.themoviedb.org/3/movie/{tmdb_id}", params={'api_key': TMDB_API_KEY, 'language': 'vi-VN'})
                if resp.status_code == 200:
                    movies.append(resp.json())
            
            # Lưu movies vào DB
            if Session and movies:
                session = Session()
                for movie_data in movies:
                    save_movie_to_db(session, movie_data)
                session.commit()
                session.close()
        
        return jsonify({
            'message': ai_json.get('message', ai_message),
            'suggest_movies': suggest_movies,
            'movies': movies,
            'explanation': ai_json.get('explanation', '')
        })
        
    except json.JSONDecodeError:
        return jsonify({'message': ai_message, 'suggest_movies': False, 'movies': []})

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

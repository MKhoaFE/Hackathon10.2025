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


app = Flask(__name__)
CORS(app)

# API keys
# Load .env
load_dotenv()

TMDB_API_KEY = os.getenv('TMDB_API_KEY')
OPENAI_API_KEY = os.getenv('OPEN_API_KEY')
print(OPENAI_API_KEY)
print("TMDB_API_KEY:", os.getenv("TMDB_API_KEY"))
# DB connection
DB_CONNECTION_STRING = os.getenv('DB_CONNECTION_STRING')
engine = None
Session = None

if DB_CONNECTION_STRING and DB_CONNECTION_STRING.startswith('mssql'):
    try:
        engine = create_engine(DB_CONNECTION_STRING)
        Session = sessionmaker(bind=engine)
    except Exception as e:
        print(f"Database connection error: {e}")

Base = declarative_base()

# Movie model
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
    created_at = Column(DateTime, default=datetime.utcnow)

if engine:
    Base.metadata.create_all(engine)

# ------------------------
# Helper functions
# ------------------------

def save_movie_to_db(session, movie_data):
    """Lưu phim vào database"""
    try:
        existing_movie = session.query(Movie).filter_by(tmdb_id=movie_data['id']).first()
        genres_str = json.dumps(movie_data.get('genres', [])) if 'genres' in movie_data else json.dumps(movie_data.get('genre_ids', []))
        
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
                original_language=movie_data.get('original_language', '')
            )
            session.add(new_movie)
    except Exception as e:
        print(f"Error saving movie: {e}")

def retrieve_movies_from_db(session, user_message, limit=10):
    """Lấy phim từ DB dựa trên từ khóa trong prompt"""
    query = session.query(Movie)
    keywords = user_message.lower().split()
    
    for kw in keywords:
        query = query.filter(
            (Movie.title.ilike(f'%{kw}%')) |
            (Movie.overview.ilike(f'%{kw}%')) |
            (Movie.genres.ilike(f'%{kw}%'))
        )
    
    return query.limit(limit).all()

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
# AI Chat (RAG)
# ------------------------

@app.route('/api/chat', methods=['POST'])
def ai_chat():
    data = request.json
    user_message = data.get('message', '')
    conversation_history = data.get('history', [])
    
    if not OPENAI_API_KEY or not TMDB_API_KEY:
        return jsonify({'error': 'API keys not configured'}), 500
    
    # 1️⃣ Retrieval từ DB
    retrieved_movies = []
    if Session:
        session = Session()
        retrieved_movies = retrieve_movies_from_db(session, user_message, limit=5)
        session.close()
    
    context_text = ""
    if retrieved_movies:
        context_text = "\n".join([f"{m.title} ({m.release_date}): {m.overview}" for m in retrieved_movies])
    
    # 2️⃣ Tạo system prompt
    system_prompt = f"""
Bạn là chuyên gia tư vấn phim. Dựa vào ngữ cảnh sau, gợi ý phim cho người dùng:
{context_text}

Nếu người dùng hỏi về sở thích, trả về JSON:
{{
    "message": "Câu trả lời thân thiện",
    "suggest_movies": true,
    "movies_ids": [tmdb_id1, tmdb_id2, ...],
    "explanation": "Giải thích ngắn về lý do gợi ý"
}}

Nếu chỉ trò chuyện bình thường, trả về:
{{
    "message": "Câu trả lời",
    "suggest_movies": false
}}
"""
    
    messages = [{'role': 'system', 'content': system_prompt}]
    messages.extend(conversation_history)
    messages.append({'role': 'user', 'content': user_message})
    
    # 3️⃣ Gọi OpenAI
    openai_url = 'https://api.openai.com/v1/chat/completions'
    headers = {'Authorization': f'Bearer {OPENAI_API_KEY}', 'Content-Type': 'application/json'}
    
    payload = {'model': 'gpt-3.5-turbo', 'messages': messages, 'temperature': 0.7}
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

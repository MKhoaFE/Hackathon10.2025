🎬 AI-Powered Movie Web App
- This is a movie discovery and recommendation application that combines a Flask backend with a React frontend. The application integrates with The Movie Database (TMDB) API to fetch movie information and uses OpenAI's GPT-3.5-turbo for AI-powered movie recommendations based on natural language prompts. The system optionally stores movie data in a Microsoft SQL Server database using SQLAlchemy ORM.
------------------------------------------------------------------------
🚀 Overview
This is a web application that allows users to:

- 🎥 Browse detailed movie information, trailers, and cast.

- 🤖 Chat with an AI assistant to:
  + Recommend movies based on mood, genre, or interests.
  + Summarize or explain movie plots.
  + Discuss movie endings and storylines.

- 💾 Save favorite movies and track viewing history.
--------------------------------------------------------------------------
🧩 Technical
- Frontend: ReactJS, Vite, TailwindCSS, Axios
- Backend (BE): Flask (Python), Flask-CORS, SQLAlchemy, Dotenv, Requests
- Database: Microsoft SQL Server (SSMS)
- AI & APIs: OpenAI API, TMDB API
----------------------------------------------------------------------------
Key Features:

- Movie search with TMDB integration
- Genre-based filtering
- AI-powered movie recommendations using natural language (supports Vietnamese prompts)
- Responsive movie grid with posters and ratings
- Clean, modern UI with dark theme


----------------------------------------------------------------------------
HOW TO SETUP?
set up virtual environment .venv (if you want)
- run this in terminal:
  + python -m venv .venv
  + .\venv\Scripts\activate
  + python -m pip install -r requirements.txt
 
Or run without environment:
- run this in terminal:
  + pip install -r requirements.txt
 
TO RUN PROJECT:
Backend: python app.py
Frontend: npm run dev



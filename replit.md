# Overview

This is a movie discovery and recommendation application that combines a Flask backend with a React frontend. The application integrates with The Movie Database (TMDB) API to fetch movie information and uses OpenAI's GPT-3.5-turbo for AI-powered movie recommendations based on natural language prompts. The system optionally stores movie data in a Microsoft SQL Server database using SQLAlchemy ORM.

**Key Features**:
- Movie search with TMDB integration
- Genre-based filtering
- AI-powered movie recommendations using natural language (supports Vietnamese prompts)
- Responsive movie grid with posters and ratings
- Clean, modern UI with dark theme

**Last Updated**: October 5, 2025

# User Preferences

Preferred communication style: Simple, everyday language (Vietnamese and English).

# System Architecture

## Frontend Architecture

**Technology Stack**: React 19 with Vite 7 as the build tool
- **Styling**: Tailwind CSS v3 for utility-first styling approach
- **HTTP Client**: Axios for API communication with the backend
- **Development Server**: Configured to run on port 5000 with host set to '0.0.0.0' for accessibility
- **UI Features**: 
  - Tab-based navigation (Search / AI Recommendations)
  - Real-time movie search
  - Genre filtering with Vietnamese labels
  - AI recommendation interface with natural language input

**Design Decision**: Vite was chosen over Create React App for faster development experience with Hot Module Replacement (HMR) and optimized build performance. Tailwind CSS v3 provides rapid UI development without custom CSS. The interface is designed for Vietnamese users with Vietnamese language support throughout.

## Backend Architecture

**Framework**: Flask with Flask-CORS for cross-origin resource sharing
- **Port**: Runs on port 8080 (configurable via allowed Replit ports)
- **ORM**: SQLAlchemy for database operations (optional)
- **Environment Management**: python-dotenv for configuration
- **HTTP Client**: requests library for external API calls
- **AI Integration**: OpenAI GPT-3.5-turbo with markdown fence stripping for reliable JSON parsing

**API Endpoints**:
- `GET /api/search?query={query}` - Search movies by title
- `GET /api/discover?genre={id}&year={year}` - Discover movies with filters
- `GET /api/genres` - Get all movie genres
- `GET /api/movie/{id}` - Get movie details
- `POST /api/recommend` - AI-powered recommendations (accepts Vietnamese prompts)
- `GET /api/health` - Health check endpoint

**Design Decision**: Flask provides a lightweight, flexible framework suitable for building RESTful APIs. The backend implements robust error handling for database failures and API errors. AI recommendations use markdown fence stripping to handle both raw JSON and markdown-wrapped responses from OpenAI.

## Data Storage

**Database**: Microsoft SQL Server (MSSQL) - Optional
- **Connection**: SQLAlchemy engine with connection string from environment variables
- **Graceful Fallback**: Application works without database; DB is used for caching only
- **Schema**: Movies table with TMDB metadata including:
  - Core identifiers (id, tmdb_id)
  - Movie details (title, original_title, overview)
  - Media paths (poster_path, backdrop_path)
  - Metrics (vote_average, vote_count, popularity)
  - Metadata (release_date, genres stored as JSON text)
  - Timestamps (created_at)

**Design Decision**: MSSQL is optional with conditional initialization - the application gracefully handles database connection failures and continues to function using live TMDB API calls. The schema is designed to cache TMDB movie data locally to reduce API calls and improve performance when database is available.

## AI Recommendation System

**Implementation Details**:
- Uses OpenAI GPT-3.5-turbo model
- Processes natural language prompts in Vietnamese and English
- Maps user preferences to TMDB genre IDs
- Returns personalized movie recommendations based on TMDB popularity
- Robust JSON parsing with markdown fence stripping (handles ```json``` wrappers)
- Error handling with descriptive messages

**Example Prompts**:
- "Tôi thích phim bạo lực" (I like violent movies) → Action, Thriller, Crime genres
- "Phim hài vui vẻ" (Fun comedy movies) → Comedy genre
- "Tôi muốn xem phim kinh dị" (I want to watch horror movies) → Horror genre

# External Dependencies

## Third-Party APIs

1. **The Movie Database (TMDB) API**
   - Purpose: Fetch movie information, metadata, and media
   - Base URL: https://api.themoviedb.org/3
   - Authentication: API key stored in TMDB_API_KEY environment variable
   - Language: vi-VN (Vietnamese) for localized content

2. **OpenAI API**
   - Purpose: AI-powered movie recommendations using natural language processing
   - Model: GPT-3.5-turbo
   - Authentication: API key stored in OPENAI_API_KEY environment variable
   - Temperature: 0.7 for balanced creativity and consistency

## Database (Optional)

- **Microsoft SQL Server**: Optional data storage for caching
- Connection configured via DB_CONNECTION_STRING environment variable
- Format: `mssql+pymssql://username:password@server/database`
- Graceful degradation if database connection fails

## Environment Variables

Required secrets:
- `TMDB_API_KEY`: The Movie Database API key
- `OPENAI_API_KEY`: OpenAI API key
- `DB_CONNECTION_STRING`: SQL Server connection string (optional)

## Package Dependencies

**Backend (Python)**:
- Flask: Web framework
- Flask-CORS: Cross-origin resource sharing
- SQLAlchemy: Database ORM
- pymssql: SQL Server driver
- requests: HTTP client for API calls
- python-dotenv: Environment variable management

**Frontend (Node.js)**:
- React 19: UI library
- Axios: HTTP client
- Tailwind CSS v3: Utility-first styling
- Vite 7: Build tool and development server
- PostCSS & Autoprefixer: CSS processing

## Workflows

1. **Backend**: `python app.py` (port 8080, console output)
2. **Frontend**: `cd client && npm run dev` (port 5000, webview output)

Both workflows are configured to run automatically and restart on file changes.

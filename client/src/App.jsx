import { useState, useEffect, useRef } from 'react'
import axios from 'axios'

const API_URL = 'http://localhost:8080/api'
const TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p/w500'

function App() {
  const [searchQuery, setSearchQuery] = useState('')
  const [movies, setMovies] = useState([])
  const [genres, setGenres] = useState([])
  const [selectedGenre, setSelectedGenre] = useState('')
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('search')
  
  const [chatMessages, setChatMessages] = useState([])
  const [chatInput, setChatInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const chatEndRef = useRef(null)
  const [userInfo, setUserInfo] = useState(null);

  // Khi app load, đọc dữ liệu từ localStorage
  useEffect(() => {
    const username = localStorage.getItem("username");
    const token = localStorage.getItem("token");
    const tinh_cach = localStorage.getItem("tinh_cach");

    if (username && token && tinh_cach) {
      setUserInfo({ username, token, tinh_cach });

      // ✅ Gửi lên backend Flask
      fetch("http://localhost:8080/api/user-info", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, token, tinh_cach }),
      })
        .then((res) => res.json())
        .then((data) => {
          console.log("Server đã nhận user info:", data);
        })
        .catch((err) => console.error("Lỗi gửi user info:", err));
    }
  }, []);
  useEffect(() => {
    fetchGenres()
    fetchPopularMovies()
  }, [])

  useEffect(() => {
    if (activeTab === 'ai' && chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [chatMessages, activeTab])

  const fetchGenres = async () => {
    try {
      const response = await axios.get(`${API_URL}/genres`)
      setGenres(response.data.genres || [])
    } catch (error) {
      console.error('Error fetching genres:', error)
    }
  }

  const fetchPopularMovies = async () => {
    try {
      setLoading(true)
      const response = await axios.get(`${API_URL}/discover`)
      setMovies(response.data.results || [])
    } catch (error) {
      console.error('Error fetching movies:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSearch = async (e) => {
    e.preventDefault()
    if (!searchQuery.trim()) return

    try {
      setLoading(true)
      const response = await axios.get(`${API_URL}/search`, {
        params: { query: searchQuery }
      })
      setMovies(response.data.results || [])
    } catch (error) {
      console.error('Error searching movies:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleGenreFilter = async (genreId) => {
    setSelectedGenre(genreId)
    try {
      setLoading(true)
      const response = await axios.get(`${API_URL}/discover`, {
        params: { genre: genreId }
      })
      setMovies(response.data.results || [])
    } catch (error) {
      console.error('Error filtering by genre:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSendMessage = async (e) => {
    e.preventDefault()
    if (!chatInput.trim() || chatLoading) return

    const userMessage = chatInput.trim()
    setChatInput('')

    const newUserMessage = {
      role: 'user',
      content: userMessage
    }

    setChatMessages(prev => [...prev, newUserMessage])
    setChatLoading(true)

    try {
      const conversationHistory = chatMessages.map(msg => ({
        role: msg.role,
        content: msg.content
      }))

      const response = await axios.post(`${API_URL}/chat`, {
        message: userMessage,
        history: conversationHistory,
        user: localStorage.getItem("username"),    
        tinh_cach: localStorage.getItem("tinh_cach") 

      })

      const aiMessage = {
        role: 'assistant',
        content: response.data.message,
        movies: response.data.movies || [],
        suggest_movies: response.data.suggest_movies || false
      }

      setChatMessages(prev => [...prev, aiMessage])
    } catch (error) {
      console.error('Error sending message:', error)
      const errorMessage = {
        role: 'assistant',
        content: 'Xin lỗi, có lỗi xảy ra. Vui lòng thử lại.',
        movies: [],
        suggest_movies: false
      }
      setChatMessages(prev => [...prev, errorMessage])
    } finally {
      setChatLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold text-center mb-8 text-red-500">
          🎬 Movie Discovery App
        </h1>
              <h1 className="text-2xl">
        {userInfo
          ? `Xin chào ${userInfo.username} `
          : "Chưa đăng nhập"}
      </h1>
        <div className="mb-6 flex justify-center space-x-4">
          <button
            onClick={() => setActiveTab('search')}
            className={`px-6 py-2 rounded-lg font-semibold transition ${
              activeTab === 'search'
                ? 'bg-red-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Tìm kiếm
          </button>
          <button
            onClick={() => setActiveTab('ai')}
            className={`px-6 py-2 rounded-lg font-semibold transition ${
              activeTab === 'ai'
                ? 'bg-red-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            💬 Chat với AI
          </button>
          <button
            onClick={() => setActiveTab('login')}
            className={`px-6 py-2 rounded-lg font-semibold transition ${
              activeTab === 'ai'
                ? 'bg-red-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Login
          </button>
        </div>

        {activeTab === 'search' && (
          <div className="mb-8">
            <form onSubmit={handleSearch} className="mb-6">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Tìm kiếm phim..."
                  className="flex-1 px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-red-500"
                />
                <button
                  type="submit"
                  className="px-8 py-3 bg-red-600 hover:bg-red-700 rounded-lg font-semibold transition"
                >
                  Tìm
                </button>
              </div>
            </form>

            <div className="mb-6">
              <h3 className="text-lg font-semibold mb-3">Lọc theo thể loại:</h3>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => handleGenreFilter('')}
                  className={`px-4 py-2 rounded-full text-sm transition ${
                    selectedGenre === ''
                      ? 'bg-red-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  Tất cả
                </button>
                {genres.map((genre) => (
                  <button
                    key={genre.id}
                    onClick={() => handleGenreFilter(genre.id)}
                    className={`px-4 py-2 rounded-full text-sm transition ${
                      selectedGenre === genre.id
                        ? 'bg-red-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    {genre.name}
                  </button>
                ))}
              </div>
            </div>

            {loading ? (
              <div className="text-center py-12">
                <div className="inline-block h-12 w-12 animate-spin rounded-full border-4 border-solid border-red-500 border-r-transparent"></div>
                <p className="mt-4 text-gray-400">Đang tải...</p>
              </div>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
                {movies.map((movie) => (
                  <div
                    key={movie.id}
                    className="bg-gray-800 rounded-lg overflow-hidden hover:ring-2 hover:ring-red-500 transition cursor-pointer"
                  >
                    <img
                      src={
                        movie.poster_path
                          ? `${TMDB_IMAGE_BASE}${movie.poster_path}`
                          : 'https://via.placeholder.com/500x750?text=No+Image'
                      }
                      alt={movie.title}
                      className="w-full h-80 object-cover"
                    />
                    <div className="p-4">
                      <h3 className="font-semibold text-sm mb-2 line-clamp-2">
                        {movie.title}
                      </h3>
                      <div className="flex items-center justify-between text-xs text-gray-400">
                        <span>{movie.release_date?.split('-')[0]}</span>
                        <span className="flex items-center">
                          ⭐ {movie.vote_average?.toFixed(1)}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {movies.length === 0 && !loading && (
              <div className="text-center py-12">
                <p className="text-gray-400 text-lg">
                  Không tìm thấy phim nào. Thử tìm kiếm khác!
                </p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'ai' && (
          <div className="max-w-5xl mx-auto">
            <div className="bg-gray-800 rounded-lg shadow-xl flex flex-col" style={{ height: '70vh' }}>
              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {chatMessages.length === 0 && (
                  <div className="text-center py-12">
                    <div className="text-6xl mb-4">🤖</div>
                    <p className="text-gray-400 text-lg mb-2">Xin chào! Tôi là trợ lý phim AI</p>
                    <p className="text-gray-500">Hãy hỏi tôi về phim ảnh hoặc cho tôi biết sở thích của bạn!</p>
                  </div>
                )}

                {chatMessages.map((message, index) => (
                  <div key={index} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-3xl ${message.role === 'user' ? 'bg-red-600' : 'bg-gray-700'} rounded-lg p-4`}>
                      <p className="whitespace-pre-wrap">{message.content}</p>
                      
                      {message.suggest_movies && message.movies && message.movies.length > 0 && (
                        <div className="mt-4">
                          <p className="text-sm text-gray-300 mb-3">Gợi ý phim cho bạn:</p>
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                            {message.movies.map((movie) => (
                              <div
                                key={movie.id}
                                className="bg-gray-800 rounded-lg overflow-hidden hover:ring-2 hover:ring-red-500 transition"
                              >
                                <img
                                  src={
                                    movie.poster_path
                                      ? `${TMDB_IMAGE_BASE}${movie.poster_path}`
                                      : 'https://via.placeholder.com/300x450?text=No+Image'
                                  }
                                  alt={movie.title}
                                  className="w-full h-40 object-cover"
                                />
                                <div className="p-2">
                                  <h4 className="font-semibold text-xs mb-1 line-clamp-1">
                                    {movie.title}
                                  </h4>
                                  <div className="flex items-center justify-between text-xs text-gray-400">
                                    <span>{movie.release_date?.split('-')[0]}</span>
                                    <span>⭐ {movie.vote_average?.toFixed(1)}</span>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {chatLoading && (
                  <div className="flex justify-start">
                    <div className="bg-gray-700 rounded-lg p-4">
                      <div className="flex space-x-2">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                      </div>
                    </div>
                  </div>
                )}

                <div ref={chatEndRef} />
              </div>

              <form onSubmit={handleSendMessage} className="p-4 border-t border-gray-700">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Nhắn tin với AI... (VD: Tôi muốn xem phim kinh dị)"
                    className="flex-1 px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-red-500"
                    disabled={chatLoading}
                  />
                  <button
                    type="submit"
                    disabled={chatLoading || !chatInput.trim()}
                    className="px-6 py-3 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-semibold transition"
                  >
                    Gửi
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
{activeTab === 'login' && (
  <div className="min-h-screen flex items-center justify-center bg-gray-900">
    <div className="bg-gray-800 p-8 rounded-2xl shadow-lg w-full max-w-md">
      <h2 className="text-2xl font-bold text-center text-white mb-6">
        🔐 Đăng nhập
      </h2>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          const username = e.target.username.value.trim();
          const password = e.target.password.value.trim();

          // ✅ Tài khoản fix cứng
          const FIXED_USER = "admin";
          const FIXED_PASS = "123456";

          if (username === FIXED_USER && password === FIXED_PASS) {
            // ✅ Lưu vào localStorage
            localStorage.setItem("username", username);
            localStorage.setItem("token", "fake-token-123");
            localStorage.setItem("tinh_cach", "bạo lực");

            alert("Đăng nhập thành công!");
            window.location.reload(); // reload lại app nếu muốn
          } else {
            alert("Sai tên đăng nhập hoặc mật khẩu!");
          }
        }}
        className="space-y-4"
      >
        <div>
          <label className="block text-gray-300 mb-2">Tên đăng nhập</label>
          <input
            type="text"
            name="username"
            className="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-red-500"
            required
          />
        </div>

        <div>
          <label className="block text-gray-300 mb-2">Mật khẩu</label>
          <input
            type="password"
            name="password"
            className="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-red-500"
            required
          />
        </div>

        <button
          type="submit"
          className="w-full py-3 bg-red-600 hover:bg-red-700 rounded-lg font-semibold text-white transition"
        >
          Đăng nhập
        </button>
      </form>
    </div>
  </div>
)}


      </div>
    </div>
  )
}

export default App

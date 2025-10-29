import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# =========================
# 🎥 PAGE CONFIGURATION
# =========================
st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

st.markdown("""
    <style>
        body {
            background-color: #0b0c10;
            color: #f5f5f5;
        }
        .banner {
            position: relative;
            width: 100%;
            height: 500px;
            background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.9)),
                        url('https://images8.alphacoders.com/116/1164508.jpg');
            background-size: cover;
            background-position: center;
            border-radius: 20px;
            margin-bottom: 40px;
        }
        .banner h1 {
            position: absolute;
            bottom: 60px;
            left: 50px;
            font-size: 60px;
            color: white;
            font-weight: 900;
        }
        .banner p {
            position: absolute;
            bottom: 30px;
            left: 50px;
            font-size: 20px;
            color: #ccc;
        }
        .movie-card {
            transition: 0.3s;
            cursor: pointer;
            text-align: center;
        }
        .movie-card:hover {
            transform: scale(1.05);
        }
        .movie-title {
            font-size: 16px;
            color: #fff;
            font-weight: 600;
        }
        .genre {
            color: #aaa;
            font-size: 14px;
        }
        .section-title {
            color: #fff;
            font-size: 26px;
            font-weight: 700;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# 📊 LOAD DATA
# =========================
@st.cache_data
def load_data():
    # Example dataset (replace with MovieLens or TMDb dataset)
    data = pd.DataFrame({
        'movie_id': [1, 2, 3, 4, 5, 6, 7],
        'title': ['Inception', 'Interstellar', 'The Dark Knight', 'Avengers', 'Titanic', 'Avatar', 'Joker'],
        'genres': ['Action Sci-Fi', 'Adventure Sci-Fi', 'Action Thriller', 'Action Superhero', 'Romance Drama', 'Fantasy Sci-Fi', 'Drama Crime'],
        'keywords': ['dream heist', 'space travel', 'gotham hero', 'marvel heroes', 'love ship', 'alien world', 'mental illness'],
        'poster': [
            'https://m.media-amazon.com/images/I/91p3Z4jFJ2L._AC_SL1500_.jpg',
            'https://m.media-amazon.com/images/I/71n9mXELWzL._AC_SL1500_.jpg',
            'https://m.media-amazon.com/images/I/71P8J4aIutL._AC_SL1500_.jpg',
            'https://m.media-amazon.com/images/I/81ai6zx6eXL._AC_SL1500_.jpg',
            'https://m.media-amazon.com/images/I/71rNJQ2g-LL._AC_SY679_.jpg',
            'https://m.media-amazon.com/images/I/61OUGpUfAyL._AC_SY679_.jpg',
            'https://m.media-amazon.com/images/I/71niXI3lxlL._AC_SY679_.jpg'
        ]
    })
    return data

movies = load_data()

# =========================
# 🧠 HYBRID MODEL (Content + Collaborative)
# =========================

@st.cache_resource
def build_models():
    # --- Content-based filtering ---
    movies['tags'] = movies['genres'] + " " + movies['keywords']
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    content_similarity = cosine_similarity(vectors)

    # --- Collaborative filtering (simulated small user-movie matrix) ---
    np.random.seed(42)
    user_ratings = pd.DataFrame(np.random.randint(1, 6, size=(5, len(movies))),
                                columns=movies['title'])
    collab_similarity = cosine_similarity(user_ratings.T)

    # --- Combine both ---
    hybrid_similarity = (content_similarity + collab_similarity) / 2
    return hybrid_similarity

similarity = build_models()

def recommend(movie_title):
    if movie_title not in movies['title'].values:
        return []
    idx = movies[movies['title'] == movie_title].index[0]
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]] for i in movie_list]

# =========================
# 🎬 FRONTEND UI
# =========================

# Banner Section
st.markdown("""
<div class='banner'>
  <h1>CineMatch</h1>
  <p>Discover your next favorite movie with AI-powered recommendations 🎥</p>
</div>
""", unsafe_allow_html=True)

# Filter Buttons
genre_filter = st.multiselect("🎭 Filter by Genre", 
                              options=sorted(set(sum([g.split() for g in movies['genres']], []))),
                              default=[])

if genre_filter:
    filtered = movies[movies['genres'].apply(lambda x: any(g in x for g in genre_filter))]
else:
    filtered = movies

# Display Movies
st.markdown("<div class='section-title'>🔥 Trending Now</div>", unsafe_allow_html=True)
cols = st.columns(5)
for i, col in enumerate(cols):
    if i < len(filtered):
        with col:
            movie = filtered.iloc[i]
            st.image(movie['poster'], use_column_width=True)
            st.markdown(f"<div class='movie-title'>{movie['title']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='genre'>{movie['genres']}</div>", unsafe_allow_html=True)
            if st.button(f"See Details - {movie['title']}", key=movie['movie_id']):
                st.session_state['selected_movie'] = movie['title']

# =========================
# 📖 MOVIE DETAIL PAGE
# =========================
if 'selected_movie' in st.session_state:
    st.markdown("---")
    movie_name = st.session_state['selected_movie']
    movie = movies[movies['title'] == movie_name].iloc[0]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(movie['poster'], width=300)
    with col2:
        st.markdown(f"## {movie['title']}")
        st.write(f"🎭 Genres: {movie['genres']}")
        st.write("⭐ Average Rating: 8.4/10")
        st.write("🕒 Duration: 120 min")
        st.write("🗝 Keywords:", movie['keywords'])
        st.write("💬 Overview: A mind-bending journey through imagination and destiny.")

    st.markdown("### 🎞 Similar Movies You May Like:")
    for rec in recommend(movie_name):
        c1, c2 = st.columns([1, 4])
        with c1:
            st.image(rec['poster'], width=100)
        with c2:
            st.markdown(f"{rec['title']}")
            st.caption(rec['genres'])
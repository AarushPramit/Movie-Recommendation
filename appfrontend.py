import streamlit as st
import pandas as pd
import pickle
import requests

# --- PAGE CONFIG ---
st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

# --- STYLING ---
st.markdown("""
    <style>
    body {
        background-color: #0b0c10;
        color: #f5f5f5;
    }
    .movie-card {
        transition: transform 0.2s;
        cursor: pointer;
    }
    .movie-card:hover {
        transform: scale(1.05);
    }
    .title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 10px;
        color: #fff;
    }
    .sub {
        color: #aaa;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD DATA & MODEL ---
movies = pd.read_csv(r"D:\movie recommendation\movies.csv")  # columns: ['movie_id','title','genres','poster']
similarity = pickle.load(open(r"D:\movie recommendation\similarity.pkl", "rb"))
# --- FUNCTIONS ---
def recommend(movie_name):
    if movie_name not in movies['title'].values:
        return []
    idx = movies[movies['title'] == movie_name].index[0]
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]] for i in movie_list]

def movie_details(movie):
    st.markdown(f"### 🎞 {movie['title']}")
    st.image(movie['poster'], use_column_width=True)
    st.write(f"*Genre:* {movie['genres']}")
    st.write("⭐ *Rating:* 8.3/10 (sample)")
    st.write("🕒 *Duration:* 120 min")
    st.markdown("---")
    st.write("*Overview:* A thrilling story of courage and redemption.")
    st.markdown("### Similar Movies You May Like:")

    for rec in recommend(movie['title']):
        col1, col2 = st.columns([1,3])
        with col1:
            st.image(rec['poster'], width=120)
        with col2:
            st.subheader(rec['title'])
            st.write(rec['genres'])
        st.markdown("---")

# --- HOME PAGE ---
st.markdown("<h1 style='text-align:center;'>🍿 CineMatch - Movie Recommender</h1>", unsafe_allow_html=True)
st.write("Find your next favorite movie instantly using AI-powered recommendations.")

search = st.text_input("🔍 Search for a movie")

if search:
    if search in movies['title'].values:
        movie = movies[movies['title'] == search].iloc[0]
        movie_details(movie)
    else:
        st.warning("Movie not found.")
else:
    st.subheader("🔥 Trending Now")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            movie = movies.iloc[i]
            if st.button(movie['title'], key=movie['movie_id']):
                movie_details(movie)
            st.image(movie['poster'], use_column_width=True)
            st.caption(movie['genres'])
🎬 CineMatch – Movie Recommendation App
📖 Overview

CineMatch is an AI-powered movie recommendation web app built using Streamlit.
It provides personalized movie recommendations by combining content-based and collaborative filtering approaches into a hybrid recommendation system.

The app allows users to:

🎞 Browse trending movies

🔍 Filter by genres

💡 Get AI-based similar movie suggestions

📈 Explore how hybrid ML models work

🧰 Tech Stack

Python 3.10+

Streamlit – for the web interface

Pandas / NumPy – for data manipulation

Scikit-learn – for similarity and vectorization

HTML + CSS – for design and UI styling

🗂️ Project Structure
movie-recommendation/
│
├── app.py                  # Main Streamlit app
├── README.md               # Documentation (this file)
├── requirements.txt        # Python dependencies
├── movies.csv              # (optional) Movie dataset
└── similarity.pkl          # (optional) Pre-trained similarity model

⚙️ Installation & Setup
1️⃣ Clone or copy the project

Place all project files inside one folder, for example:

D:\movie-recommendation

2️⃣ Open PowerShell or Anaconda Prompt
D:
cd "movie-recommendation"

3️⃣ Install dependencies

If you’re using pip:

pip install -r requirements.txt


If you prefer conda:

conda install -c conda-forge streamlit pandas numpy scikit-learn

▶️ Running the App

To launch CineMatch, run this command inside the folder:

streamlit run app.py


If Streamlit isn’t recognized, use:

python -m streamlit run app.py


Then open the local URL displayed (e.g., http://localhost:8501) in your browser.

🧠 How It Works

Dataset Loading
Loads a small built-in dataset of movie titles, genres, keywords, and posters.

Hybrid Model

Content-based filtering: Uses CountVectorizer and cosine_similarity to compare movie text features.

Collaborative filtering: Simulates user ratings and measures similarity.

Combines both similarities to generate hybrid recommendations.

User Interface

Browse trending movies.

Filter by genre.

Click a movie to view details, overview, and similar movie suggestions.

🧩 Example Code Snippet
movies['tags'] = movies['genres'] + " " + movies['keywords']
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
content_similarity = cosine_similarity(vectors)

user_ratings = pd.DataFrame(np.random.randint(1, 6, size=(5, len(movies))),
                            columns=movies['title'])
collab_similarity = cosine_similarity(user_ratings.T)

hybrid_similarity = (content_similarity + collab_similarity) / 2

🌐 App Features

🎥 Trending Section: Displays top movies dynamically

🧠 AI Recommender: Suggests similar movies instantly

🎭 Genre Filter: Narrow down movies by genre

🖼 Posters & Details: Interactive and visually engaging

🕶 Dark Mode UI: Clean modern design using custom CSS

📦 Optional Add-ons

You can replace the dummy dataset with real movie data:

movies.csv → dataset with columns: movie_id, title, genres, poster

similarity.pkl → precomputed similarity matrix for faster loading

Update the file paths in app.py accordingly.

👨‍💻 Author

Aarush Pramit
Developed as an academic project demonstrating Movie Recommendation Systems using Streamlit and Machine Learning.

🧾 License

This project is open for educational and personal use. Attribution is appreciated if reused or modified.

📦 requirements.txt

Copy this into a file named requirements.txt in the same folder:

streamlit==1.39.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.2# Movie-Recommendation

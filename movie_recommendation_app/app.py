from flask import Flask, render_template, request
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

app = Flask(__name__)

# 🔹 Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔹 CSV file paths
MOVIE_CSV = os.path.join(BASE_DIR, 'movies.csv')    # updated
RATINGS_CSV = os.path.join(BASE_DIR, 'ratings.csv') # updated

# 🔹 Check if files exist
if not os.path.exists(MOVIE_CSV) or not os.path.exists(RATINGS_CSV):
    raise FileNotFoundError(
        f"❌ Missing CSV files! Make sure both 'movies.csv' and 'ratings.csv' exist in {BASE_DIR}"
    )

# 🔹 Load CSV files
movies = pd.read_csv(MOVIE_CSV)
ratings = pd.read_csv(RATINGS_CSV)

# 🔹 Merge datasets
movie_data = pd.merge(ratings, movies, on='movieId')

# 🔹 Create user-movie ratings matrix
ratings_matrix = movie_data.pivot_table(index='userId', columns='title', values='rating')
ratings_matrix.fillna(0, inplace=True)

# 🔹 Compute cosine similarity between movies
movie_similarity = cosine_similarity(ratings_matrix.T)
movie_similarity_df = pd.DataFrame(movie_similarity, index=ratings_matrix.columns, columns=ratings_matrix.columns)

# 🔹 Recommendation function
def recommend_movies(movie_name, num=5):
    movie_list = movie_similarity_df.columns.tolist()
    close_matches = get_close_matches(movie_name, movie_list, n=1, cutoff=0.3)

    if not close_matches:
        return [f"❌ '{movie_name}' not found in database. Try another movie."]

    best_match = close_matches[0]
    similar_scores = movie_similarity_df[best_match].sort_values(ascending=False)
    similar_movies = list(similar_scores.iloc[1:num + 1].index)

    return [f"✅ Showing results for: {best_match}"] + similar_movies

# 🔹 Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie_name']
    recommendations = recommend_movies(movie_name)
    return render_template('recommend.html', movie_title=movie_name, recommendations=recommendations)

# 🔹 Run app
if __name__ == '__main__':
    print("🚀 Flask app running... Visit http://127.0.0.1:5000")
    app.run(debug=True)

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load data
ratings = pd.read_csv('data/train.csv')
movies = pd.read_csv('data/movies.csv')

# Create pivot table (userId vs movieId)
ratings_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Compute cosine similarity between movies
movie_similarity = cosine_similarity(ratings_matrix.T)
movie_similarity_df = pd.DataFrame(movie_similarity, index=ratings_matrix.columns, columns=ratings_matrix.columns)

# Save both dataframes
joblib.dump(ratings_matrix, 'ratings_matrix.pkl')
joblib.dump(movie_similarity_df, 'movie_similarity.pkl')
print("✅ Model files saved: ratings_matrix.pkl, movie_similarity.pkl")

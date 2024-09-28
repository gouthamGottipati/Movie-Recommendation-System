import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Loading the dataset
def load_data():
    column_names = ['userID', 'movieID', 'rating', 'timestamp']
    df = pd.read_csv('data/u.data', sep='\t', names=column_names)
    df.drop('timestamp', axis=1, inplace=True)
    return df

# Item Matrix
def create_user_item_matrix(df):
    user_item_matrix = df.pivot_table(index='userID', columns='movieID', values='rating')
    user_item_matrix_filled = user_item_matrix.fillna(0)
    return user_item_matrix, user_item_matrix_filled

# Cosine Similarity Functionality
def compute_user_similarity(user_item_matrix_filled):
    user_similarity = cosine_similarity(user_item_matrix_filled)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix_filled.index, columns=user_item_matrix_filled.index)
    return user_similarity_df

# Prediction of rating for a specific user and movie 
def predict_rating(user_id, movie_id, user_item_matrix, user_similarity_df):
    user_ratings = user_item_matrix.loc[user_id]
    similar_users = user_similarity_df[user_id]
    users_who_rated = user_item_matrix[movie_id].dropna().index
    similar_users = similar_users[users_who_rated]

    weighted_ratings = 0
    similarity_sum = 0
    for other_user in similar_users.index:
        weighted_ratings += similar_users[other_user] * user_item_matrix.loc[other_user, movie_id]
        similarity_sum += similar_users[other_user]

    if similarity_sum != 0:
        predicted_rating = weighted_ratings / similarity_sum
    else:
        predicted_rating = user_ratings.mean()
    
    return predicted_rating

# Recommending top N movies for a user
def recommend_movies(user_id, user_item_matrix, user_similarity_df, N=5):
    user_ratings = user_item_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings.isna()].index
    predicted_ratings = {movie_id: predict_rating(user_id, movie_id, user_item_matrix, user_similarity_df) for movie_id in unrated_movies}
    top_movies = sorted(predicted_ratings, key=predicted_ratings.get, reverse=True)[:N]
    return top_movies

if __name__ == "__main__":
    # Loading data
    df = load_data()
    user_item_matrix, user_item_matrix_filled = create_user_item_matrix(df)
    
    # Computing user similarity
    user_similarity_df = compute_user_similarity(user_item_matrix_filled)
    
    # Recommending top 5 movies for user 1
    recommendations = recommend_movies(1, user_item_matrix, user_similarity_df, N=5)
    print(f"Top 5 recommended movies for user 1: {recommendations}")

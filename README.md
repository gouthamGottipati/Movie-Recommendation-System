# Simple Movie Recommendation System

## Objective
This project creates a simple movie recommendation system using collaborative filtering based on user ratings. The system predicts how much a user might like a movie they haven't seen by finding similar users.

## How It Works
- **User-Based Collaborative Filtering**: This system calculates similarities between users using cosine similarity based on their movie ratings.
- **Movie Recommendations**: The system predicts ratings for movies a user hasn't rated and recommends the top movies.

## Dataset
The project uses the **MovieLens 100k dataset**. Source: https://grouplens.org/datasets/movielens/100k/

Unzip the dataset and place the `u.data` file in the `data/` folder.

## Setup

### Install dependencies:
```bash
pip install pandas numpy scikit-learn

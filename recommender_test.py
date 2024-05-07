import pandas as pd

from recommender.kNNRecommender import kNNRecommender

if __name__ == "__main__":
    ratings = pd.read_csv('https://s3-us-west-2.amazonaws.com/recommender-tutorial/ratings.csv')
    movies = pd.read_csv('https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv')

    print(ratings.head(10))
    recommender = kNNRecommender(10)
    recommender.train(ratings, users="userId", items="movieId", ratings="rating")
    movieId = 1
    movie_titles = dict(zip(movies['movieId'], movies['title']))
    nearest_neighbours = recommender.recommend(movieId)
    print("Because you watched " + movies[movies['movieId'] == movieId]["title"])
    for i in nearest_neighbours:
        print(movie_titles[i])
    print(nearest_neighbours)
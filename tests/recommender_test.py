from pathlib import Path
import pandas as pd
import sys

root_dir = Path(".").resolve()
sys.path.append(str(root_dir.absolute()))

from recommender.kNNRecommender import kNNRecommender

def test_knn():
    ratings = pd.read_csv(str(root_dir / "ml-latest-small" / "ml-latest-small" / "ratings.csv"))
    movies = pd.read_csv(str(root_dir / "ml-latest-small" / "ml-latest-small" / 'movies.csv'))

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
    assert nearest_neighbours == [3114, 480, 780, 260, 356, 364, 1210, 648, 1265, 1270]


if __name__ == "__main__":
    test_knn()
import pandas as pd

from ifilike import MovieLens, Recommender

def main():
    #movielens = MovieLens('ml-latest-small')
    movielens = MovieLens('ml-latest')
    rec_engine = Recommender(movielens.get_matrix(), 10, 10)

    test_user_df = pd.read_csv("example.user", sep='|')
    test_user_vec = movielens.df_to_vector(test_user_df)
    for movie_id in rec_engine.get_recommendations(test_user_vec):
        print(movielens.movie_id_to_title(movie_id))

if __name__ == '__main__':
    main()

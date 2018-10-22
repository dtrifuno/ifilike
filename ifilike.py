import copy
import random
from heapq import nlargest

import numpy as np
import pandas as pd

from scipy.sparse.linalg import norm as sp_norm
from scipy.linalg import norm
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
from scipy.sparse.linalg import svds

random.seed(42)

MEAN_SCORE = 2.75    # (MaxScore - MinScore) / 2, for normalization
TRAIN_RATIO = 0.8    # what percentage of users will be used for training


def get_n_largest(n, sparse_matrix):
    m = coo_matrix(sparse_matrix)
    if m.shape[0] == 1:
        idx = m.col
    elif m.shape[1] == 1:
        idx = m.row
    else:
        raise ValueError("Expected row or column vector, not " + m.shape)
    return [i for v, i in nlargest(n, (x for x in zip(m.data, idx)))]


class MovieLens:
    def __init__(self,
                 movielens_dir,
                 movies_file="movies.csv",
                 ratings_file="ratings.csv"):
        # Load movies file
        self._movies_df = pd.read_csv(movielens_dir + '/' + movies_file)
        self._movies_df.drop("genres", axis=1, inplace=True)

        # Load ratings file
        self._ratings_df = pd.read_csv(movielens_dir + '/' + ratings_file)
        self._ratings_df.drop("timestamp", axis=1, inplace=True)
        self._ratings_df["rating"] = self._ratings_df["rating"] - MEAN_SCORE

        self.rekey_movies()
        self.n_movies = max(self._movies_df["movieId"]) + 1
        self._ratings_matrix = None
        self._train_matrix = None
        self._test_matrix = None


    def rekey_movies(self):
        # Drop movies without ratings
        rated_ids = set(self._ratings_df["movieId"])
        self._movies_df = self._movies_df[
            self._movies_df["movieId"].isin(rated_ids)]

        # Compute new IDs for the reduced data set
        reduced_ids = {k:v for (v, k) in enumerate(self._movies_df["movieId"])}
        self._movies_df["movieId"] = self._movies_df["movieId"].apply(
            reduced_ids.get)
        self._ratings_df["movieId"] = self._ratings_df["movieId"].apply(
            reduced_ids.get)

    @staticmethod
    def ratings_to_matrix(ratings_df, columns):
        return coo_matrix((ratings_df["rating"],
                           (ratings_df["userId"], ratings_df["movieId"])),
                          shape=(max(ratings_df["userId"])+1, columns))

    def title_to_movie_id(self, title):
        return next(i for i, v in enumerate(self._movies_df['title'] == title)
                    if v)

    def movie_id_to_title(self, movie_id):
        return self._movies_df.iloc[movie_id, ]['title']

    def df_to_vector(self, df):
        vector = lil_matrix((1, self.n_movies))
        for _, (title, rating) in df.iterrows():
            vector[0, self.title_to_movie_id(title)] = rating - MEAN_SCORE
        return vector.tocsr()

    def get_matrix(self):
        return self.ratings_to_matrix(self._ratings_df, self.n_movies).tocsr()


    def get_train_test_split(self, train_ratio):
        user_ids = list(set(self._ratings_df["userId"]))
        random.shuffle(user_ids)
        number_of_train_ids = round(len(user_ids) * TRAIN_RATIO)
        train_ids = set(user_ids[:number_of_train_ids])

        train_df = self._ratings_df[self._ratings_df["userId"].isin(train_ids)]
        #reduced_train_ids = {k:v for (v, k) in enumerate(train_df["userId"])}
        #train_df["userId"] = train_df["userId"].apply(reduced_train_ids.get)
        test_df = self._ratings_df[~self._ratings_df["userId"].isin(train_ids)]
        #reduced_test_ids = {k:v for (v, k) in enumerate(test_df["userId"])}
        #test_df["userId"] = test_df["userId"].apply(reduced_test_ids.get)

        train_matrix = self.ratings_to_matrix(train_df, self.n_movies).tocsr()
        test_matrix = self.ratings_to_matrix(test_df, self.n_movies).tocsr()
        return (train_matrix, test_matrix)



class Recommender:
    def __init__(self, train_matrix, dimension, neighbors=10):
        u, s, vt = svds(train_matrix, k=dimension)
        self._neighbors = neighbors
        self._vt = vt
        self._original_matrix = train_matrix
        self._reduced_matrix = u.dot(np.diag(s))
        # normalize the rows in the reduced matrix
        self._reduced_matrix = self._reduced_matrix / \
                np.linalg.norm(self._reduced_matrix, axis=-1)[:, np.newaxis]

    def get_estimated_vector(self, user_vector):
        # project and normalize user vector
        reduced_vector = self._vt * user_vector.T
        reduced_vector = reduced_vector / norm(reduced_vector)
        cos_sims = self._reduced_matrix.dot(reduced_vector)

        k = self._neighbors
        neighbors = get_n_largest(k, cos_sims)
        estimated_vector = csr_matrix((1, self._original_matrix.shape[1]))
        for neighbor in neighbors:
            estimated_vector += self._original_matrix.getrow(neighbor)
        estimated_vector /= len(neighbors)
        return estimated_vector

    def get_recommendations(self, user_vector, n=10):
        estimated_vec = self.get_estimated_vector(user_vector)
        already_rated = user_vector.nonzero()[1]
        top_scorers = get_n_largest(n + len(already_rated), estimated_vec)
        return [id_ for id_ in top_scorers if id_ not in already_rated][:n]


class GridSearch:
    def __init__(self, train_matrix, max_components, max_neighbors):
        self._train_matrix = train_matrix
        self._max_components = max_components
        self._max_neighbors = max_neighbors

    def _drop_entries(self, vec, ratio):
        """ return a copy of the vector with ratio of nonzero entries zeroed """
        nonzero_idxs = vec.nonzero()[1]
        random.shuffle(nonzero_idxs)
        n_drop = round(ratio * len(nonzero_idxs))
        drop_idxs = nonzero_idxs[:n_drop]
        dropped_vec = lil_matrix(vec)
        for idx in drop_idxs:
            dropped_vec[:, idx] = 0
        return dropped_vec.tocsr()

    def _get_error(self, rec, test_matrix, norm_ord=2):
        total_error = 0
        test_users = test_matrix.shape[0]
        for i in range(test_users):
            vec = test_matrix.getrow(i)
            dropped_vec = self._drop_entries(vec, 0.25)
            estimated_vec = rec.get_estimated_vector(dropped_vec)
            total_error += sp_norm(vec - estimated_vec, ord=norm_ord, axis=0)[0]
        return total_error

    def find_params(self, test_matrix):
        errors = pd.DataFrame(index=range(1, self._max_neighbors+1),
                              columns=range(1, self._max_components+1))
        for dim in range(1, self._max_components+1):
            rec_engine = Recommender(self._train_matrix, dim, 1)
            for k in range(1, self._max_neighbors+1):
                print((dim, k))
                rec_engine._neighbors = k
                errors.loc[dim, k] = self._get_error(rec_engine, test_matrix)
        print(errors)


def main():
    movielens = MovieLens("ml-latest-small")
    test_matrix, train_matrix = movielens.get_train_test_split(0.75)
    gridsearcher = GridSearch(train_matrix, 2, 2)
    gridsearcher.find_params(test_matrix)


if __name__ == '__main__':
    main()

import random
import numpy as np
import pandas as pd

from scipy.linalg import norm
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

random.seed(42)

MEAN_SCORE = 2.75    # (MaxScore - MinScore) / 2, for normalization
TRAIN_RATIO = 0.8    # what percentage of users will be used for training
MOVIELENS_DIR = 'ml-latest-small/'

movies_df = pd.read_csv(MOVIELENS_DIR + 'processed_movies.csv')
movies_df.drop("genres", axis=1, inplace=True)
n_movies = max(movies_df['movieId'])
movies_df['movieId'] = movies_df['movieId'] - 1

ratings_df = pd.read_csv(MOVIELENS_DIR + 'processed_ratings.csv')
ratings_df.drop("timestamp", axis=1, inplace=True)
ratings_df['movieId'] = ratings_df['movieId'] - 1
ratings_df['rating'] = ratings_df['rating'] - MEAN_SCORE


# Function
user_ids = list(set(ratings_df["userId"]))
random.shuffle(user_ids)
number_of_train_ids = round(len(user_ids) * TRAIN_RATIO)
train_ids = set(user_ids[:number_of_train_ids])

train_df = ratings_df[ratings_df["userId"].isin(train_ids)]
test_df = ratings_df[~ratings_df["userId"].isin(train_ids)]


def rekey_df(df, colname):
    old_ids = set(df[colname])
    old_to_new_id = {k:v for (v, k) in enumerate(old_ids)}
    df[colname] = df[colname].apply(old_to_new_id.get)

rekey_df(train_df, 'userId')
rekey_df(test_df, 'userId')

def ratings_to_matrix(ratings_df, columns):
    return coo_matrix((ratings_df['rating'],
                       (ratings_df['userId'], ratings_df['movieId'])),
                      shape=(max(ratings_df['userId'])+1, columns))

train_matrix = ratings_to_matrix(train_df, n_movies).tocsr()
test_matrix = ratings_to_matrix(test_df, n_movies).tocsr()


class Recommender:
    def __init__(self, train_matrix, dimension, neighbors=10):
        u, s, vt = svds(train_matrix, k=dimension)
        self._neighbors = neighbors
        self._vt = vt
        self._original_matrix = train_matrix
        self._reduced_matrix = u.dot(np.diag(s))
        # normalize the rows in the reduced matrix
        for i, _ in enumerate(self._reduced_matrix):
            norm_i = norm(self._reduced_matrix[i])
            self._reduced_matrix[i] = self._reduced_matrix[i] / norm_i

    def get_estimated_vector(self, user_vector):
        # project and normalize user vector
        reduced_vector = (self._vt).dot(user_vector)
        reduced_vector = reduced_vector / norm(reduced_vector)
        cos_sims = self._reduced_matrix.dot(reduced_vector.T)

        neighbors = np.argpartition(cos_sims, -self._neighbors)[-(self._neighbors):]
        estimated_vector = np.zeros(self._original_matrix.shape[1])
        for neighbor in neighbors:
            estimated_vector += self._original_matrix[neighbor].toarray()[0]
        estimated_vector /= len(neighbors)
        return estimated_vector

    def get_recommendations(self, user_vector, n=10):
        estimated_vec = self.get_estimated_vector(user_vector)
        estimated_vec[user_vector.nonzero()[0]] = -2.5
        top_scores = np.argpartition(estimated_vec, -n)[-n:]
        return top_scores

def relative_error(actual, approximate):
    return abs((actual - approximate) / actual)


class GridSearch:
    def __init__(self, train_matrix, max_components, max_neighbors):
        self._rec_engine = train_matrix
        self._max_components = max_components
        self._max_neighbors = max_neighbors

    def _drop_entries(self, vec, ratio):
        nonzero_idxs = vec.nonzero()[0]
        random.shuffle(nonzero_idxs)
        n_drop = round(len(nonzero_idxs) * (1-ratio))
        drop_idxs = nonzero_idxs[:n_drop]
        vec[drop_idxs] = 0

    def _get_error(self, rec, test_matrix, norm_ord=2):
        total_error = 0
        for vec in test_matrix.toarray():
            dropped_vec = vec
            self._drop_entries(dropped_vec, 0.75)
            estimated_vec = rec.get_estimated_vector(dropped_vec)
            total_error += norm(vec - estimated_vec, norm_ord)
        return total_error

    def find_params(self, test_matrix, tol):
        errors = pd.DataFrame(index=range(1, self._max_neighbors+1),
                              columns=range(1, self._max_components+1))
        for dim in range(1, self._max_components+1):
            rec_engine = Recommender(train_matrix, dim, 1)
            for k in range(1, self._max_neighbors+1):
                rec_engine._neighbors = k
                errors.loc[dim, k] = self._get_error(rec_engine, test_matrix)
                if dim > 1 and k > 1 \
                    and relative_error(errors.loc[dim, k], errors.loc[dim-1, k-1]) < tol:
                        return (dim, k)
        return (dim, k)

gs = GridSearch(train_matrix, 10, 10)
print(gs.find_params(test_matrix, 0.002))

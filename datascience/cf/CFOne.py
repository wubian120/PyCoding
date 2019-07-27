"""
协同过滤实例1
https://www.jianshu.com/p/7fe564cf0d7c

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD

print("hello cf")

links = pd.read_csv("/Users/richard.wu/bigdata/movie_data/ml-latest-small/links.csv")
links.head()

movies = pd.read_csv("/Users/richard.wu/bigdata/movie_data/ml-latest-small/movies.csv")
movies.head()

ratings = pd.read_csv("/Users/richard.wu/bigdata/movie_data/ml-latest-small/ratings.csv")
ratings.head()

movies_add_links = pd.merge(movies, links, on='movieId')


def get_year(x):
    try:
        y = int(x.strip()[-5:-1])
    except:
        y = 0

    return y


movies_add_links['movie_year'] = movies_add_links['title'].apply(get_year)

ratings_counts = pd.DataFrame(ratings.groupby('movieId').count()['rating'])
ratings_counts.rename(columns={'rating': 'ratingCount'}, inplace=True)

movie_add_rating = pd.merge(movies_add_links, ratings_counts, on='movieId')

ratings_means = pd.DataFrame(ratings.groupby('movieId').mean()['rating'])
ratings_means.columns = ['rating_mean']
movie_total = pd.merge(movie_add_rating, ratings_means, on='movieId')
movie_total.head()

# ratings.rating.value_counts(sort=True).plot('bar')
# plt.title('Rating Distribution\n')
# plt.xlabel('Rating')
# plt.ylabel('Count')
# plt.savefig('rating.png', bbox_inches='tight')
# plt.show()

# 设置画幅
plt.figure(figsize=(16, 16))

plt.subplot(2, 1, 1)

movie_total.groupby('movie_year')['ratingCount'].count().plot('bar')
plt.title('Movies counts by years\n')
plt.xlabel('years')
plt.ylabel('counts')

plt.subplot(2, 1, 2)
movie_total.groupby('movie_year')['ratingCount'].sum().plot('bar')
plt.title('Movies rating by years\n')
plt.xlabel('years')
plt.ylabel('ratings')
plt.savefig('mix.png', bbox_inches='tight')
# plt.show()

combine_movie = pd.merge(ratings, ratings_counts, on='movieId')
combine_movie = combine_movie.dropna()
combine_movie.head()

combine_movie.ratingCount.quantile(np.arange(.5, 1, .05))

popularity_threshold = 69
rating_popular_movie = combine_movie.query('ratingCount >= @popularity_threshold')
rating_popular_movie.head()

movie_pivot = rating_popular_movie.pivot(index='movieId', columns='userId', values='rating').fillna(0)
movie_pivot.shape

movie_1 = ratings.groupby('movieId').count().sort_values('rating', ascending=False).index[0]
m1 = movie_total[movie_total.movieId == movie_1]


def get_movie(df_movie, movie_list):
    df_movie_id = pd.DataFrame(movie_list, index=np.arange(len(movie_list)), columns=['movieId'])
    corr_movies = pd.merge(df_movie_id, movie_total, on='movieId')
    return corr_movies


def person_method(df_movie, pivot, movie, num):
    bones_ratings = pivot[movie]
    similar_to_bones = pivot.corrwith(bones_ratings)
    corr_bones = pd.DataFrame(similar_to_bones, columns=['pearson'])

    corr_bones.dropna(inplace=True)
    corr_summary = corr_bones.join(df_movie[['movieId', 'ratingCount']].set_index('movieId'))

    movie_list = corr_summary[corr_summary['ratingCount'] >= 100].sort_values('pearson', ascending=False).index[
                 :num].tolist()
    return movie_list


movie_list = person_method(movie_total, movie_pivot, movie_1, 6)
corr_movies = get_movie(movie_total, movie_list)

print(corr_movies)

def knn_method(movie_pivot, movie, num):
     


print('end')

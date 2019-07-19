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




print(links)

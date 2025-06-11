import pandas as pd
import random
import math
import numpy as np

def euclidean_dist(v1, v2):
    inner = 0
    for i in range(0, len(v1)):
        inner += (v1[i] - v2[i])**2
    return math.sqrt(inner)

# initially makes the centroids by choosing random values for each attribute/feature
def create_centroids(k, ranges):
    # ranges is a vector of size [number of features] with pair values representing the min and max ranges for each attribute
    centroids = []
    for i in range(0, k):
        c = []
        for feature in ranges:
            c.append(random.uniform(feature[0], feature[1]))
        centroids.append(c)
        c = []
    return(centroids)

# returns a centroid whose attribute values are the averages of all its respective cluster point attribute values
def update_centroid(cluster):
    if len(cluster) == 0:
        return cluster
    new_centroid = np.zeros(len(cluster[0]))
    for point in cluster:
        for a in range(0, len(point)):
            new_centroid[a] += point[a]

    for i in range(0, len(new_centroid)):
        new_centroid[i] /= len(cluster)
    return new_centroid

# returns an array of pairs (point, cluster) ***NOTE: clusters are numbered 0, 1 , 2 etc. and are correlated with the centroid indexes
def create_clusters(data, centroids):
    k = len(centroids)
    clusters = []
    
    for i in range(k):
        clusters.append([])

    for p in data:
        distances = [euclidean_dist(p, c) for c in centroids]
        best_cluster = distances.index(min(distances))
        clusters[best_cluster].append(p)
    return clusters

# Data loading + prep
songs = pd.read_csv("Data/spotify_songs.csv")
with_titles = songs[["track_id", "danceability", "energy", "loudness", "mode", "valence", "tempo"]]
subset = songs[["danceability", "energy", "loudness", "mode", "valence", "tempo"]]

data = subset.values

# k means, k = 2
feature_ranges = []
for feat in subset:
    if feat == "track_id":
        continue
    mini = subset[feat].min()
    maxi = subset[feat].max()
    feature_ranges.append([mini, maxi])

centroids = create_centroids(1, feature_ranges)
# create empty column cluster
for i in centroids:
    for x in i:
        print(x)

# for iteration in range(100):
#     clusters = create_clusters(data, centroids)
#     new_centroids = []
#     for cluster in clusters:
#         new_c = update_centroid(cluster)
#         if new_c is not None:
#             new_centroids.append(new_c)
#     centroids = new_centroids

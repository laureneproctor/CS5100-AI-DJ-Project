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
    return(centroids)

# returns a centroid whose attribute values are the averages of all its respective cluster point attribute values
def update_centroid(cluster):
    if len(cluster) == 0:
        return cluster
    return np.mean(cluster, axis=0)

# returns an array of pairs (point, cluster) ***NOTE: clusters are numbered 0, 1 , 2 etc. and are correlated with the centroid indexes
def create_clusters(data, centroids):
    k = len(centroids)
    clusters = []
    
    for i in range(k):
        clusters.append([])

    for p in data:
        distances = [euclidean_dist(p, c) for c in centroids]
        # we can use argmin to speed up this process
        best_cluster = np.argmin(distances)
        clusters[best_cluster].append(p)
    return clusters

# Data loading + prep
songs = pd.read_csv("Data/spotify_songs.csv")
with_titles = songs[["track_id", "danceability", "energy", "loudness", "mode", "valence", "tempo"]]
track_ids = songs[["track_id"]].values
subset = songs[["danceability", "energy", "loudness", "mode", "valence", "tempo"]]
features = subset.values
song_map = {tuple(features[i]): track_ids[i] for i in range(len(with_titles))}



# k means, k = 2
def find_ranges():
    feature_ranges = []
    for feat in subset:
        mini = subset[feat].min()
        maxi = subset[feat].max()
        feature_ranges.append([mini, maxi])
    return feature_ranges

def k_means_cluster(k, data):
    centroids = create_centroids(k, find_ranges())
    converged = False
    iteration = 1
    threshold = 1e-5
    while not converged:
        print(f"{iteration} iteration")
        clusters = create_clusters(data, centroids)
        new_centroids = []
        differences = []
        for i, cluster in enumerate(clusters):
            old_c = centroids[i]
            new_c = update_centroid(cluster)
            if new_c is not None:
                new_centroids.append(new_c)
            differences.append(euclidean_dist(old_c, new_c) < threshold)
        converged = all(differences)
        centroids = new_centroids
        iteration += 1
    return clusters

new_clusters = k_means_cluster(2, features)
for i in range(2):
    print(f"sample songs from cluster {i}")
    for data in new_clusters[i][:5]:
        print(song_map[tuple(data)])


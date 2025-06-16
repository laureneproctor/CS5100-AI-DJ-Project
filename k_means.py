import random
import math
import numpy as np

def euclidean_dist(v1, v2):
    inner = 0
    for i in range(0, len(v1)):
        inner += (v1[i] - v2[i])**2
    return math.sqrt(inner)

# initially makes the centroids by choosing random values for each attribute/feature
def create_centroids(k, min_ranges, max_ranges):
    centroids = []
    for i in range(0, k):
        c = []
        for j in range(len(min_ranges)):
            c.append(random.uniform(min_ranges[j], max_ranges[j]))
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

# k means, k = 2
def find_ranges(data):
    return np.min(data, axis=0), np.max(data, axis=0)

def k_means_cluster(k, data):
    min_range, max_range = find_ranges(data)
    centroids = create_centroids(k, min_range, max_range)
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
    return centroids

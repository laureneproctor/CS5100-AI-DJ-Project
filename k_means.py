import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_prep as prep
import pca
import pickle

K = 6
LABEL_MAP = {0: "Happy", 1: "Sad", 2: "Powerful", 3: "Chill" ,
             4: "Tense", 5: "Party"}

def euclidean_dist(v1, v2):
    inner = 0
    for i in range(0, len(v1)):
        inner += (v1[i] - v2[i])**2
    return math.sqrt(inner)

# initially makes the centroids by choosing random values for the first attribute, 
# then pick points that are as far away from one another as possible
def create_centroids(k, data):
    # randomly choose the first point
    centroids = [random.choice(data)[1]]
    # add the point whose minimum distance from the selected points as large as possible
    while len(centroids) < k:
        min_distances = []
        for _, p in data:
            distances = [euclidean_dist(p, c) for c in centroids]
            min_distances.append(min(distances))
        next_c = data[np.argmax(min_distances)][1]
        centroids.append(next_c)
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

    for i, p in data:
        distances = [euclidean_dist(p, c) for c in centroids]
        # use argmin to speed up this process
        best_cluster = np.argmin(distances)
        # adding the index of data into cluster to help get back the track name, tracck artist
        clusters[best_cluster].append((i, p))
    return clusters

def k_means_cluster(k, data):
    centroids = create_centroids(k, data)
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
            new_c = update_centroid([feature for (_, feature) in cluster])
            if new_c is not None:
                new_centroids.append(new_c)
                differences.append(euclidean_dist(old_c, new_c) < threshold)
        converged = all(differences)
        centroids = new_centroids
        iteration += 1
    return clusters, centroids

def find_radius(clusters, centroids):
    radius = []
    for i in range(len(centroids)):
        features = [feature for (_, feature) in clusters[i]]
        if len(features) == 0:
            radius.append(0)
        else:
            radius.append(np.max([euclidean_dist(data_point, centroids[i]) 
                                  for data_point in features]))
    return radius

def find_k():
    train_data = pca.TRAIN_DATA
    avg_radius = []
    for k in range(2, 16):
        radius_list = []
        for _ in range(5):
            clusters, centroids = k_means_cluster(k, train_data)
            radius = find_radius(clusters, centroids)
            radius_list.append(np.mean(radius))
        avg_radius.append(np.mean(radius_list))
    plt.plot(range(2, 16), avg_radius)
    plt.xlabel("Number of clusters")
    plt.ylabel("Average Radius")
    plt.show()

def save_clusters():
    with open("clusters.pkl", "wb") as f:
        pickle.dump(k_means_cluster(K, pca.TRAIN_DATA), f)

def load_clusters():
    with open("clusters.pkl", "rb") as f:
        return pickle.load(f)

CLUSTERS, CENTROIDS = load_clusters()

def output_clusters():
    dfs = []
    for i in range(K):
        features = [feature for (_, feature) in CLUSTERS[i]]
        # get back the features before pca
        data_before_pca = pca.MY_PCA.inverse_transform(features)
        # get back the features before normalization
        original_data = prep.ss.inverse_transform(data_before_pca)
        df = pd.DataFrame(original_data, columns = prep.FEATURES)
        df["cluster"] = i
        dfs.append(df)
    full_data = pd.concat(dfs, ignore_index=True)
    full_data.to_csv("clustered_songs.csv", index=False)
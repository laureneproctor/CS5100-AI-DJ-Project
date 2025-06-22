import numpy as np
import data_prep as prep
import random
import pca
import k_means

DATA_AFTER_PCA = pca.DATA_AFTER_PCA
TRAIN_DATA, TEST_DATA = pca.TRAIN_DATA, pca.TEST_DATA
SONG_MAP = prep.map_data()
LABEL_MAP = k_means.LABEL_MAP
K = k_means.K
CLUSTERS, CENTROIDS = k_means.CLUSTERS, k_means.CENTROIDS

sampled_test_data = random.sample(TEST_DATA, 60)

def show_options():
    print("Moods availabe to choose from: ")
    for key, value in LABEL_MAP.items():
        print(f"{key+1}: {value}")

def get_playlist(input_songs=sampled_test_data):
    cluster = None
    valid_cluster = [str(i) for i in range(1, K+1)]
    while cluster not in valid_cluster:
        show_options()
        cluster = input("Choose the mood of your playlist (type the number): ")
    # a playlist of track name and artist, print out for debugging 
    cluster_output = k_means.create_clusters(input_songs, CENTROIDS)
    playlist_info = [SONG_MAP[idx] for (idx, _) in cluster_output[int(cluster)-1]]
    raw_playlist = cluster_output[int(cluster)-1]
    # transfer the pcas back to original features to be used in local search
    pcas = [feature for (_, feature) in raw_playlist]
    normalized_features = pca.MY_PCA.inverse_transform(pcas)
    original_features = prep.ss.inverse_transform(normalized_features)
    playlist = [(raw_playlist[i][0], original_features[i]) for i in range(len(raw_playlist))]
    print(len(playlist_info))
    print(playlist_info)
    # The data structure is (index, array["danceability", "energy", "key", "loudness", "mode", "valence", "tempo"])
    return playlist

if __name__ == "__main__":
    get_playlist()
import numpy as np
import data_prep as prep
import pca
import k_means

DATA_AFTER_PCA = pca.DATA_AFTER_PCA
TRAIN_DATA, TEST_DATA = pca.TRAIN_DATA, pca.TEST_DATA
SONG_MAP = prep.map_data()
LABEL_MAP = k_means.LABEL_MAP
K = k_means.K
CLUSTERS, CENTROIDS = k_means.CLUSTERS, k_means.CENTROIDS

sampled_test_data = TEST_DATA[np.random.choice(len(TEST_DATA), 100, replace=False)]

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
    playlist = [SONG_MAP[idx] for (idx, _) in 
                k_means.create_clusters(input_songs, CENTROIDS)[int(cluster)-1]]
    print(len(playlist))
    print(playlist)
    return playlist
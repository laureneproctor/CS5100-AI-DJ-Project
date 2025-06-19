import data_prep as prep
import pca
import k_means

DATA_AFTER_PCA = pca.DATA_AFTER_PCA
TRAIN_DATA, TEST_DATA = pca.TRAIN_DATA, pca.TEST_DATA
SONG_MAP = prep.map_data()
K = k_means.K
CLUSTERS, CENTROIDS = k_means.CLUSTERS, k_means.CENTROIDS

def get_playlist(k=K, input_songs=TEST_DATA):
    return [SONG_MAP[idx] for (idx, _) in k_means.create_clusters(input_songs, CENTROIDS)[k]]

cluster = input("Enter the cluster: ")
print(get_playlist(int(cluster))[:5])
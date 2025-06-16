import data_prep
import pca
import k_means

features = data_prep.prep_data()
data_after_pca = pca.apply_pca(3, features)
train_data, test_data = data_prep.split_data(data_after_pca)
song_map = data_prep.map_data(data_after_pca)

def get_clusters(features):
    return k_means.k_means_cluster(2, features)

def get_playlist(cluster, train_data=train_data, input_songs=test_data):
    centroids = get_clusters(train_data)
    return [song_map[tuple(feature)] for feature in k_means.create_clusters(input_songs, centroids)[cluster]]

cluster = input("Enter the cluster: ")
print(get_playlist(int(cluster))[:5])
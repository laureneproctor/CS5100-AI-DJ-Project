"""
This file serves as a space to stored commented out code in case it should be needed again
"""

"""
This code comes from main.py: It was initially used for printing the track names, and artist names of the 
selected cluster of songs.
"""
# cluster_output = k_means.create_clusters(input_songs, CENTROIDS)
    # playlist_info = [SONG_MAP[idx] for (idx, _) in cluster_output[int(cluster)-1]]
    # raw_playlist = cluster_output[int(cluster)-1]
    # # transfer the pcas back to original features to be used in local search
    # pcas = [feature for (_, feature) in raw_playlist]
    # normalized_features = pca.MY_PCA.inverse_transform(pcas)
    # original_features = prep.ss.inverse_transform(normalized_features)
    # playlist = [(raw_playlist[i][0], original_features[i]) for i in range(len(raw_playlist))]
    # print(len(playlist_info))
    # print(playlist_info)
    # # The data structure is (index, array["danceability", "energy", "key", "loudness", "mode", "valence", "tempo"])
    # return playlist
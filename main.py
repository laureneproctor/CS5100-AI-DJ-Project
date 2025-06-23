import numpy as np
import data_prep as prep
import random
import pca
import k_means
import playlist as play
import pandas as pd

DATA_AFTER_PCA = pca.DATA_AFTER_PCA
TRAIN_DATA, TEST_DATA = pca.TRAIN_DATA, pca.TEST_DATA
SONG_MAP = prep.map_data()
LABEL_MAP = k_means.LABEL_MAP
K = k_means.K
CLUSTERS, CENTROIDS = k_means.CLUSTERS, k_means.CENTROIDS
NO_SAMPLE_SONGS = 75

sampled_test_data = random.sample(TEST_DATA, NO_SAMPLE_SONGS)

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

    # Collect songs for selected cluster
    cluster_output = k_means.create_clusters(input_songs, CENTROIDS)
    cluster_index = int(cluster) - 1
    cluster_songs = cluster_output[cluster_index]  # A list of (idx, vector) for the selected mood
    playlist_info = [SONG_MAP[idx] for (idx, _) in cluster_songs]

    # Convert playlist_info to DataFrame for local search
    playlist_df = pd.DataFrame(
        playlist_info,
        columns=['track_name', 'track_artist', 'tempo', 'key']
    )
    
    # Run local search (hill climbing)
    ordered_playlist, cost = play.search(playlist_df)

    # Display results
    print(f"\nOptimized playlist (transition cost: {cost:.2f}):")
    for idx, song in enumerate(ordered_playlist):
        print(f"{idx+1:02d}. {song['track_name']} - {song['track_artist']} (Key: {song['key']}, Tempo: {song['tempo']})")



if __name__ == "__main__":
    get_playlist()

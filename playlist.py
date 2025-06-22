import pandas as pd
import math
import random

"""
Features that all songs have: danceability, energy, key, loudness, mode, speechiness, instrumentalness, valence, tempo

Features that will be used in our cost function is key, and tempo
"""


def load_songs(file_path="Data/spotify_songs.csv", sample_size=20):
    df = pd.read_csv(file_path)
    df = df[['track_name', 'track_artist', 'tempo', 'key']].dropna().reset_index(drop=True)

    if len(df) >= sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        print("Only {len(df)} songs available, using all of them.")

    return df

# key_dist returns the circular distance betweeon 2 keys
def key_dist(k1, k2):
    return min(abs(k1-k2), 12 - abs(k1-k2))

# tempo_dist returns the distance between two tempos
def tempo_dist(tempo1, tempo2):
    return abs(tempo1 - tempo2)

def transition_cost(song1, song2, alpha=1.0, beta=1.0):
    return alpha * key_dist(song1['key'], song2['key']) + beta * tempo_dist(song1['tempo'], song2['tempo'])

def total_cost(playlist):
    cost = 0
    for i in range (len(playlist) - 1):
        cost += transition_cost(playlist[i], playlist[i+1])
    return cost

# Performs hill climbing to find best order of playlist
def search(songs, max_iterations=1000):
    best_order = songs.copy().to_dict(orient="records")
    best_cost = total_cost(best_order)

    for _ in range(max_iterations):
        # Different neighbors by swapping any two songs
        i, j = random.sample(range(len(best_order)),2)
        neigh = best_order.copy()
        neigh[i], neigh[j] = neigh[j], neigh[i]

        current_cost = total_cost(neigh)
        if current_cost < best_cost:
            best_order = neigh
            best_cost = current_cost

    return best_order, best_cost

if __name__ == "__main__":
    songs = load_songs()
    ordered_playlist, cost = search(songs)
    print(f"Final cost: {cost: .2f}")
    for index, song in enumerate(ordered_playlist):
        print(f"{index+1:02d}. {song['track_name']} - {song['track_artist']} (Key: {song['key']}, Tempo: {song['tempo']})")
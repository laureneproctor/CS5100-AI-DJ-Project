import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA = pd.read_csv("Data/spotify_songs.csv")
ss = StandardScaler()
FEATURES = ["danceability", "energy", "key", "loudness", "mode", "valence", "tempo"]

# returns data with relevant columns to be trained on and normalize it
def prep_data(data=DATA):
    new_data = data[FEATURES].values
    return ss.fit_transform(new_data)

# get back the song name, artist from index, this function returns a map
def map_data():
    track_info = DATA[["track_name", "track_artist"]].values
    return {i: (track_name, track_artist) for i, (track_name, track_artist) in 
            enumerate(track_info)}

def split_data(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    return train, test

NORMALIZED_DATA = prep_data()
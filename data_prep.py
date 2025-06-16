import pandas as pd
from sklearn.model_selection import train_test_split

DATA = pd.read_csv("Data/spotify_songs.csv")

# returns data with relevant columns to be trained on 
def prep_data(data=DATA):
    return data[["danceability", "energy", "loudness", "mode", "valence", "tempo"]].values

# get back the song name from features used to train data, this function returns a map
def map_data(data):
    track_name = DATA["track_name"].values
    return {tuple(data[i]): track_name[i] for i in range(len(track_name))}

def split_data(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    return train, test


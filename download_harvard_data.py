# download_harvard_kaggle.py

import os
import zipfile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_extract_kaggle_dataset():
    os.makedirs("data", exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files("pavanelisetty/sample-audio-files-for-speech-recognition", path="data", unzip=True)

def prepare_harvard_dataframe():
    audio_dir = "data"
    all_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
    texts = [
        "The birch canoe slid on the smooth planks.",
        "Glue the sheet to the dark blue background.",
        "It's easy to tell the depth of a well.",
        "These days a chicken leg is a rare dish.",
        "Rice is often served in round bowls.",
        "The juice of lemons makes fine punch.",
        "The box was thrown beside the parked truck.",
        "The hogs were fed chopped corn and garbage.",
        "Four hours of steady work faced us.",
        "Large size in stockings is hard to sell.",
        "The boy was there when the sun rose.",
        "A rod is used to catch pink salmon.",
        "The source of the huge river is the clear spring.",
        "Kick the ball straight and follow through.",
        "Help the woman get back to her feet.",
        "A pot of tea helps to pass the evening.",
        "Smoky fires lack flame and heat.",
        "The soft cushion broke the man's fall.",
        "The salt breeze came across from the sea.",
        "The girl at the booth sold fifty bonds.",
        "The small pup gnawed a hole in the sock.",
        "The fish twisted and turned on the bent hook.",
        "Press the pants and sew a button on the vest.",
        "The swan dive was far short of perfect.",
        "The beauty of the view stunned the young boy.",
        "Two blue fish swam in the tank.",
        "Her purse was full of useless trash.",
        "The colt reared and threw the tall rider.",
        "It snowed, rained, and hailed the same morning.",
        "Read verse out loud for pleasure.",
        "The huge pile of coal is blazing brightly.",
        "Four hours of steady work faced us.",
        "The couch cover and hall drapes were blue.",
        "A wisp of cloud hung in the blue air.",
        "A gray mare walked before the colt.",
        "He wrote his last novel there at the inn.",
        "Even the worst will beat his low score.",
        "The cement had dried when he moved it.",
        "The loss of the second ship was hard to take.",
        "The bark of the pine tree was shiny and dark.",
        "The scrawl on the wall would not wash off.",
        "A load of wheat came up the coast.",
        "We need more examples of fast thought.",
        "When you hear the bell, come quickly.",
        "A ripe plum is fit for a king's palate.",
        "The colt reared and threw the tall rider.",
        "It snowed, rained, and hailed the same morning.",
        "These days a chicken leg is a rare dish.",
        "Help the woman get back to her feet.",
        "The juice of lemons makes fine punch.",
        "Rice is often served in round bowls.",
    ]

    all_files = all_files[:len(texts)]
    df = pd.DataFrame({
        "file": [os.path.join(audio_dir, f) for f in all_files],
        "text": texts
    })
    df.to_csv("data/harvard_metadata.csv", index=False)

if __name__ == "__main__":
    download_and_extract_kaggle_dataset()
    prepare_harvard_dataframe()

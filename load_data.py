import pandas as pd  # for working with dataframes
from os import listdir  # for retrieving files from directory
from os.path import isfile, join  # for retrieving files from directory
from pathlib import Path  # for retrieving files from directory
import re


def load_text_files(data_dir="./modified_texts/"):
    """Loads and processes text files from a directory, returning file names and their contents"""
    file_names = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    file_names.remove('.DS_Store')
    file_names.sort()

    subchapter_names = [name[:-4] for name in file_names]
    texts = [Path(data_dir + file).read_text().replace('\n', ' ')
             for file in file_names]

    return file_names, subchapter_names, texts


def segment_nahuatl(text):
    """Segments Nahuatl text into morphemes using common patterns"""
    suffixes = ['tzin', 'tli', 'tin', 'que', 'pan', 'can', 'tic', 'toc',
                'tia', 'lia', 'hui', 'tla', 'tli', 'tl', 'li', 'in']
    prefixes = ['ni', 'ti', 'xi', 'mo', 'no', 'to', 'ne', 'te', 'tla', 'on']

    words = str(text).split()
    segmented_words = []

    for word in words:
        if len(word) < 3:
            segmented_words.append(word)
            continue

        segmented = word
        for suffix in suffixes:
            if segmented.endswith(suffix):
                segmented = segmented[:-len(suffix)] + '-' + suffix
                break

        for prefix in prefixes:
            if segmented.startswith(prefix):
                segmented = prefix + '-' + segmented[len(prefix):]
                break

        segmented_words.append(segmented)

    return ' '.join(segmented_words)


def load_subchapter_data(filepath="./verify_subchapter.csv"):
    """Loads and processes subchapter data from CSV, returning relevant columns"""
    df = pd.read_csv(filepath)
    subchapter = df["subchapter"].str.split(";", expand=True).fillna(0)

    nahuatl = df["nahuatl"].apply(segment_nahuatl)

    return {
        'subchapter': subchapter,
        'nahuatl': nahuatl,
        'official_name': df["official_name"],
        'ID': df["ID"],
        'type': df["type"]
    }


def process_texts_with_morphemes(texts):
    """Processes a list of texts to segment them into morphemes"""
    return [segment_morphemes(text) for text in texts]

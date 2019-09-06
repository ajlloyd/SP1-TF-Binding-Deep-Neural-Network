import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import numpy as np

print(os.getcwd())
np.random.seed(42)
raw_data = pd.read_csv("./Raw_data.csv").values

def base_to_numeric(base_data):
    sequences = base_data[:,0]
    labels = base_data[:,1]
    conversion = {"A" : [0,0], "T" : [0,1], "C" : [1,0], "G" : [1,1]}
    # Sequence Conversion (to Numeical):
    converted_data = []
    for sequence in sequences:
        converted_seq = []
        for base in sequence:
            for key, val in conversion.items():
                if base == key:
                    converted_seq.append(val)
        flattened = [y for x in converted_seq for y in x]
        converted_data.append(flattened)
    #Label Conversion (to binary):
    converted_labels = []
    for label in labels:
        if label == "binding site":
            converted_labels.append(1)
        elif label == "non-binding site":
            converted_labels.append(0)
    converted_data = np.c_[np.array(converted_data), np.array(converted_labels).reshape(-1,1)]
    np.random.shuffle(converted_data)
    return converted_data
processed_data = base_to_numeric(raw_data)


def train_test_split(data, test_size=0.2):
    cutoff_index = int(len(data) * test_size)
    test_set = data[:cutoff_index, :]
    train_set = data[cutoff_index:, :]
    X_train = train_set[:, :-1]
    X_test = test_set[:, :-1]
    y_train = train_set[:, -1].reshape(-1,1)
    y_test = test_set[:, -1].reshape(-1,1)
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(processed_data, test_size=0.2)

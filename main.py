import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.callbacks import TensorBoard
import time

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
    converted_data = np.c_[np.array(converted_data).astype(float), np.array(converted_labels).reshape(-1,1)]
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

layers = [3, 4, 5]
nodes = [128, 256, 384]
activations = ["relu", "sigmoid"]
dropouts = [0.1, 0.15, 0.2]

print("sss")

"""for n_layers in layers:
    for n_nodes in nodes:
        for activation in activations:
            for dropout in dropouts:
                NAME = f"NN-[{n_layers}x{n_nodes}-nodes]-[{n_layers}x{dropout}-dropout]-[{activation}-activation]-{int(time.time())}"
                tensorboard = TensorBoard(log_dir=".\logs\\{}".format(NAME))

                model = Sequential()
                for i in range(n_layers):
                    model.add(Dense(n_nodes, activation=activation))
                    model.add(Dropout(dropout))

                model.add(Dense(1, activation="sigmoid"))
                model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
                model.fit(X_train,y_train,epochs=20, validation_split=0.1, callbacks=[tensorboard])"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.callbacks import TensorBoard
import time
import matplotlib.pyplot as plt

##### Checking cwd and import of raw DNA base .CSV
print(os.getcwd())
np.random.seed(42)
raw_data = pd.read_csv("./Raw_data.csv").values

##### Converting the DNA bases (ATCG) into numeric
def base_to_numeric(base_data):
    sequences = base_data[:,0]
    labels = base_data[:,1]
    conversion = {"A" : [1,0,0], "T" : [1,0,1], "C" : [1,1,0], "G" : [1,1,1]}
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

##### Splitting the data into train and test sets:
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

#### Finding the best model:
def finding_best_model(X_train, layers, nodes, activations, dropouts):
    for n_layers in layers:
        for n_nodes in nodes:
            for activation in activations:
                for dropout in dropouts:
                    NAME = f"NN-[{n_layers}x{n_nodes}-nodes]-[{n_layers}x{dropout}-dropout]-[{activation}-activation]-{int(time.time())}"
                    tensorboard = TensorBoard(log_dir=".\logs\\{}".format(NAME))
                    model = Sequential()
                    model.add(Input((X_train.shape[1:])))
                    for i in range(n_layers):
                        model.add(Dense(n_nodes, activation=activation))
                        model.add(Dropout(dropout))
                    model.add(Dense(1, activation="sigmoid"))
                    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
                    model.fit(X_train,y_train,epochs=10, validation_split=0.2, callbacks=[tensorboard])
#finding_best_model(X_train, layers = [3, 4, 5], nodes = [384, 448],activations = ["relu"],dropouts = [0.1, 0.15])

#-------------------------------------------------------------------------------
##### Confusion_matrix and clf metrics classes:

class matrix:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
    def confusion_matrix(self):
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        true_pred = np.c_[self.y_true, self.y_pred]
        for row in true_pred:
            if row[0] == 1 and row[1] == 1:
                TP += 1
            if row[0] == 0 and row[1] == 0:
                TN += 1
            if row[0] == 1 and row[1] == 0:
                FN += 1
            if row[0] == 0 and row[1] == 1:
                FP += 1
        return np.array(([TP, FP],[FN, TN]))

class metrics(matrix):
    def __init__(self, y_true, y_pred):
        super().__init__(y_true, y_pred)
        self.cm = super().confusion_matrix().ravel()
        self.TP = self.cm[0]
        self.FP = self.cm[1]
        self.FN = self.cm[2]
        self.TN = self.cm[3]
    def accuracy(self):
        accuracy = (self.TN + self.TP) / (self.TP + self.FP + self.FN + self.TN)
        return accuracy
    #When it is {A} how often does {B} occur?
    def recall(self):
        # Sensitivity/Recall/TPR --> when {y_true = 1} how often does {y_pred = 1}?
        return self.TP / (self.TP + self.FN)
    def false_positive_rate(self):
        # FPR --> when {y_true = 0} how often does {y_pred = 1}?
        return self.FP / (self.TN + self.FP)
    def true_negative_rate(self):
        # Specificity --> when {y_true = 0} how often does {y_pred = 0}?
        actual_negatives = len(self.y_true[self.y_true==0])
        return self.TN / (self.TN + self.FP)
    def precision(self):
        # Precision --> when {y_pred = 1} how often does {y_true = 1}
        return self.TP / (self.TP + self.FP)
    def F1_score(self):
        return (2 * ((self.precision() * self.recall())/(self.precision() + self.recall())))


#-------------------------------------------------------------------------------
##### Best Model test:
def test_best_model():
    NAME = f"NN-[3x448-nodes]-[3x0.1-dropout]-[relu-activation]-{int(time.time())}"
    tensorboard = TensorBoard(log_dir=".\logs\\{}".format(NAME))
    model = Sequential()
    model.add(Input((X_train.shape[1:])))
    for i in range(3):
        model.add(Dense(448, activation="relu"))
        model.add(Dropout(0.1))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    model.fit(X_train,y_train,epochs=10, validation_split=0.2, callbacks=[tensorboard])
    model.save("best_model.model")
#test_best_model()

##### Making predictions
model = load_model("best_model.model")
predict = model.predict([X_test])
np.place(predict, predict >= 0.45, [1])
np.place(predict, predict < 0.45, [0])

##### Confusion matrix:
mat = matrix(y_test, predict)
mat.confusion_matrix()

##### Classifier metrics:
acc = metrics(y_test, predict)
print("Accuracy:                ", acc.accuracy())
print("Sensitivity/Recall/TPR:  ", acc.recall())
print("FPR:                     ", acc.false_positive_rate())
print("Specificity/TNR:         ", acc.true_negative_rate())
print("Precision:               ", acc.precision())
print("F1 Score:                ", acc.F1_score())

##### ROC curves:
predict = model.predict([X_test])
def replace(y_pred, threshold):
    y_pred_threshold = []
    for value in y_pred:
        if value >= threshold:
            y_pred_threshold.append(1)
        elif value < threshold:
            y_pred_threshold.append(0)
    return np.array(y_pred_threshold).reshape(-1,1)

def plot_ROC(y_true, y_pred, n_threshold=10):
    false_positive_x = []
    sensitivity_y = []
    for threshold in np.linspace(0,1,n_threshold):
        y_thresh = replace(y_pred, threshold)
        th_metrics = metrics(y_true, y_thresh)
        false_positive_x.append(th_metrics.false_positive_rate())
        sensitivity_y.append(th_metrics.recall())
    ax = plt.subplot(111)
    ax.plot(false_positive_x, sensitivity_y, "r-", label="AUC =")
    ax.plot([0, 0.5, 1.0], [0, 0.5, 1.0], "b--", label="AUC = 0.5 (random)")
    ax.set_ylabel("Sensitivity:")
    ax.set_xlabel("False Positive Rate:")
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    ax.legend()
    plt.show()
plot_ROC(y_test, predict, n_threshold=50)

def plot_precision_recall():
    pass

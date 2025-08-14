import pandas as pd
import numpy as np
import random
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def load_data(train_path, test_path, train_size, test_size):
    train = pd.read_csv(train_path, header=None)
    test = pd.read_csv(test_path, header=None)
    train_indices = random.sample(range(len(train)), train_size)
    test_indices = random.sample(range(len(test)), test_size)
    train = train.iloc[train_indices]
    test = test.iloc[test_indices]
    x_train = train.iloc[:, 1:].values
    y_train = train.iloc[:, 0].values
    x_test = test.iloc[:, 1:].values
    y_test = test.iloc[:, 0].values
    return x_train, y_train, x_test, y_test

def normalize(data):
    return data.astype(np.float32) / 255.0

def evaluate_knn(x_train, y_train, x_test, y_test, k, distance_metric):
    knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric, n_jobs=-1)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, predictions

train_csv_path = r'dataset/emnist-balanced-train.csv'
test_csv_path = r'dataset/emnist-balanced-test.csv'

x_train, y_train, x_test, y_test = load_data(train_csv_path, test_csv_path, 112800, 18800)
x_train = normalize(x_train)
x_test = normalize(x_test)

pca = PCA(n_components=50)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

k = 7
distance_metric = 'euclidean'

accuracy, predictions = evaluate_knn(x_train, y_train, x_test, y_test, k, distance_metric)
print(f'KNN classification accuracy on test set: {accuracy:.4f}')

with open('predictions_and_labels.pkl', 'wb') as f:
    pickle.dump((predictions, y_test), f)
#accuracy:0.8086

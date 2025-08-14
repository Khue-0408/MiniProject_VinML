import pandas as pd
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
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
    return accuracy

train_csv_path = r'C:\Users\Admin\Documents\python\EMNIST\emnist-balanced-train.csv'
test_csv_path = r'C:\Users\Admin\Documents\python\EMNIST\emnist-balanced-test.csv'

x_train, y_train, x_test, y_test = load_data(train_csv_path, test_csv_path, 112800, 18800)
x_train = normalize(x_train)
x_test = normalize(x_test)

pca = PCA(n_components=50)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

k_values = [1, 3, 5, 7, 9, 15]
distance_metrics = ['mahattan', 'euclidean', 'chebyshev']
kf = KFold(n_splits=3, shuffle=True, random_state=42)

results = pd.DataFrame(index=k_values, columns=distance_metrics)

with open('knn_results.txt', 'w') as file:
    file.write('k\tdistance_metric\taccuracy\n')
    for k in k_values:
        for distance_metric in distance_metrics:
            cv_accuracies = []
            for train_index, val_index in kf.split(x_train):
                x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
                accuracy = evaluate_knn(x_train_fold, y_train_fold, x_val_fold, y_val_fold, k, distance_metric)
                cv_accuracies.append(accuracy)
            mean_cv_accuracy = np.mean(cv_accuracies)
            results.loc[k, distance_metric] = mean_cv_accuracy
            file.write(f'{k}\t{distance_metric}\t{mean_cv_accuracy:.4f}\n')

results = results.apply(pd.to_numeric, errors='coerce').fillna(0)

best_params = results.stack().idxmax()
best_k, best_distance_metric = best_params
best_accuracy = results.loc[best_k, best_distance_metric]

print(f'Best k: {best_k}')
print(f'Best distance metric: {best_distance_metric}')
print(f'Best cross-validation accuracy: {best_accuracy:.4f}')

best_knn = KNeighborsClassifier(n_neighbors=best_k, metric=best_distance_metric, n_jobs=-1)
best_knn.fit(x_train, y_train)
predictions = best_knn.predict(x_test)
test_accuracy = accuracy_score(y_test, predictions)

print(f'KNN classification accuracy on test set: {test_accuracy:.4f}')

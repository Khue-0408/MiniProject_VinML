import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from DecisionTree import DecisionTree
import joblib

n_models = 100
train_data = pd.read_csv(r'dataset/emnist-balanced-train.csv', header=None)
test_data = pd.read_csv(r'dataset/emnist-balanced-test.csv', header=None)
train_limit = 112800
X_test, y_test = test_data.iloc[:, 1:].values, test_data.iloc[:, 0].values
X_test = X_test[:1]
y_test = y_test[:1]
X_test = X_test.astype('float64')
y_test = y_test.astype('intc')
X_test /= 255.0
for i in range(n_models):
    train_sample = train_data.sample(n=train_limit, replace=True, random_state=np.random.randint(10000))
    X_train, y_train = train_sample.iloc[:, 1:].values, train_sample.iloc[:, 0].values
    X_train = X_train.astype('float64')
    y_train = y_train.astype('intc')
    X_train /= 255.0
    pca = PCA(n_components=50)
    X_train = pca.fit_transform(X_train)
    X_test_transformed = pca.transform(X_test)
    model = DecisionTree(max_depth=15)
    model.fit(X_train, y_train)
    model_filename = f'C:\\Users\\Admin\\Documents\\python\\ML\\RandomForest\\Model{i + 1}.pkl'
    joblib.dump((model, pca), model_filename)
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Model {i + 1} - Training Accuracy: {train_accuracy:.4f}")
    y_test_pred = model.predict(X_test_transformed)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Model {i + 1} - Test Accuracy: {test_accuracy:.4f}")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from DecisionTree import DecisionTree
import joblib

train_data = pd.read_csv(r'dataset/emnist-balanced-train.csv', header=None)
test_data = pd.read_csv(r'dataset/emnist-balanced-test.csv', header=None)
train_limit = 112800
train_sample = train_data.sample(n=train_limit, random_state=42)
X_train, y_train = train_sample.iloc[:, 1:].values, train_sample.iloc[:, 0].values
X_test, y_test = test_data.iloc[:, 1:].values, test_data.iloc[:, 0].values
X_test = X_test[:1880]
y_test = y_test[:1880]
X_train = X_train.astype('float64')
y_train = y_train.astype('intc')
X_test = X_test.astype('float64')
y_test = y_test.astype('intc')
X_train /= 255.0
X_test /= 255.0
pca = PCA(n_components=50)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
model = DecisionTree(max_depth=15)
model.fit(X_train, y_train)
joblib.dump((model, pca), r'C:\Users\Admin\Documents\python\ML\RandomForest\Model3.pkl')
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

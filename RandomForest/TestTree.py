import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from DecisionTree import DecisionTree
import joblib
#{...} file name in folder TreeModels
model, pca = joblib.load(r'RandomForest/RandomForest/TreeModels/{...}.pkl')

test_data = pd.read_csv(r'dataset/emnist-balanced-test.csv', header=None)
X_test, y_test = test_data.iloc[:, 1:].values, test_data.iloc[:, 0].values
X_test = X_test[:18800]
y_test = y_test[:18800]
X_test = X_test.astype('float64')
y_test = y_test.astype('intc')
X_test /= 255.0
X_test = pca.transform(X_test)
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

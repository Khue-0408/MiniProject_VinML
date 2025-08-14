from scipy.stats import mode
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA
test_data = pd.read_csv(r'dataset/emnist-balanced-test.csv', header=None)
X_test, y_test = test_data.iloc[:, 1:].values, test_data.iloc[:, 0].values
X_test = X_test[:18800]
y_test = y_test[:18800]
X_test = X_test.astype('float64')
y_test = y_test.astype('intc')
X_test /= 255.0
model_folder = r'RandomForest/RandomForest/ForestModels'
model_paths = [os.path.join(model_folder, filename) for filename in os.listdir(model_folder) if filename.endswith('.pkl')]
def evaluate_model(model_path, X):
    try:
        model, pca = joblib.load(model_path)
        X_pca = pca.transform(X)
        predictions = model.predict(X_pca)
        return predictions
    except Exception as e:
        print(f"Failed to load or predict with model {model_path}: {e}")
        return None
all_predictions = [evaluate_model(path, X_test) for path in model_paths]
def random_forest_vote(predictions):
    valid_predictions = [pred for pred in predictions if pred is not None]
    if not valid_predictions:
        raise ValueError("None of the models could be loaded or made predictions.")
    predictions_transposed = np.array(valid_predictions).T
    majority_votes, _ = mode(predictions_transposed, axis=1)
    return majority_votes.flatten()
final_predictions = random_forest_vote(all_predictions)
final_accuracy = accuracy_score(y_test, final_predictions)
final_precision = precision_score(y_test, final_predictions, average='macro', zero_division=1)
final_recall = recall_score(y_test, final_predictions, average='macro', zero_division=1)
print(f"Random Forest Ensemble Accuracy: {final_accuracy:.4f}")
print(f"Precision: {final_precision:.4f}, Recall: {final_recall:.4f}")

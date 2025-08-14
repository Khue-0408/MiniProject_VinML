# FNN and CNN

## How to Use

1. **Upload Files:**
   - Upload all the files in the `FNN_and_CNN` folder to your Google Drive.

2. **Open in Google Colab:**
   - Open the `EMNIST_classification.ipynb` file in Google Colab.

3. **Run the Code:**
   - Execute the code cells in order.

## Folders

- `model`, `model_cnn`, `models_fnn` are used to store the trained weights of the models.

## Contact

In case the code does not run, contact me through [Teams](mailto:dao.th220061@sis.hust.edu.vn).


# KNN

## How to Use

1. **Set Dataset Paths:**
   - Change the `train_csv_path` and `test_csv_path` to the path of the EMNIST training and testing dataset on your PC.

2. **Tuning Parameters in `Tuning.py`:**
   - Change the `k_values` list to the list of k values you want to tune with.
   - Change the `distance_metrics` list to the list of distance metrics (from `scikit-learn`) you want to tune with.

3. **Model Parameters in `Model.py`:**
   - Change `x_test`, `y_test` to the size of the training and testing dataset.
   - Set `k` and `distance_metric` to the values you want to test with.
   - The model will save a `.pkl` file with the list of predicted labels and their true labels.

## Libraries Needed

The libraries needed for this project are listed in the `requirements.txt` file.


# Random Forest (RF)

## How to Use
1. Download Microsoft C++ build tools
   - Download Microsoft C++ build tools version 14 or later from https://visualstudio.microsoft.com/visual-cpp-build-tools/
	(This is for compiling the model into C++ using Cython)
2. Build the Module:
   - Navigate to the model folder (the folder with the file DecisionTree.pyx) and run the following command in the terminal:
     
     python setup.py build_ext --inplace
     

3. Set Dataset Paths:
   - Change the train_data and test_data to the filepath of the EMNIST training and testing dataset on your PC.

4. *Create a Single Tree Model (MakeTree.py):*
   - In model = DecisionTree(max_depth=XXX), change XXX to the max_depth of the tree you want to create.
   - In joblib.dump((model, pca), r'XXX'), change XXX to the filepath where you want to save the Tree and its name.

5. *Create Multiple Trees with Bootstrap (MakeTrees.py):*
   - Change n_models to the number of trees you want to create.
   - In model = DecisionTree(max_depth=XXX), change XXX to the max_depth of the tree you want to create.
   - In model_filename = f'XXX', change XXX to the filepath and the name of the trees you want to save.

6. *Test the Accuracy of a Specific Tree (TestTree.py):*
   - In model, pca = joblib.load(r'XXX'), change XXX to the filepath of the tree you want to test.

7. *Random Forest Testing (ForestTest.py):*
   - Change model_folder to the filepath of the folder containing all the models you want to test with.
   - Ensure model_folder points to the filepath of the folder ForestModels on your PC.




# cython: language_level=3
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef int classes = 47

cdef tuple getSplit(np.ndarray[np.float64_t, ndim=2] X_set, np.ndarray[np.int_t, ndim=1] y_set):
    cdef int chosenFeature = -1
    cdef double chosenThreshold = -1.0
    cdef double lowestImpurity = 1.0

    if X_set.shape[0] == 0:
        return chosenFeature, chosenThreshold

    cdef int start_feature = 0
    cdef int end_feature = X_set.shape[1]  # Adjusted to use the number of features in X_set
    cdef int feature
    cdef double threshold, gini

    for feature in range(start_feature, end_feature):
        threshold, gini = getThreshold(X_set[:, feature], y_set)
        if gini < lowestImpurity:
            chosenFeature = feature
            chosenThreshold = threshold
            lowestImpurity = gini
    return chosenFeature, chosenThreshold

cdef tuple getThreshold(np.ndarray[np.float64_t, ndim=1] X, np.ndarray[np.int_t, ndim=1] y_set):
    cdef double chosenThreshold = -1.0
    cdef double bestGini = gini_impurity(y_set)
    cdef np.ndarray thresholds = np.quantile(X, np.linspace(0, 1, num=50)[1:-1])
    cdef np.ndarray unique_thresholds = np.unique(thresholds)

    cdef double threshold, gini
    cdef np.ndarray left_mask, right_mask, y_left, y_right

    for threshold in unique_thresholds:
        left_mask = X <= threshold
        right_mask = ~left_mask
        y_left = y_set[left_mask]
        y_right = y_set[right_mask]

        gini = gini_impurity(y_left) * np.sum(left_mask) + gini_impurity(y_right) * np.sum(right_mask)
        gini /= X.size  # Normalize by the number of samples to avoid size bias

        if gini < bestGini:
            bestGini = gini
            chosenThreshold = threshold
    return chosenThreshold, bestGini

cdef class DecisionNode:
    cdef public int feature
    cdef public double threshold
    cdef public int depth
    cdef public int max_depth
    cdef DecisionNode left, right
    cdef public object result  # Using 'object' to handle both int and NoneType

    def __init__(self, int feature=-1, double threshold=-1.0, int depth=0, int max_depth=2, left=None, right=None, result=None):
        self.feature = feature
        self.threshold = threshold
        self.depth = depth
        self.max_depth = max_depth
        self.left = left
        self.right = right
        self.result = result
    def __reduce__(self):
        # Return the constructor and its arguments along with left and right children
        return (self.__class__, (self.feature, self.threshold, self.depth, self.max_depth, self.left, self.right, self.result))

    def grow(self, np.ndarray[np.float64_t, ndim=2] X_train, np.ndarray[int, ndim=1] y_train):
        if X_train.shape[0] == 0 or y_train.size == 0:
            return  # Prevent growing on empty data

        if gini_impurity(y_train) == 0 or self.depth >= self.max_depth:
            self.result = np.bincount(y_train).argmax()
            return

        cdef tuple split_results = split(X_train, y_train, self.feature, self.threshold)
        cdef np.ndarray X_left, y_left, X_right, y_right
        X_left, y_left, X_right, y_right = split_results

        if X_left.size > 0:
            left_feature, left_threshold = getSplit(X_left, y_left)
            self.left = DecisionNode(left_feature, left_threshold, self.depth + 1, self.max_depth)
            self.left.grow(X_left, y_left)
        if X_right.size > 0:
            right_feature, right_threshold = getSplit(X_right, y_right)
            self.right = DecisionNode(right_feature, right_threshold, self.depth + 1, self.max_depth)
            self.right.grow(X_right, y_right)
    cdef split(self, np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.int32_t, ndim=1] y):
        cdef list X_left = []
        cdef list y_left = []
        cdef list X_right = []
        cdef list y_right = []
        cdef int index
        for index in range(X.shape[0]):
            if X[index, self.feature] <= self.threshold:
                X_left.append(X[index].tolist())  # Ensuring data is appended as a list
                y_left.append(y[index])
            else:
                X_right.append(X[index].tolist())  # Ensuring data is appended as a list
                y_right.append(y[index])

    # Convert lists to numpy arrays before returning
        return (np.array(X_left, dtype=np.float64),
                np.array(y_left, dtype=np.int32),
                np.array(X_right, dtype=np.float64),
                np.array(y_right, dtype=np.int32))


    cdef classify(self, np.ndarray[np.float64_t, ndim=1] X):
        if self.result is not None:
            return self.result
        elif X[self.feature] <= self.threshold:
            return self.left.classify(X)
        else:
            return self.right.classify(X)

cdef double gini_impurity(np.ndarray[np.int_t, ndim=1] y_set):
    if y_set.size == 0:
        return 0.0
    cdef np.ndarray counts = np.bincount(y_set)
    cdef double impurity = 1.0 - np.sum((counts / y_set.size) ** 2)
    return impurity

cdef tuple split(np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.int_t, ndim=1] y, int feature, double threshold):
    cdef np.ndarray mask = X[:, feature] <= threshold
    return (X[mask], y[mask], X[~mask], y[~mask])

cdef class DecisionTree:
    cdef int max_depth
    cdef DecisionNode root

    def __init__(self, int max_depth):
        self.max_depth = max_depth
        self.root = None
    def __reduce__(self):
        # Serialize the root node and the maximum depth
        return (self.__class__, (self.max_depth,), (self.root,))
    def get_root_details(self):
        if self.root is not None:
            return {
                'feature': self.root.feature,
                'threshold': self.root.threshold,
                'depth': self.root.depth,
                'max_depth': self.root.max_depth
            }
        else:
            return "Root is not initialized."
    def __setstate__(self, state):
        # This sets the state from the serialized object; since state is a tuple, unpack it
        self.root, = state
    cpdef fit(self, np.ndarray[np.float64_t, ndim=2] X_train, np.ndarray[np.int_t, ndim=1] y_train):
        cdef int feature
        cdef double threshold
        feature, threshold = getSplit(X_train, y_train)
        self.root = DecisionNode(feature, threshold, depth=0, max_depth=self.max_depth)
        self.root.grow(X_train, y_train)

    cpdef np.ndarray predict(self, np.ndarray[np.float64_t, ndim=2] X_test):
        cdef list y_pred = []
        cdef np.ndarray X
        for X in X_test:
            y_pred.append(self.root.classify(X))
        return np.array(y_pred)

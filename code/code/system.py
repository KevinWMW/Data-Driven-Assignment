"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""
from collections import Counter
from typing import List

import numpy as np
from scipy import linalg
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter

N_DIMENSIONS = 10


def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """Classify a set of feature vectors using a training set.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    k=5
    
    # Euclidean distance using Scipy.
    dist = distance.cdist(test, train, 'euclidean')
    
    # Get the indices of the k nearest neighbours
    k_nearest = np.argsort(dist, axis=1)[:,:k] 

    k_nearest_labels = train_labels[k_nearest]

    # Uses majority voting to get the most common labels.
    majority_labels = np.array([Counter(row).most_common(1)[0][0] for row in k_nearest_labels])
    
    return majority_labels



# The functions below must all be provided in your solution. Think of them
# as an API that it used by the train.py and evaluate.py programs.
# If you don't provide them, then the train.py and evaluate.py programs will not run.
#
# The contents of these functions are up to you but their signatures (i.e., their names,
# list of parameters and return types) must not be changed. The trivial implementations
# below are provided as examples and will produce a result, but the score will be low.


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).
    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    
    eigenvectors_train = np.array(model["eigenvectors"])
    
    pca_data = np.dot(data - np.mean(data), eigenvectors_train)
    reduced_data = pca_data 

    return reduced_data 
    
    # reduced_data = data[:, 0:N_DIMENSIONS]


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Note, the contents of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    model = {}
    model["labels_train"] = labels_train.tolist()
    
    # Get the covariance matrix of the training data
    covariance_matrix = np.cov(fvectors_train, rowvar=0)

    # Calculate the eigenvectors
    N = covariance_matrix.shape[0]
    v = linalg.eigh(covariance_matrix, eigvals=(N - 10, N - 1))
    eigenvectors = v[1]
    
    model["eigenvectors"] = eigenvectors.tolist()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        image = gaussian_filter(image, sigma=1) # Apply Gaussian filter
        fvectors[i, :] = image.reshape(1, n_features)
    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get some data out of the model. It's up to you what you've stored in here
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test)
    
    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    In the dummy code below, we just re-use the simple classify_squares function,
    i.e. we ignore the ordering.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    # Call the classify_squares function.
    labels = classify_squares(fvectors_test, model)

    # Reshape the labels into an array representing the boards.
    labels = labels.reshape(-1, 8, 8)
    
    for i in range(labels.shape[0]):
        # Count the number of kings of each color
        num_white_kings = np.count_nonzero(labels[i] == 'K')
        num_black_kings = np.count_nonzero(labels[i] == 'k')
        num_white_pawns = np.count_nonzero(labels[i] == 'P')
        num_black_pawns = np.count_nonzero(labels[i] == 'p')
        num_white_queens = np.count_nonzero(labels[i] == 'Q')
        num_black_queens = np.count_nonzero(labels[i] == 'q')

        # If a board has more than one king of either color, change one of them.
        if num_white_kings > 1 or num_black_kings > 1:
            print("More than two kings")
            for j in range(labels.shape[1]): 
                for k in range(labels.shape[2]):  
                    if labels[i][j][k] == 'K':
                        labels[i][j][k] = 'Q'
                        break
                        
                    if labels[i][j][k] == 'k':
                        labels[i][j][k] = 'q'
                        break
            
        if num_white_pawns > 8 or num_black_pawns > 8:
            print("More than 8 pawns")
            
        if num_white_queens > 9 or num_black_queens > 9:
            print("More than 9 queens")
            
            
    labels = labels.reshape(-1,)
    
    return labels



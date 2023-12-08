"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""
from typing import List

import numpy as np
from scipy import linalg

N_DIMENSIONS = 10

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """Classify a set of feature vectors using a training set.

    This dummy implementation simply returns the empty square label ('.')
    for every input feature vector in the test data.

    Note, this produces a surprisingly high score because most squares are empty.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    '''
    predictions = []
    for test_vector in test:
        distances = np.sqrt(np.sum((euclidean_distance(train[:1600], test))**2, axis=0))
        nearest_neighbours = np.argsort(distances)[:k]
        most_frequent_label = np.bincount(train_labels[nearest_neighbours]).argmax()
        predictions = np.append(predictions, most_frequent_label)
    return predictions
    '''
    k=10
    
    # Super compact implementation of nearest neighbour
    x = np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test * test, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())
    
    # cosine distance
    nearest = np.argmax(dist, axis=1) #[:k]
    mdist = np.max(dist, axis=1)
    
    # Get the closest k
    k_indices = np.argsort(dist)[:k]
    print("k indices shape : ", k_indices.shape)
    print("k indices : ", k_indices)
    k_nearest_labels = train_labels[k_indices]
    

    label = train_labels[nearest]
    print("label shape :",label.shape)
    print("Label (nearest neighbour no k) : ", label)
    print("k nearest labels : ", k_nearest_labels)
    
    k_nearest_labels = k_nearest_labels.transpose()
    print("k nearest labels shape : ", k_nearest_labels.shape)
    # Attempted knn implementation
    most_freq_labels = np.unique(k_nearest_labels)
    # print(most_freq_labels)
    # print(len(most_freq_labels))
    # print(most_freq_labels[1].shape)
    # print(most_freq_labels[0])
    # print(most_freq_labels[1])
    print("This is [0] shape ", most_freq_labels[0].shape)
    return k_nearest_labels


    # # Find the k nearest neighbors
    # nearest = np.argpartition(dist, k, axis=0)[:k]

    # # Get the labels of the k nearest neighbors
    # nearest_labels = train_labels[nearest]

    # # Use majority voting to decide the label
    # labels = stats.mode(nearest_labels, axis=0)
    # labels = labels.ravel()

    # return labels.tolist()
    
    
    # n_images = test.shape[0]
    # return ["."] * n_images


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
    
    nmax = 8  # number of pca components
    sample_index = 5


    
    # Calculate the covariance matrix
    covariance_matrix = np.cov(data, rowvar=0)

    # Calculate the eigenvalues and eigenvectors
    N = covariance_matrix.shape[0]
    # print(covariance_matrix.shape)
    
    # N = covariance_matrix.shape[0]
    # eigenvectors = linalg.eigh(covariance_matrix, eigvals=(N - 10, N - 1))
    v = linalg.eigh(covariance_matrix, eigvals=(N - 10, N - 1))
    eigenvalues = v[0]
    eigenvectors = v[1]
    # print(np.shape(eigenvectors))
    # print(np.shape(eigenvalues))
    # print("before shape", np.shape(data))
    # # Projecting the data onto the principal components axis.
    pca_data = np.dot((data - np.mean(data)), eigenvectors)
    # print("pca shape : ", np.shape(pca_data))
    # reconstructed = np.dot(pcatrain_data, v.transpose()) + np.mean(train_data) #reconstructing the data back from 10 to 2500 dimensions, I don't see the need for it.
    reduced_data = pca_data 

    # eigenvectors = np.fliplr(eigenvectors)
    # Step 4: Sort the eigenvectors by decreasing eigenvalues
    # and choose the first k eigenvectors
    # W = Vt.T[:, :model[10]]
    
    data_mean = np.mean(data, axis=0)

    # data_list = data.tolist()
    # for n in range(1, nmax):
    #     npca = n * 2
    #     reconstructed = (
    #         np.dot(
    #             np.dot(data - data_mean, eigenvectors[:, 0:npca]), eigenvectors[:, 0:npca].transpose()
    #         )
    #         + data_mean
    #     )
    #     data_list.append(reconstructed)
    
    # N=6
    # reconstructed = (
    #     np.dot(
    #         np.dot(data[0, :] - data_mean, eigenvectors[:, 0 : N - 1]),
    #         eigenvectors[:, 0 : N - 1].transpose(),
    #     )
    #     + data_mean
    # )
    # print(reconstructed.shape)
    # print(reconstructed)
    # print(np.ndim(reconstructed))
    # reduced_data = data[:, 0:N_DIMENSIONS]
    # print(len(data_list))
    return reduced_data 
    
    # pca_data = np.dot((data - np.mean(data)), v)
    # N = 6 # project the images from the data set into N-dimensional PCA space.
    # nmax = 8 # number of pca components
    # reconstructed = (
    #     np.dot(
    #         np.dot(data[0, :] - mean_train, v[:, 0 : N - 1]),
    #         v[:, 0 : N - 1].transpose(),
    #     )
    #     + mean_train
    # )
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

    # The design of this is entirely up to you.
    # Note, if you are using an instance based approach, e.g. a nearest neighbour,
    # then the model will need to store the dimensionally-reduced training data and labels.
    model = {}
    model["labels_train"] = labels_train.tolist()
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

    return classify_squares(fvectors_test, model)

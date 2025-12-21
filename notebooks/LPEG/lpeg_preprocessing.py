import numpy as np

def minmax_scaler(x, new_min, new_max):
    """
    Rescales x from [old_min, old_max] to [new_min, new_max]
    
    Parameters
    ----------
    x : float
        Value to be rescaled
    new_min : float
        New minimum value of the rescaled x
    new_max : float
        New maximum value of the rescaled x
    old_min : float
        Old minimum value of x (default 0)
    old_max : float
        Old maximum value of x (default 1)
        
    Returns
    -------
    float
        Rescaled value of x
    """
    old_min = np.min(x)
    old_max = np.max(x)
    #TODO check for edge cases and 0 division
    if old_max == old_min:
        raise ValueError("old_max and old_min cannot be the same")
    x = (new_max - new_min)*(x - old_min)/(old_max-old_min) + new_min
    return x

def add_bias_term(X):
    """
    Adds a bias term (column of 1s) to the input data matrix X.
    
    Parameters
    ----------
    X : numpy.ndarray
        Input data matrix of shape (n_samples, n_features)
        
    Returns
    -------
    numpy.ndarray
        Data matrix with bias term added, shape (n_samples, n_features + 1)
    """
    X = np.hstack((X, np.ones((X.shape[0],1))))    #add 1s to X so we learn b as well
    return X

def remove_bias_term(X):
    """
    Removes the bias term (last column) from the input data matrix X.
    
    Parameters
    ----------
    X : numpy.ndarray
        Input data matrix of shape (n_samples, n_features + 1)
        
    Returns
    -------
    numpy.ndarray
        Data matrix with bias term removed, shape (n_samples, n_features)
    """
    if (X[:,:-1] == 1).all():
        print("Warning: The last column does not appear to be a bias term of all 1s. You might be removing important data.")
    X = X[:,:-1]    #remove last column (bias term)
    return X


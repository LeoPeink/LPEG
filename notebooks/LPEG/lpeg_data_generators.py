import numpy as np
import numpy.linalg as LA

#ANALYTICAL FUNCTIONS

def linear_data_generator(n,dim=1,lower=0,upper=1,w=None, q=0, sigma=0, truth=False):
    """
    Generates either clean or noisy, linearly-generated data.
    Formally returns y=wX + eps, where eps is gaussian noise.

    Parameters
    ----------
    n : int
        Number of points to be generated (num. of rows, entries)
    dim : int
        Dimension of each data sample (num. of columns, features)
    lower : float
        Lower bound for the domain of the data points
    upper : float
        Upper bound for the domain of the data points
    w : float array of dim dim, optional
        Vector of weights of the linear model. If None, defaults to ones
    q : float
        intercept with y axis, (y = mx+q)
    sigma : float
        Standard deviation of the noise eps
    truth : bool
        If True, returns ground truth data without noise
    
    Returns
    -------
    X : ndarray of shape (n, dim)
        Generated input data
    y : ndarray of shape (n,)
        Generated output data
    yt : ndarray of shape (n,), optional
        Ground-truth data, un-noised model (only if truth=True)
    
    Examples
    --------
    Generate simple 1D linear data:
    
    >>> X, y = linear_data_generator(n=50, dim=1, w=[2.0], q=1.0, sigma=0.1)
    >>> print(f"X shape: {X.shape}, y shape: {y.shape}")
    X shape: (50, 1), y shape: (50,)
    
    Generate 2D data with custom weights and noise:
    
    >>> X, y, y_true = linear_data_generator(n=100, dim=2, w=[3, -1], q=0.5, 
    ...                                     sigma=0.2, truth=True)
    >>> print(f"Ground truth weights: [3, -1], intercept: 0.5")
    
    Generate clean data (no noise):
    
    >>> X, y = linear_data_generator(n=30, dim=1, lower=-1, upper=1, 
    ...                            w=[0.5], q=2, sigma=0)
    >>> # y = 0.5*X + 2 (perfect linear relationship)
    
    Visualize the generated data:
    
    >>> import matplotlib.pyplot as plt
    >>> X, y, y_true = linear_data_generator(n=50, dim=1, w=[2], q=1, 
    ...                                     sigma=0.3, truth=True)
    >>> plt.scatter(X, y, alpha=0.6, label='Noisy data')
    >>> plt.plot(X, y_true, 'r-', label='True line')
    >>> plt.legend()
    >>> plt.show()
    """
    if w is None:
        w = np.ones((dim))
    X = np.zeros((n,dim))
    for i in range(n):
        X[i,:] = (np.random.uniform(lower,upper,dim))
    eps = np.random.normal(0,sigma,n)
    y = X @ w + q + eps
    yt = X @ w + q
    if truth:
        return X,y,yt
    else:
        return X,y
    
def polynomial_data_generator(n,deg=1,lower=0,upper=1,w=None, sigma=0, truth=False):
    """
    Generates either clean or noisy, polynomially-generated data.
    Formally returns y = sum(w_i*x^i) + eps, where eps is gaussian noise.
    
    Parameters
    ----------
    n : int
        Number of points to be generated (num. of rows, entries)
    deg : int
        Degree of the polynomial (number of features)
    lower : float
        Lower bound for the domain of the data points
    upper : float
        Upper bound for the domain of the data points
    w : ndarray of shape (deg,), optional
        Vector of weights of the polynomial model. If None, random weights are generated
    sigma : float
        Standard deviation of the noise eps
    truth : bool
        If True, returns ground truth data without noise
        
    Returns
    -------
    x : ndarray of shape (n,)
        Generated input data (1D array of x values)
    y : list of length n
        Generated output data with noise
    y_true : list of length n, optional
        Ground-truth data, un-noised model (only if truth=True)
        
    Examples
    --------
    Generate quadratic data:
    
    >>> x, y = polynomial_data_generator(n=50, deg=3, w=[1, 2, -0.5], sigma=0.1)
    >>> # Generates y = 1 + 2x - 0.5x^2 + noise
    
    Generate cubic polynomial with ground truth:
    
    >>> x, y, y_true = polynomial_data_generator(n=100, deg=4, 
    ...                                         w=[0, 1, 0, -0.1], 
    ...                                         sigma=0.2, truth=True)
    >>> # Generates y = x - 0.1x^3 + noise
    
    Visualize polynomial data:
    
    >>> import matplotlib.pyplot as plt
    >>> x, y, y_true = polynomial_data_generator(n=50, deg=3, w=[1, -2, 1], 
    ...                                         sigma=0.3, truth=True)
    >>> plt.scatter(x, y, alpha=0.6, label='Noisy data')
    >>> plt.plot(x, y_true, 'r-', label='True polynomial')
    >>> plt.legend()
    >>> plt.show()
    """ 
    if w is None:
        w = np.ones((deg))
        w = np.random.rand(deg)
    y = []
    y_true = []
    yc = 0
    x = np.linspace(lower,upper,n)
    eps = np.random.normal(0,sigma,n)
    for i in range(n):
        for d in range(deg):
            yc +=  w[d]*np.pow(x[i],d)
        y.append(yc+eps[i])
        y_true.append(yc)
        yc = 0  # Reset for next iteration
    if truth:
        return x, y, y_true
    return x, y


#RANDOM GENERATIONS

def gaussian_clouds_data_generator(n,sparcity=1,n_classes=2,flip=0,labels=None,means=None,sigmas=None):
    """
    Generate N Gaussian clouds of 2D points for classification tasks.

    Each cloud contains `n` points distributed according to a multivariate normal
    with mean vector `means` and covariance matrix built from `sigmas`.  
    Optionally, a percentage of labels can be randomly flipped to add noise.

    Parameters
    ----------
    n : int
        Number of points per cloud.
    sparcity : float, optional
        Average distance between the centers of the clouds. Default is 1.
    n_classes : int, optional
        Number of Gaussian clouds (and corresponding labels). Default is 2.
    flip : float, optional
        Percentage (0-100) of points whose labels will be randomly flipped. Default is 0.
    labels : list of int, optional
        List of labels to assign to each cloud.  
        If None, defaults to `[-1, 1]` for n_classes=2, or `range(n_classes)` otherwise.
    means : list of float, optional
        List of mean coordinates for each cloud (length 2*n_classes).  
        If None, random centers are generated within the range `[0, sparcity]`.
    sigmas : array-like of shape (n_classes,), optional
        Standard deviations for each cloud's covariance matrix.
        If None, uses identity covariance matrices.

    Returns
    -------
    d : ndarray of shape (n_classes * n, 3)
        Array containing generated points and labels.  
        Columns are `[x, y, label]`.

    Examples
    --------
    Generate simple binary classification data:
    
    >>> data = gaussian_clouds_data_generator(n=100, n_classes=2, sparcity=5)
    >>> print(f"Data shape: {data.shape}")
    >>> # 200 points total (100 per class), 3 columns [x, y, label]
    
    Generate multi-class data with custom parameters:
    
    >>> data = gaussian_clouds_data_generator(n=50, n_classes=3, sparcity=10, 
    ...                                      flip=5, sigmas=[1, 2, 0.5])
    >>> # 150 points total, 5% label noise, different cluster sizes
    
    Generate overlapping clusters:
    
    >>> data = gaussian_clouds_data_generator(n=80, n_classes=2, 
    ...                                      means=[0, 0, 1, 1], 
    ...                                      sigmas=[2, 2])
    >>> # Two overlapping clusters centered at (0,0) and (1,1)
    
    Visualize the generated data:
    
    >>> import matplotlib.pyplot as plt
    >>> data = gaussian_clouds_data_generator(n=100, n_classes=3, sparcity=8)
    >>> plt.scatter(data[:,0], data[:,1], c=data[:,2], cmap='viridis')
    >>> plt.title('Gaussian Clouds Classification Data')
    >>> plt.colorbar()
    >>> plt.show()
    """
    
    if means is None:
        means = np.random.randint(0,sparcity,2*n_classes)
    if sigmas is None:
        sigmas = np.ones(n_classes)
    if labels is None:
        labels = np.arange(n_classes)
        if n_classes == 2:
            labels = [-1,1]
    
    d = np.zeros((n_classes*n, 3))  # x, y, label

    for i in range(n_classes):
        cov = np.eye(2) * np.square(sigmas[i])
        points = np.random.multivariate_normal([means[2*i], means[2*i+1]], cov, n)
        d[i*n : n*(i+1), :] = np.hstack([points, np.full((n,1), labels[i])])
        
    for i in range(n_classes*n):
        coin_flip = np.random.randint(0,100)
        if coin_flip < flip:
            d[i][2] = labels[coin_flip % n_classes]
    return d


def uniform_clouds_data_generator(n, n_classes=1, n_dimension=2, limits=None, labels=None, labels_prob=None):
    """
    Generate n points uniformly distributed in a box defined by limits.
    Each point is assigned a random class label according to given probabilities.

    Parameters
    ----------
    n : int
        Total number of points to generate
    n_classes : int, optional
        Number of classes. Default is 1.
    n_dimension : int, optional
        Dimensionality of the feature space. Default is 2.
    limits : ndarray of shape (n_dimension, 2), optional
        Upper and lower limits for each dimension.
        If None, defaults to [-1, 1] for all dimensions.
    labels : array-like of length n_classes, optional
        Names/values for each class. If None, uses integers 0 to n_classes-1.
    labels_prob : array-like of length n_classes, optional
        Probability of each class. If None, uses uniform distribution.

    Returns
    -------
    dataset : ndarray of shape (n, n_dimension+1)
        Generated dataset with features and labels.
        Last column contains the class labels.

    Examples
    --------
    Generate 2D binary classification data:
    
    >>> data = uniform_clouds_data_generator(n=200, n_classes=2, 
    ...                                     limits=[[-2, 2], [-1, 3]])
    >>> print(f"Data shape: {data.shape}")
    >>> # 200 points in 2D space with binary labels
    
    Generate 3D multi-class data with custom probabilities:
    
    >>> data = uniform_clouds_data_generator(n=300, n_classes=3, n_dimension=3,
    ...                                     labels=['A', 'B', 'C'],
    ...                                     labels_prob=[0.5, 0.3, 0.2])
    >>> # 300 points in 3D, with class 'A' appearing 50% of the time
    
    Generate high-dimensional data:
    
    >>> data = uniform_clouds_data_generator(n=100, n_dimension=5, 
    ...                                     limits=[[-1, 1]] * 5)
    >>> # 100 points in 5D hypercube [-1,1]^5
    
    Visualize 2D uniform data:
    
    >>> import matplotlib.pyplot as plt
    >>> data = uniform_clouds_data_generator(n=500, n_classes=4, 
    ...                                     limits=[[-3, 3], [-2, 4]])
    >>> plt.scatter(data[:,0], data[:,1], c=data[:,2], alpha=0.6)
    >>> plt.title('Uniform Random Classification Data')
    >>> plt.xlabel('Feature 1')
    >>> plt.ylabel('Feature 2')
    >>> plt.colorbar()
    >>> plt.show()
    """
    if limits is None:
        limits = np.array([[-1,1]]*n_dimension)
    if limits.shape != (n_dimension,2):
        raise ValueError("limits must be of shape (n_dimension, 2)")
    if labels is None or len(labels) != n_classes:
        labels = np.arange(n_classes)
    if labels_prob is None:
        labels_prob = np.array([1/n_classes]*n_classes)

    dataset = np.empty((n, n_dimension+1))
    dataset[:, :n_dimension] = (np.random.rand(n, n_dimension)-0.5)*abs(limits[:,0]-limits[:,1])+(limits[:,0]+limits[:,1])/2
    dataset[:, n_dimension] = np.random.choice(labels, size=(1, n), p=labels_prob)
    return dataset


#TODO FIX THIS NAME
def partialBallCreate(w):
    """
    create a point for each dimension of W with distance 1 from initial point

    Parameters:
        w : array-like
    a point

    Returns:
        ws : list of array-like
    """
    ws = [w]
    for i in range(len(w)):
        new_w = w.copy()
        new_w[i] += 1
        ws.append(new_w)
    return ws


def dataset_relabler(sign_function, dataset, label1 = 1, label2 = 0, mislabeling_prob = 0):
    """
    given a dataset and an appropriate function, return a numpy.array with the new lables
    
    Parameters

    ----------
    
    sign_function : function
        a function that operate over a dataset and return a one-dimensional array of True/False
    dataset : np.array()
        the dataset that we'll use to generate the new labels
    label1
        label of points that makes the function true
    label2
        label of points that makes the function false
    mislabeling_prob : int
        probability of mislabeling
    
    Returns
    
    -------
    
    a one dimensional array with the new classification of the points

    """
    
    return np.where( (sign_function(dataset)) ^ (np.random.rand(len(dataset[:,0]))<mislabeling_prob), label1, label2 )

def sine_sign_relabler(period = 2*np.pi, amplitude = 1, quota = 0, slope = 0):
    """
    Return a function that classify the data if it's over (True) or under (False) the sinusoid
    defined by period and amplitude. Quota and slope defines a line that is added to the sinusoid

    Parameters

    ----------
    period : float
        distance from two peak
    amplitude : float
        half distance from local max and min
    slope : float
        pendence of the line
    intercept : float
        quota of the line
    
    Returns
    
    -------
    
    line : function
        A function that return a list of true/false. Will be used for classification in datasetRelable
    
    """
    def sinusoid(data):
        return np.sin(data[:,0]/(period/(np.pi*2)))*amplitude + quota + slope * data[:,0] - data[:,1]<0
    return sinusoid

def cosine_sign_relabler(period = 2*np.pi, amplitude = 1, quota = 0, slope = 0):
    def sinusoid(data):
            return (np.sin(data[:,0]/(period/(np.pi*2)))*amplitude + quota + slope * data[:,0] - data[:,1])-period<0
    return sinusoid

def linear_sign_relabler(slope, intercept):
    """
    return a function that classify the data if it's over (True) or under (False) the line defined by slope and intercept

    Parameters

    ----------

    slope : float
        pendence of the line
    intercept : float
        quota of the line
    
    Returns
    
    -------
    
    line : function
        A function that return a list of true/false. Will be used for classification in datasetRelable
    
    """

    def line(data):
        return data[:,0]*slope+intercept-data[:,1]<0
    return line

def exponential_sign_relabler(exponent):
    """
    return a function that classify the data if it's over (True) or under (False) the exponential curve defined by exponent

    Parameters

    ----------

    exponent : float
        exponent of the exponential curve
    
    Returns
    
    -------
    
    exp_curve : function
        A function that return a list of true/false. Will be used for classification in datasetRelable
    
    """

    def exp_curve(data):
        return data[:,1]-np.exp(1)**(data[:,0]/exponent)<0
    return exp_curve
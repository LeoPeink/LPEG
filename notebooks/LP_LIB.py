import numpy as np


def rescale(x,new_min,new_max,old_min=0,old_max=1):
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
    #TODO check for edge cases and 0 division
    if old_max == old_min:
        raise ValueError("old_max and old_min cannot be the same")
    for i in range(len(x)):
        return (new_max - new_min)*(x[i] - old_min)/(old_max-old_min) + new_min

def multiDimLinDataGen(n_points, n_dimension = 1, limits = None, coeficients = None, ofsets = None, sigma = 0, truth=False):
    """
    limits : array[[int,int]]
        upper and lower limit for each dimension of the dataset

    """
    if not hasattr(limits, "shape"):
        limits = np.array([limits]*n_dimension)
    if limits is None:
        limits = np.array([[-1,1]]*n_dimension)
    if limits.shape != (n_dimension,2):
        raise ValueError("limits must be of shape (n_dimension, 2)")
    
    if coeficients is None:
        coeficients = np.ones(n_dimension)
    if coeficients.shape[0] < n_dimension:
        coeficients = np.append(coeficients, np.zeros(n_dimension-coeficients.shape[0]))
    
    if ofsets is None:
        ofsets = np.zeros(n_dimension)
    if ofsets.shape[0] < n_dimension:
        ofsets = np.append(ofsets, np.zeros(n_dimension-ofsets.shape[0]))
    
    dataset = np.empty((n_points, n_dimension))
    for i in range(n_points):
        dataset[i] = coeficients @ np.random.uniform(limits[0], limits[1]) + ofsets + np.random.normal(0, sigma, n_dimension)
    # TODO rivedre questa parte, not happy
    return dataset
    
def linDataGen(n,dim=1,lower=0,upper=1,w=None, q = 0, sigma=0, truth=False):
    """
    Generates either clean or noisy, linearly-generated data.
    Formally returns y,X where y=wX + eps, where eps is gaussian noise.

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
    w : float array of dim dim
        Vector of weights of the linear model
    q : float
        intercept with y axis, (y = mx+q)
    sigma : float
        Standard deviation of the noise eps
    
    Returns
    ----------
    X : array
        Generated input data
    y : array
        Generated output data
    (optional - if truth=True)
    yt : array
        Ground-truth data, un-noised model
    """
    if w is None:
        w = np.ones((dim,1))
    X = np.zeros((n,dim)) #creates "n" points, empty
    for i in range(n): #for each point
        X[i,:] = (np.random.uniform(lower,upper,dim))
    #X = np.random.rand(n,dim)
    eps = np.random.normal(0,sigma,n)
    #print(eps)
    y = X @ w + q + eps
    yt = X @ w + q
    #y = np.dot(X,w)+q+eps
    yt = np.dot(X,w)+q
    if truth:
        return X,y,yt
    else:
        return X,y #TODO questo deve tornare un dataset normale, non due array distinti
"""
#DEMO linDataGen
import numpy as np
import Z_LIB as zl
import matplotlib.pyplot as plt
dim = 1 #number of features
n = 50  #number of data points
l = 0 #lower bound
u = 1 #upper bound
w = np.ones(dim) #weights (linear coefficients)
#w = np.random.normal(0, 1, dim) #random weight (one coefficient per each feature)
#print(w)
X,y = linDataGen()
print(X)
print("w:")
print(w)
plt.scatter(X,y)
plt.show()
"""

def gaussianDataGen(n,sparcity=1,classes=2,flip=0,labels=None,means=None,sigmas=None): #todo sistema sigmas
    """
    Generate N Gaussian clouds of 2D points.

    Each cloud contains `n` points distributed according to a multivariate normal
    with mean vector `means` and covariance matrix `sigmas`.  
    Optionally, a percentage of labels can be randomly flipped.

    Parameters
    ----------
    n : int
        Number of points per cloud.
    sparcity : float, optional
        Average distance between the centers of the clouds. Default is 1.
    classes : int, optional
        Number of Gaussian clouds (and corresponding labels). Default is 2.
    flip : float, optional
        Percentage (0–100) of points whose labels will be randomly flipped. Default is 0.
    labels : list of int, optional
        List of labels to assign to each cloud.  
        If None, defaults to `[-1, 1]` for classes=2, or `range(classes)` otherwise.
    means : list of float, optional
        List of mean coordinates for each cloud (length 2N).  
        If None, random centers are generated within the range `[0, sparcity]`.
    sigmas : array-like of shape (classes), optional
        Will be used to build the covariance matrix (or list of matrices) for each cloud. Each element
        represents the covariance inside individual clouds.  
        If None, uses the 2×2 identity matrix `np.eye(2)`.

    Returns
    -------
    d : ndarray of shape (classes * n, 3)
        Array containing generated points and labels.  
        Columns are `[x, y, label]`.

    Examples
    --------
    >>> d = gCloudDataGen(n=100, classes=3, sparcity=5, flip=10)
    >>> d.shape
    (300, 3)

    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(d[:,0], d[:,1], c=d[:,2])
    >>> plt.show()
    """
    
    if means is None:
        #means = np.arange(2*N)
        means = np.random.randint(0,sparcity,2*classes)
    if sigmas is None: #TODO sistema
        sigmas=np.eye((2))
    if labels is None:
        labels = np.arange(classes)
        if classes ==2:
            labels = [-1,1]
    
    d = np.zeros((classes*n, 3))  # x, y, label

    for i in range(classes):  #for each cloud (= label)
        cov = (np.eye(2)*np.square(sigmas[i]))
        points = np.random.multivariate_normal([means[i],means[i+1]],cov,n)
        d[i*n : n*(i+1) , : ] = np.hstack([points, np.full((n,1),labels[i])])
        
    for i in range(classes*n):
        monetina = np.random.randint(0,100) #lancia monetina (numero da 0 a 100)
        if monetina < flip:                 #se esce testa (cioe' se il numero e' minore della percentuale di flip)     
            d[i][2] = labels[(monetina)%classes]  #cambia etichetta
    return d
'''
#DEMO gCloudDataGen
import Z_LIB.py as zl
d = zl.gCloudDataGen(n=50, classes=2, sparcity=10, flip=0, sigmas=[1,3], labels=[-1,1], means=[0,0,5,5])
#print(d)
# plotting the generated dataset
fig, ax = plt.subplots()
ax.scatter(d[:,0], d[:,1], c=d[:,2])
ax.set_title('Data')
plt.show()
'''

def uniformDataGen(n_points, n_classes = 1, n_dimension = 2, limits = None, labels = None, labels_prob = None):
    """
    generate n_points uniformli distributed in the box delimitate by limits.
    each point have a random class

    Parameters

    ----------
    
    n_points : int
        Number of points per cloud
    n_classes : int
        Number of classes
    n_dimension : int
        How many dimension will have the dataset
    limits : array[[int,int]]
        upper and lower limit for each dimension of the dataset
    labels : array
        names of each classes
    labels_prob = None
        probability of each classes

    Returns
    
    -------
    
    d : array of shape (n_points, n_dimension+1)
        one point each row with lable in the last column
    """
    if limits is None:
        limits = np.array([[-1,1]]*n_dimension)
    if limits.shape != (n_dimension,2):
        raise ValueError("limits must be of shape (n_dimension, 2)")
    if labels is None or labels.shape[1] != n_classes:
        labels = np.arange(n_classes)
    if labels_prob is None:
        labels_prob = np.array([1/n_classes]*n_classes)

    dataset = np.empty((n_points, n_dimension+1))
    dataset[:, :n_dimension] = (np.random.rand(n_points, n_dimension)-0.5)*abs(limits[:,0]-limits[:,1])+(limits[:,0]+limits[:,1])/2
    dataset[:, n_dimension] = np.random.choice(labels, size=(1, n_points), p=labels_prob)
    return dataset
    
# just to remember data[:,2] = np.where( (data[:,0]*m+q-data[:,1]<0) ^ (np.random.rand(len(data[:,1]))<0.05), -1, 1 )

def sinusoidalForRelableBidimensional(period = 2*np.pi, amplitude = 1, quota = 0, slope = 0):
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

def lineForRelableBidimensional(slope, intercept):
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

def datasetRelable(function, dataset, label1 = 1, label2 = 0, mislabeling_prob = 0):
    """
    given a dataset and an appropriate function, return a numpy.array with the new lables
    
    Parameters

    ----------
    
    function : function
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
    
    return np.where( (function(dataset)) ^ (np.random.rand(len(dataset[:,0]))<mislabeling_prob), label1, label2 )

""" DEMO

data = zl.gCloudDataGen(n_points=500, sparcity=1, classes=3, labels=[1,1,1])
myfun = zl.lineForRelableBidimensional(m,q)
dataset[:,2] = zl.datasetRelable(myfun, dataset, mislabeling_prob=0.1)
"""

def linearRegression(x_train, y_train):
    """
    Implements linear regression using the closed-form solution.
    Parameters
    ----------
    x_train : array
        Training input data
    y_train : array
        Training output data
    
    Returns
        ----------
    w : array
        Estimated weights of the linear model
    """
    from numpy.linalg import inv
    w = inv(x_train.T @ x_train) @ x_train.T @ y_train
    return w


def ridgeRegression(x_train, y_train, lam):
    """
    Implements ridge regression using the closed-form solution.
    Parameters
    ----------
    x_train : array
        Training input data
    y_train : array
        Training output data
    lam : float
        Hyperparameter
    
    Returns
        ----------
    w : array
        Estimated weights of the linear model
    """
    import numpy as np
    import numpy.linalg as la
    from numpy.linalg import inv

    try:
        d = np.shape(x_train)[1]
    except IndexError:
        d = 1
    w = (inv(x_train.T@x_train+lam*np.eye(d))@x_train.T)@y_train
    
    return w



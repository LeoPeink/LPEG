import numpy as np
import numpy.linalg as LA

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
    if hasattr(limits, "shape") & limits.shape[0] == 2:
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
    
def linDataGen(n,dim=1,lower=0,upper=1,w=None, q=0, sigma=0, truth=False):
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
        #w = np.ones((dim,1))
        w = np.ones((dim))
    X = np.zeros((n,dim)) #creates "n" points, empty
    for i in range(n): #for each point
        X[i,:] = (np.random.uniform(lower,upper,dim))
    #X = np.random.rand(n,dim)
    eps = np.random.normal(0,sigma,n)
    #print(eps)
    y = X @ w + q + eps
    yt = X @ w + q
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
    #TODO SEPARATE IN FUNCTION TO ADD 1s as last column of X to learn b (intercept) as well

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

def squaredLoss(X,y,w):
    """
    Calcualtes squared loss for linear model with weights w on data X with targets y.
    Parameters
    ----------
    X : 2dimensional np.array
        Input data
    y : array
        Output data
    w : np.array
        Weights of the linear model
    Returns
    -------
    float
        Squared loss value 
    """
    return (np.linalg.norm((y-X@w),2)**2)/len(X)

def squaredLossGradient(X : np.ndarray, y : np.ndarray, w : np.array):
    """
    Calculates the gradient of the squared loss for linear model with weights w on data X with targets y.
    Parameters
    ----------
    X : np.array
        Input data
    y : np.array
        Output data
    w : array-like
        Weights of the linear model
    Returns
    -------
    np.array
        Gradient of the squared loss
    """
    #TODO fix overflow
    #print((X@w-y))
    return (2/len(X))*(X@w-y)@(X)

def polySquaredLoss(X, y, w):
    """
    Calculates squared loss for polynomial regression.
    Compatible with gradientDescent function interface.
    
    Parameters
    ----------
    X : array
        Input data (1D array of x values)
    y : array
        True output values
    w : array
        Polynomial coefficients [highest degree first, ..., constant term]
    
    Returns
    -------
    float
        Squared loss value
    """
    yp = np.polyval(w, X)  # calculate the polynomial (use the model on the data)
    loss = 0
    for i in range(len(X)):
        loss += (yp[i] - y[i])**2
        
    return loss/len(X)

def polySquaredLossGradient(X, y, w):
    """
    Calculates the gradient of the squared loss for polynomial regression.
    Compatible with gradientDescent function interface.
    
    For a polynomial model y = w[0]*x^(n-1) + w[1]*x^(n-2) + ... + w[n-1],
    this function computes the gradient of the mean squared error with respect 
    to the polynomial coefficients.
    
    Parameters
    ----------
    X : array
        Input data (1D array of x values)
    y : array
        True output values
    w : array
        Polynomial coefficients [highest degree first, ..., constant term]
    
    Returns
    -------
    array
        Gradient of the squared loss with respect to polynomial coefficients
    """
    n = len(X)              #dimension of input array
    deg = len(w) - 1        #maximum degree of regression polynomial
    y_pred = np.polyval(w, X)      #calculate polynomial predictions
    residuals = y_pred - y          #calculate residuals
    gradient = np.zeros_like(w)     #initialize gradient array, 1 element per polynomial coefficient
    
    # Build feature matrix for vectorized computation
    X_powers = np.column_stack([X**power for power in range(deg, -1, -1)]) #RIGA PAZZESCA COPILOT DRAGO
    # Vectorized gradient computation: gradient = (2/n) * X^T * residuals
    gradient = (2/n) * X_powers.T @ residuals
    return gradient

def gradientDescent(gradientFunction,lossFunction,X,y,w_0=None, alpha=0.1, t_max=1000, tol=1e-15, fixed_alpha=True):
    if w_0 is None:
        w_0 = np.ones(X.shape[1]) #if starting weights arent specified, generate 0s for every feature. #TODO fix
        
    w = w_0
    ws = []
    losses = []
    t0 = alpha
    for t in range(t_max):                  #stopper at t_max, maximum iteractions TODO fix
        if not fixed_alpha:
            alpha = t0/(t+1)                    #optional: update alpha each step, as per Cornell's Best Practices
        s = -alpha*gradientFunction(X,y,w)      #evaluate stepsize using learning rate (alpha) and the given gradient function
        w = w + s                               #update model weights
        ws.append(w.copy())                     #add to output (copy to avoid reference issues)
        #print('weights at iteration ',t)
        #print(w)
        losses.append(lossFunction(X,y,w))        #add to output
        #print(s)
        if(np.linalg.norm(s,2) < tol):      #if stepsize is smaller than tol, stop
            print("Converged in %d iterations at tol=%g" % (t, tol))
            return ws, losses                       
    print("Max iterations reached: %d" % t_max)
    return ws, losses

def adaGraD(gradientFunction,lossFunction,X,y,w_0=None, alpha=0.1, t_max=1000, tol=1e-15, fixed_alpha=True):
    if w_0 is None:
        w_0 = np.ones(X.shape[1]) #if starting weights arent specified, generate 0s for every feature. #TODO fix
    w = w_0
    ws = []
    regularization = np.zeros_like(w, dtype=np.float64)
    losses = []
    alpha = 1
    for t in range(t_max):                  #stopper at t_max, maximum iteractions TODO fix
        g = gradientFunction(X,y,w)      #evaluate stepsize using learning rate (alpha) and the given gradient function
        regularization += g*g
        s = -alpha*g/(np.sqrt(regularization + 0.1))
        w = w + s                               #update model weights
        ws.append(w.copy())                     #add to output (copy to avoid reference issues)
        #print('weights at iteration ',t)
        #print(w)
        losses.append(lossFunction(X,y,w))        #add to output
        #print(s)
        if(np.linalg.norm(s,2) < tol):      #if stepsize is smaller than tol, stop
            print("Converged in %d iterations at tol=%g" % (t, tol))
            return ws, losses                       
    print("Max iterations reached: %d" % t_max)
    return ws, losses

def polyDataGen(n,deg=1,lower=0,upper=1,w=None, q=0, sigma=0, truth=False):
    #NB: deg is the number of "dimensions" of the polynomial, every point has deg features.
    """
    Generates either clean or noisy, polynomially-generated data.
    Formally returns y,X where y= sum(w_i*x^i) + eps, where eps is gaussian noise.
    Parameters
    ----------
    n : int
        Number of points to be generated (num. of rows, entries)
    deg : int
        Degree of the polynomial
    lower : float
        Lower bound for the domain of the data points
    upper : float
        Upper bound for the domain of the data points
    w : float array of dim deg
        Vector of weights of the polynomial model
    q : float
        intercept with y axis, (y = mx+q)
    sigma : float
        Standard deviation of the noise eps
    Returns
    ----------
    x : array
        Generated input data
    y : array
        Generated output data
    (optional - if truth=True)
    yt : array
        Ground-truth data, un-noised model
    """ 
    if w is None:
        w = np.ones((deg)) #set default coefficients as 1s
        w = np.random.rand(deg)
        #TODO random weights
    y = []
    y_true = []
    yc = 0
    #x = np.random.uniform(lower,upper,n) #TODO see if uniform + sort is better than linspace
    #x = sorted(x)
    #alternative to the last 2 lines: 
    x = np.linspace(lower,upper,n)
    #print(x)
    #evaluate poly in every x
    #y = x + x*w + x*w^2 ... + x*w^deg
    eps = np.random.normal(0,sigma,n)
    for i in range(n):              #for each x,
        for d in range(deg):        #compute yc += x*w^d, yc is "y current"
            yc +=  w[d]*np.pow(x[i],d)
        y.append(yc+eps[i])
        y_true.append(yc)
    #print(y)
    if truth:
        return x, y, y_true
    return x, y

def secantMethod(loss, ws, datas, lables):
    """
    
    Secant method for optimization.
    This function approximates the next point in an linear trajectory
    using a secant-like approach in multi-dimensional space.
    
    Parameters
    ----------
    loss : callable
        Loss function that takes (data, labels, weights) and returns a scalar loss value.
    ws : list or array-like
        List of weight vectors used to compute the points in the space.
        The first element (ws[0]) is used as the reference point.
    datas : 2 dimensional np.array
        Input features or training data.
    lables : 1 dimensional np.array
        Target labels or ground truth values.
    
    Returns
    -------
    w_2 : array-like
        The computed secondary weight vector using the secant method approximation.
    Notes
    -----
    - The first weight vector ws[0] is treated as the initial reference point.
    - The direction vector is computed as the sum of differences of each points anche the initial.
    - The step size t_2 is determined by dividing the negative loss at the reference point
      by the cumulative direction vector.
    
    The theory behind:
    In iperspace the function of a iperline passing throw two points is T*(p1 - p2) + p1
    I add the loss of each point to create the points to work with,
    Than i create a line in the direction of the vector determinated by the sum
    of the difference from each point and the initial point. 
    Than intersect the line and the iperplane where lives the data without the loss by setting the loss to 0
    and compute the T value.
    Than evaluate the rimaning value that is the coordinates of the returning point
    """
    
    # prende in input un array di punti, assumo il primo come punto iniziale 
    direction_vector = sum([loss(datas, lables, ws[i]) - loss(datas, lables, ws[0]) for i in range(len(ws))])
    t_2 = -loss(datas, lables, ws[0])/direction_vector
    w_2 = t_2 * sum([ ws[i] - ws[0] for i in range(len(ws))]) + ws[0]
    return w_2

def GDSecantMethod(loss ,X ,y , ws, t_max, tol=1e-15):
    """
    Iteratively generate weight vectors using a secant-method update and collect them in a list.

    This function performs up to t_max iterations.
    Parameters
    ----------
    loss : callable
        A loss (or objective) function passed through to secantMethod. The exact expected
        signature is determined by the implementation of secantMethod, but it should accept
        the loss argument as the first parameter.
    X : 2 dimensional np.array
        Feature matrix / input data passed to secantMethod.
    y : 1 dimensional np.array
        Target vector / labels passed to secantMethod.
    ws : list-like
        Initial list (or sequence) of weight vectors. The function works on a shallow copy of
        this list, appending successive iterates to that copy; the original ws provided by the
        caller is not modified. The initial length of ws determines the sliding window size
        (n_dim) used when calling secantMethod.
    t_max : int
        Maximum number of iterations to perform.
    tol : float, optional (default=1e-15)
        Convergence tolerance. If the L2 norm (Euclidean norm) of a newly computed weight
        vector is less than tol, the function prints a convergence message and returns early.

    Returns
    -------
    ws : list
        A list containing the initial weight vectors (copied from the input ws) followed by
        the appended iterates produced during optimization. If convergence is reached before
        t_max iterations, the list returned contains all iterates up to and including the
        converged vector.

    Notes
    -----
    - The function relies on an external function secantMethod; its required call signature
      (in this code) is secantMethod(loss, ws_window, X, y), where ws_window is the most
      recent n_dim elements of the current ws copy.

    """
    ws = ws.copy()
    n_dim = len(ws)
    for i in range(t_max):
        w_next = secantMethod(loss, ws[-n_dim:], X, y)
        ws.append(w_next)
        if(np.linalg.norm(w_next,2) < tol):
            print("Converged in %d iterations" % i)
            return ws
    return ws

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
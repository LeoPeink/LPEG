#LAB 2


#step 1: generate data from y=sin(pi*x)+eps
#   where eps ~ N(0,sigma)


import numpy as np
import LP_LIB as lp
import matplotlib.pyplot as plt

# Import the 'randint' function from the 'random' library for generating random integers
from random import randint

# Import the constant 'PI' from the 'math' library for mathematical calculations
from math import pi as PI

# Import mean_squared_error for evaluating regression models by calculating mean squared error
from sklearn.metrics import mean_squared_error

# Import train_test_split for splitting datasets into training and testing subsets
from sklearn.model_selection import train_test_split




#DATA PARAMETERS
global_seed = 43
np.random.seed(global_seed)  #random seed
n = 1000     #number of points       (observations)
dim = 1     #number of dimensions   (features)
l = 0      #lower bound for x
u = 2       #upper bound for x
sigma = 0.5 #noise variance
degs = 50 #numbers of polynomials to evaluate
X = np.linspace(l,u,n)  #creates x axis to be evaluated
eps = np.random.normal(0,sigma,n)   #creates the actual noise
y = (np.sin(np.pi*X+eps))   #evaluates the function, generating data

#DATA OUTPUT
#plt.scatter(X,y)
#plt.show()

# splitting training and test data using sklearn
X_train, X_test, y_train, y_test = train_test_split(X,y ,random_state=global_seed,test_size=0.25,shuffle=True)
print(X_train)
#plot train,test sets on the same plot

plt.figure(0)


ax = plt.axes()
''''''
ax.scatter(X_train,y_train, label="Training set", c="black", alpha=0.6)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.scatter(X_test,y_test, label="Test set", c="green", alpha=0.6)


MLE_TR = []
MLE_TE = []

#polynomial fitting of degree k - plot the polynomials on a domain bigger than the training set
#to show the behaviour outside the training set
for k in range(1,degs):                       #for each degree
    cp = np.polyfit(X_train,y_train,k)      #find coefficents in training - THIS IS OUR MODEL
    yp_tr = np.polyval(cp,X_train)          #evaluate polynomial on training points 
    yp_te = np.polyval(cp,X_test)           #evaluate polynomial on test points
    xp = np.linspace(l,u,len(X_train))      #plot tech only
    yplot = np.poly1d(cp)(xp)               #plot tech only
    ax.plot(xp,yplot, label="k="+str(k))    #plot tech only
    MLE_TR.append(mean_squared_error(yp_tr,y_train))    #training error
    MLE_TE.append(mean_squared_error(yp_te,y_test))     #test error

#print(p)
ax.legend()


plt.figure(1)
ax = plt.axes()
ax.plot(range(1,degs),MLE_TR,marker='o',label="Training error")
ax.plot(range(1,degs),MLE_TE,marker='*',label="Test error")
ax.set_yscale('log')
ax.legend()

print("TRAINING ERROR:" + str(MLE_TR))
print("TEST ERROR:" + str(MLE_TE))

plt.show()


"""
X,y = lp.linDataGen(50,1,0,1,None,0.1)
print(X)
plt.scatter(X,y)
plt.show()
"""
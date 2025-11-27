import LP_LIB as lp
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


n = 500
dim = 2
l = 0
u = 1
w = [3,2]
q=0
sigma = 0.25
global_seed = 1
np.random.seed(global_seed)

X,y,t_true = lp.linDataGen(n,dim,l,u,w,q,sigma,True) #generate linear data with noise

X0 = np.array(X[:,0])
X1 = np.array(X[:,1])

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d').scatter(X0,X1,y)

#implement linear regression: find a hyperplane that best approximates the data.
w_lr = lp.linearRegression(X,y) #two weights

'''
#plot the linear regression plane
fig = plt.figure() #new figure
ax = fig.add_subplot(projection='3d') #3D axis
#predicted (TODO plot plane)
bx = ax.plot_surface(X0,X1,X@w_lr, c='red') 
ax.scatter(X0,X1,y, c='blue', alpha=0.5) #actual outputs
ax.set_xlabel('X0')
ax.set_ylabel('X1')
ax.set_zlabel('y')
plt.show()
'''

w_0 = [-100,50]
alpha = 0.5

ws_lrgd = lp.squared_loss_gradientDescent(X,y,w_0,alpha,10000)
ws_lrgd = lp.gradientDescent(lp.squared_loss_gradient,X,y,w_0,alpha,10000)

print(w_lr)
print(ws_lrgd[-1])


#plot the weights and the targets
plt.plot(range(len(ws_lrgd)), ws_lrgd)
plt.hlines(w_lr,0,len(ws_lrgd), colors='red', label='Linear Regression Weights')
plt.xlabel('Iterations')
plt.ylabel('Weights')
plt.legend()
plt.show()

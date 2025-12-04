import LP_LIB as lp
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


n = 100
dim = 2
l = 0
u = 1
w = [3,2]
#w = None
q=2
sigma = 0.5
global_seed = 1
np.random.seed(global_seed)

X,y,t_true = lp.linDataGen(n,dim,l,u,w,q,sigma,True) #generate linear data with noise

X0 = np.array(X[:,0])
X1 = np.array(X[:,1])


#WE LEARN B AS WELL - MOD X TO HAVE 1s
X = np.hstack((X, np.ones((X.shape[0],1))))    #add 1s to X so we learn b as well

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d').scatter(X0,X1,y)

#implement linear regression: find a hyperplane that best approximates the data.
w_lr = lp.linearRegression(X,y) #two weights

#w_0 = [-100,50]
w_0 = [1,1,0]
alpha = 0.1
t_max = 2000
tol = 1e-5

print(f"Linear Regression - Coefficients: {w_lr[:-1]}, Intercept: {w_lr[-1]}")

ws_lrgd, losses = lp.adaGraD(lp.squaredLossGradient,lp.squaredLoss,X,y,w_0,alpha,t_max,tol,fixed_alpha=True)

print(f"Gradient Descent - Coefficients: {ws_lrgd[-1][:-1]}, Intercept: {ws_lrgd[-1][-1]}")

#plot losses
plt.figure()
plt.plot(range(len(losses)), losses)
plt.xlabel('Iterations')
plt.ylabel('Squared Loss')
plt.title('Squared Loss vs Iterations')



#plot the weights and the targets
plt.figure()
plt.plot(range(len(ws_lrgd)), ws_lrgd)
plt.hlines(w_lr,0,len(ws_lrgd), colors='red', label='Linear Regression Weights')
plt.xlabel('Iterations')
plt.ylabel('Weights')
plt.legend()

# Create meshgrid for surface plotting
x0_range = np.linspace(X0.min(), X0.max(), 20)
x1_range = np.linspace(X1.min(), X1.max(), 20)
X0_mesh, X1_mesh = np.meshgrid(x0_range, x1_range)

# Create feature matrix for the mesh (include bias term)
X_mesh = np.column_stack([X0_mesh.ravel(), X1_mesh.ravel(), np.ones(X0_mesh.size)])
# Predict outputs for the mesh
y_mesh = X_mesh @ w_lr
y_mesh2 = X_mesh @ ws_lrgd[-1]
# Reshape predictions back to meshgrid shape
y_mesh = y_mesh.reshape(X0_mesh.shape)
y_mesh2 = y_mesh2.reshape(X0_mesh.shape)
# Plot the surface
fig = plt.figure() #new figure
ax = fig.add_subplot(projection='3d') #3D axis
surf1 = ax.plot_surface(X0_mesh, X1_mesh, y_mesh, color='red', alpha=0.5)
surf2 = ax.plot_surface(X0_mesh, X1_mesh, y_mesh2, color='green', alpha=0.5)
ax.scatter(X0, X1, y, c='blue', alpha=0.5) #actual outputs
ax.set_xlabel('X0')
ax.set_ylabel('X1')
ax.set_zlabel('y')
# Add a simple legend using patches (surface collections don't map directly to legend labels)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='red', label='Linear Regression'),
				   Patch(color='green', label='GD final'),
				   Patch(color='blue', label='Data')])
plt.show()

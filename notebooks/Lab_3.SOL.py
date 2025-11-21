import LP_LIB as lp
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



#generate linear (albeit noisy) data for linear regression.

n = 50
dim = 2
l = 0
u = 1
w = [3,2]
q=10
sigma = 0
global_seed = 3

np.random.seed(global_seed)

#create an array for weights
ws = []
lams = []
n_models = 10

MLE_TR = []
MLE_TE = []
#generate data

d = lp.linDataGen(n,dim,l,u,w,q,sigma)
plt.figure()
#3d plot of data
X = d[0]
y = d[1]
#y_t = d[2]
ax = plt.axes(projection='3d')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
ax.scatter(X[:, 0], X[:,1],y, c='red')
plt.title('Generated linear data with noise')
plt.show() 
#evaluate ridge regression for different values of lambda
#split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=global_seed)


#print(d)
fig = plt.figure(0)
#set scale
plt.xlim(l, u)
plt.ylim(min(y), max(y))
#scatter test and training points in different colors
plt.scatter(X_train,y_train, label = 'Training data')
plt.scatter(X_test,y_test, label = 'Testing data')
plt.plot(X,y_t, c='green', label='Ground truth',alpha=1)

#implement linear regression in closed form

#w = (XX^t)^-1 XY^t
#w = inv(X.T @ X) @ X.T @ y
w = lp.linearRegression(X,y)
plt.plot(X, X * w, c='yellow', label='Linear regression')
#xp = np.linspace(l, u, 100)



for i in range(n_models):
    #lam = n_models/(2*i+1) #select lambda values decreasing
    lam = np.linspace(0.01,10,n_models)[i]
    w = lp.ridgeRegression(X_train,y_train,lam) #compute ridge regression weights
    ws.append(w)    #add weight to array
    lams.append(lam)  #add lambda to array
    yp = w * X_train
    #use gradient to plot clearly every different model
    
    ##plt.plot(X_train, yp, c=plt.cm.Greys(i/n_models), alpha=1, label='Ridge Regression lambda=%.2f' % lam)
    MLE_TR.append(mean_squared_error(y_train, yp))    #training error
    
    
#evaluate models on test set
for i in range(n_models):
    yp_test = ws[i] * X_test
    MLE_TE.append(mean_squared_error(y_test, yp_test))    #testing error

print("Estimated w:")
print(w)
plt.legend()


plt.figure()
plt.scatter(lams,ws, label='weight over lambda value')
#plt.xscale('log')
plt.xlabel('lambda')
plt.ylabel('w')
plt.legend()



#print errors
plt.figure()
plt.plot(lams, MLE_TR, marker='o', label='Training error')
plt.plot(lams, MLE_TE, marker='*', label='Testing error')
plt.yscale('log')
plt.xlabel('Lambda index')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

'''
legendFlag = True
def mouse_event(event):
    #print('x: {} and y: {}'.format(event.xdata, event.ydata))
    x_click = event.xdata
    #add point to plot
    y_click = x_click * w
    plt.scatter(x_click, y_click, c='orange', label='linear regression')
    plt.draw()
    global legendFlag
    if legendFlag:
        plt.legend()
        legendFlag = False
#connect the event to the figure
cid = fig.canvas.mpl_connect('button_press_event', mouse_event)







#while clicking on the plot, add a point and predict its output using the linear model
def onclick(event):
    x_new = event.xdata
    y_new = event.ydata
    #predict using linear model
    x_vec = np.array([[x_new]])
    y_pred = x_vec @ w
    print("Predicted output using linear model: %.2f" % y_pred)
    plt.scatter(x_new, y_pred, c='green', label='Predicted point')
    plt.draw()
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.legend()
plt.show()




'''
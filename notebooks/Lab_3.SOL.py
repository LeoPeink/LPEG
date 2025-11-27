import LP_LIB as lp
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


n = 5000
dim = 1
l = 0
u = 1
w = None
q=0
sigma = 1
global_seed = 1
np.random.seed(global_seed)


d = lp.linDataGen(n,dim,l,u,w,q,sigma,True) #generate linear data with noise
X = d[0]    #feature matrix, input
# print(X)
y = d[1]    #output vector, target with noise
# print(y)
y_true = d[2]   #ground truth output vector, target without noise

#create an array for weights (ridge regression)
weights_rr = []
lams = []
n_models = 10

#arrays to store MLE errors (ridge regression)
MLE_TR = []
MLE_TE = []


#split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=global_seed)


plt.scatter(X_train,y_train, label = 'Training data')
plt.scatter(X_test,y_test, label = 'Testing data')

weight_lr = lp.linearRegression(X_train,y_train)                #compute linear regression weights (lr model)


for i in range(n_models):
    #lam = n_models/(2*i+1) #select lambda values decreasing
    lam = np.linspace(0.01,100,n_models)[i]
    w = lp.ridgeRegression(X_train,y_train,lam) #compute ridge regression weights
    weights_rr.append(w)    #add weight to array
    lams.append(lam)  #add lambda to array
    yp = w * X_train
    plt.plot(X_train, yp, c=plt.cm.plasma(i/n_models), alpha = 0.75,label='RR lambda=%.2f' % lam)
    MLE_TR.append(mean_squared_error(y_train, yp))    #training error
    
    
plt.plot(X,y_true, c='green', label='Ground truth')
plt.plot(X_train, X_train @ weight_lr, c='red', label='Linear regression')    
#evaluate models on test set
for i in range(n_models):
    MLE_TE.append(mean_squared_error(y_test, X_test * weights_rr[i]))    #testing error
    
    
    

#print("Estimated w:")
#print(w)
plt.legend()


plt.figure()
plt.scatter(lams,weights_rr, label='weight over lambda value')
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
linerr = mean_squared_error(y_train, weight_lr@X_train)
sqtest = lp.squaredLoss(X_train,y_train,weight_lr)
print(sqtest)
print(linerr)
'''


"""
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
"""





"""
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

"""





import LP_LIB as lp
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from numpy.linalg import inv


#generate linear (albeit noisy) data for linear regression.

n = 100
dim = 1
l = -10
u = 10
w = [10]
q=0
sigma = 0.5

np.random.seed(42)

d = lp.linDataGen(n,dim,l,u,w,q,sigma,True)
X = d[0]
y = d[1]
y_t = d[2]

#print(d)
fig = plt.figure(0)
#set scale
plt.xlim(l, u)
plt.ylim(min(y), max(y))
plt.scatter(X,y, label = 'Noisy data')
plt.plot(X,y_t, c='red', label='Ground truth',alpha=0.5)

#implement linear regression in closed form

#w = (XX^t)^-1 XY^t
w = inv(X.T @ X) @ X.T @ y



for i in range(10):
    lam = 10/(2*i+1)
    w = lp.ridgeRegression(X,y,lam)
    xp = np.linspace(0,1,100)
    yp = w * xp
    plt.plot(xp,yp, label = 'RR [lam='+str(lam)+']')

print("Estimated w:")
print(w)
plt.legend()




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


plt.show()



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



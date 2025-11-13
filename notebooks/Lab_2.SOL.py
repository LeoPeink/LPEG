#LAB 2


#step 1: generate data from y=sin(pi*x)+eps
#   where eps ~ N(0,sigma)


import numpy as np
import LP_LIB as lp
import matplotlib.pyplot as plt

#STEP 1:
#generate uniform, random X array
#generate y = sin(pi*x)

    #(1) ASAIW:
    #generate gaussian, random noise array as eps
    #re-generate y = sin(pi*x)+eps
    
#STEP 2:
#interpolate the points with a polinomial curve.
#plot the results.

    #(2) ASAIW:
    #generate gaussian, random noise array as eps
    #re-generate y = sin(pi*x)+eps

n = 500  #number of points       (observations)
dim = 1 #number of dimensions   (features)
l = 0   #lower bound for x
u = 100 #upper bound for x
sigma = 0.1

#X = np.array(np.random.uniform(l,u,n))
X = np.linspace(-1,1,n)


eps = np.random.normal(0,sigma,n)



y = (np.sin(np.pi*X)+eps)


#print(X)
#print(y)  

plt.scatter(X,y)
plt.show()


"""
X,y = lp.linDataGen(50,1,0,1,None,0.1)
print(X)
plt.scatter(X,y)
plt.show()
"""
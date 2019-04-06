#importing dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

#importing dataset
data = pd.read_csv("ex2data1.txt",header = None,names=['exam1','exam2','admitted'])
#feature values
x = data.iloc[:,:-1]
#target values
y = data.iloc[:,-1]
admitted = data.loc[y == 1]
not_admitted = data.loc[y == 0]
#visualizing the dataset
plt.scatter(admitted.iloc[:,0],admitted.iloc[:,1])
plt.scatter(not_admitted.iloc[:,0],not_admitted.iloc[:,1])
plt.xlabel('exam1 score')
plt.ylabel('exam2 score')
plt.legend('admitted','not_admitted')
plt.show()
#preparing data for model
ones = np.ones([len(x),1])
#thetaones = np.ones([len(x),1])
x = x[:,np.newaxis]
x = np.hstack([ones,x])
y = y[:,np.newaxis]
theta = np.zeros([len(x),1])
x
#defining sigmod fun
def sigmoid(z):
    return 1 /(1 + np.exp(-z))
#weigted sum of inputes
def net_input(theta,x):
    return np.dot(x,theta)
#theta transpose of x
def probability(theta,x):
    return sigmoid(net_input(theta,x))

#defining cost fun
def costfun(x,y,theta):
    m = len(x)
    cost = -(1/m)*np.sum(y*np.log(probability(theta,x))(1-y)*np.log(1-probablity(theta,x)))
    return cost
#defining gradient descent
def gradientdes(x,y,theta):
    return (1/m)*np.dot(x.T,sigmoid(net_input(theta,x))-y)
def fit(x, y, theta):
    opt_weights = opt.fmin_tnc(func=costfun, x0=theta,
                  fprime=gradientdes,args=(x, y.flatten()))
    return opt_weights[0]
parameters = fit(x, y, theta)

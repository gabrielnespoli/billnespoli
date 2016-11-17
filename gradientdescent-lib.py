#Author: Gabriel Mesquita Nespoli
#Student of Masters in Data Science at Università Sapienza di Roma

import numpy as np
import math
import matplotlib.pyplot as plt
import tkinter

def plotGradientDescent(x,y,theta):
	plt.figure()  
	#line = [b a]; y = b + ax
	plt.plot(x[:,1], theta[1]*x[:,1] + theta[0], '-')
	plt.plot(x[:,1], y, 'o')
	
	plt.ylabel('log(price)')
	plt.xlabel('Average Growing Season Temperature')
	plt.title("The Wine Equation")	
	
	plt.savefig('1743585.png')

def normVector(v):
	sum = 0.0
	for i in range(0,len(v)):
		sum += v[i]**2
	return math.sqrt(sum)

def stopEvaluation(theta,thetaOld,eps):
    return not normVector(theta-thetaOld)/normVector(theta) > eps

def grad(y, x, theta):
    return 2.2026486*np.dot(x.T,(np.dot(x, theta) - y))

def descent(y, x, alpha = 0.02235, itr = 1e2, eps = 0.02):
    p = x.shape[1]
    theta = np.zeros(p)
    thetaOld = np.ones(p)
    count = 0
    start = True
    while start == True or (not stopEvaluation(theta,thetaOld,eps) and count < itr):
        start = False
        thetaOld = theta
        theta = thetaOld - alpha*grad(y,x,thetaOld)
        count += 1
    return theta

def r2(y, c, x):
    N = len(y)
    yPred = np.dot(x, c)
    yMean = y.sum()/N
    SSR = 0
    SSTO = 0
    for i in range(0,N):
        SSR = SSR + (yPred[i] - yMean)**2
        SSTO = SSTO + (y[i] - yMean)**2
    r2 = SSR/SSTO
    return r2

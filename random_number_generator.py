import numpy as np
from scipy.stats import norm, poisson
import random as rn
from matplotlib import pyplot as plt

class expon:
    def __init__(self,tau = 1):
        self.tau = tau
        self.k = 1/tau

    def pdf(self,X):
        return self.k*np.exp(-self.k * X)
    def cdf(self,X):
        return 1-np.exp(-self.k*X)
    
    @classmethod
    def epdf(cls,X,tau):
        k = 1/tau
        return k*np.exp(-k * X)
    
    @classmethod
    def ecdf(cls,X,tau):
        k = 1/tau
        return 1-np.exp(-k*X)
    

def rand_range (xMin, xMax) :
    return xMin + rn.random () * (xMax - xMin)

def rand_TAC (f, xMin, xMax, yMax) :
    x = rand_range (xMin, xMax)
    y = rand_range (0, yMax)
    while (y > f (x)) :
        x = rand_range (xMin, xMax)
        y = rand_range (0, yMax)
    return x

def random_exponential_distributed_intervals_rand_TAC(t0,N):
    fix_expon = expon(t0)
    return np.array([rand_TAC(fix_expon.pdf,0,max(X),1) for _ in range(N)])

def random_exponential_distributed_intervals(t0,N):
    Y = np.random.rand(N) # random numbers between 0 and 1 evenly distributed
    return -t0*np.log(1-Y) # inverse CDF

def random_exponential_interval(t0):
    return -t0*np.log(1-np.random.rand())

if __name__ == '__main__':
    X = np.linspace(0,10,100)
    fix = expon(5)
    Y1 = expon.epdf(X,4)
    Y2 = expon.epdf(X,2)
    Y3 = expon.ecdf(X,2)
    Y4 = fix.pdf(X)
    plt.plot(X,Y1,label='Y1')
    plt.plot(X,Y2,label='Y2')
    plt.plot(X,Y3,label='Y3')
    plt.plot(X,Y4,label='Y4')
    plt.legend()
    plt.show()
import numpy as np
import scipy.stats as sc

class RandomLCG:
  def __init__(self,A,C,M,seed=0):
    self.A = A
    self.M = M
    self.C = C
    self.x = seed
    self.seed = seed
  def nextInt(self, min,max):
    self.x = (self.A*self.x+self.C)%self.M
    self.seed = self.x
    return (self.x%(max+1)) + min
  def setSeed(self,seed):
    self.seed = seed
    self.x = seed

def rand_uniform(min,max,size=1):
    if size==1: return np.float64(np.random.rand()*(max-min) + min)
    return np.random.rand(size)*(max-min) + min

# def rand_TAC(pdf,xmin,xmax,size=1):
#     l = np.empty(size,dtype=np.float64)
#     for i in range(size):
#         randX = rand_uniform(xmin,xmax,size=1)
#         while pdf(randX) < rand_uniform(0,1,size=1):
#             randX = rand_uniform(xmin,xmax,size=1)
#         l[i] = randX
#     return l[0] if size == 1 else l

############ ANOTHER
def rand_TAC(pdf,xmin,xmax,ymax,size=1):
    l = np.empty(size,dtype=np.float64)
    for i in range(size):
        randX = rand_uniform(xmin,xmax,size=1)
        while pdf(randX) < rand_uniform(0,ymax,size=1):
            randX = rand_uniform(xmin,xmax,size=1)
        l[i] = randX
    return l[0] if size == 1 else l

# using inverse CDF
def rand_expon(t0,size=1):
  return -t0*np.log(1-rand_uniform(0,1,size=size))

# random poisson distribution using toy esperiment
def rand_poisson(mu, size = 1):
    def toy(mu): #t0 = 1, tM = mu
        N = -1
        while mu>0:
            N += 1
            mu -= rand_expon(1)
        return N
    if size==1: return toy(mu)
    else: return np.array([toy(mu) for _ in range(size)])

def expon_random(tau,N):
    if tau == 0: return np.zeros(N)
    return -tau*np.log(1-np.random.rand(N))

def poisson_stats(X):
    return {'mean':np.mean(X),'var':np.var(X),'skew':sc.stats.skew(X),'kurt':sc.stats.kurtosis(X)}

class Stats:
    def __init__(self,X):
        self.mean = np.mean(X)
        self.var = np.var(X)
        self.std = np.std(X)
        self.skew = sc.skew(X)
        self.kurt = sc.kurtosis(X)
    def __str__(self):
       return f' \
       mean: {self.mean}\n \
       var : {self.var}\n \
       skew: {self.skew}\n \
       kurt: {self.kurt}\n \
       std : {self.std}\n \
       '

def poisson_expected_stats(mu):
    return {'mean':mu,'var':mu,'skew':1/np.sqrt(mu),'kurt':1/mu}

if __name__ == '__main__':
   s = Stats([1,2,3,4,5,6])
   print(s)
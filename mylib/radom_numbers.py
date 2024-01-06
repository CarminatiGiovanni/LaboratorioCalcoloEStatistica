import numpy as np

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

def rand_TAC(pdf,xmin,xmax,size=1):
    l = np.empty(size,dtype=np.float64)
    for i in range(size):
        randX = rand_uniform(xmin,xmax,size=1)
        while pdf(randX) < rand_uniform(0,1,size=1):
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

def poisson_stats(X):
    return {'mean':np.mean(X),'var':np.var(X),'skew':sc.stats.skew(X),'kurt':sc.stats.kurtosis(X)}

def poisson_expected_stats(mu):
    return {'mean':mu,'var':mu,'skew':1/np.sqrt(mu),'kurt':1/mu}
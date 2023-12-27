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
    if size==1: return np.random.rand()*(max-min) + min
    return np.random.rand(size)*(max-min) + min

def rand_TAC(pdf,xmin,xmax,size=1):
    l = np.empty(size,dtype=np.float64)
    for i in range(size):
        randX = rand_uniform(xmin,xmax,size=1)
        while pdf(randX) < rand_uniform(0,1,size=1):
            randX = rand_uniform(xmin,xmax,size=1)
        l[i] = randX
    return l[0] if size == 1 else l
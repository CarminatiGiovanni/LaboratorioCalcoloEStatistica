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
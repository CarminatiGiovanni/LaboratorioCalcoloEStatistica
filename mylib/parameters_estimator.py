import numpy as np
import scipy.stats as sc

def loglikelyhood(f,X,tau):
  return np.sum(np.log(f(X,tau)))

def likelyhood(f,X,tau):
  return np.prod(f(X,tau))

def exponential_pdf(X,tau):
  return sc.expon.pdf(X,scale=tau)
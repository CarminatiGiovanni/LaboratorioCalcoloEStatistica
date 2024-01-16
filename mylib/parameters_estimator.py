import numpy as np
import scipy.stats as sc

def loglikelyhood(f,X,tau):
  return np.sum(np.log(f(X,tau)))

def likelyhood(f,X,tau):
  return np.prod(f(X,tau))

def exponential_pdf(X,tau):
  return sc.expon.pdf(X,scale=tau)

## EXTENDED BINNED NNL
# from iminuit.cost import ExtendedBinnedNLL
# def mod_total (bin_edges, N_signal, mu, sigma, N_background, tau):
#     return N_signal * norm.cdf (bin_edges, mu, sigma) + \
#             N_background * expon.cdf (bin_edges, 0, tau )

# bin_edges = np.linspace(min(data),max(data),sturges(N))

# # data[(data>i)*(data<i + bin_width)]
# # bin_content = np.array([len(list(filter(lambda x: x >= i and x < i + bin_width,data))) for i in bin_edges[:-1]]) # aggiustare
# bin_content = np.array([len(data[(data>i)*(data<=g)]) for i,g in zip(bin_edges[:-1],bin_edges[1:])])

# my_cost_func = ExtendedBinnedNLL (bin_content, bin_edges, mod_total)
# my_minuit = Minuit (my_cost_func,
#                     N_signal = N/2, mu = np.mean(data), sigma = np.std(data), # signal input parameters
#                     N_background = N/2, tau = 5)

# my_minuit.migrad()
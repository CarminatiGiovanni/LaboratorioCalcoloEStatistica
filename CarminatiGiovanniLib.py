import numpy as np
import scipy.stats as sc
import matplotlib.pyplot as plt
from iminuit.cost import ExtendedBinnedNLL,BinnedNLL,UnbinnedNLL,LeastSquares
from iminuit import Minuit


# best number of bins depending on the number of data:

sturges = lambda N: int(np.ceil( 1 + 3.322 * np.log(N)))

# find zero of a function f in the interval [a,b]
def zero(f,a,b,prec=0.0001):
    c = 0.5 * (b + a)
    if b - a < prec: return c
    if f(c) * f(a) > 0: return zero(f,c,b,prec)
    else: return zero(f,a,c,prec)

# finds max of a function (return x) in interval [a,b]
def maximum(f,a,b,prec = 0.0001):
    x1 = a+(1-0.618) * (b-a)
    x2 = a + 0.618 * (b-a)
    if np.abs(b - a) < prec: return x2
    if f(x1) > f(x2): return maximum(f,a,x2,prec)
    else: return maximum(f,x1,b,prec)
    
# finds min of a function (return x) in interval [a,b]
def minimum(f,a,b,prec = 0.0001):
    x1 = a+(1-0.618) * (b-a)
    x2 = a + 0.618 * (b-a)
    if np.abs(b - a) < prec: return x2
    if f(x1) < f(x2): return maximum(f,a,x2,prec)
    else: return maximum(f,x1,b,prec)

# takes h function (L - max(L) + 0.5) and finds intersection with x axis for theta-sigma,theta+sigma
def bisection_for_likelihood(h,a,xmax,b,prec: float=0.0001):
    return zero(h,a,xmax,prec),zero(h,xmax,b,prec)

# integral from [a,b] of a function f
def integral_hit_or_miss_N(f,a,b,N = 1000):
	maxf = maximum(f,a,b)
	randX = a + np.random.rand(N) * (b - a)
	randY = 0 + np.random.rand(N) * (maxf - 0)
	hit = sum(map(lambda x: x[1] < f(x[0]),zip(randX,randY)))
	p = hit/N
	expected_value = p * (b-a) * (maxf - 0)
	variance = ((b-a) * (maxf - 0))**2 * p * (1-p) / N
	return expected_value, np.sqrt(variance)

# integral from [a,b] of a function f (better than hit or miss)
def crude_montecarlo(f,a,b,N=1000000):
    X = a + np.random.rand(N) * (b-a)
    E = np.average(f(X))
    STD = np.std(f(X))
    return (b-a)*E, (b-a)*STD/np.sqrt(N)
    
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
       
       
###################################################################################################################################################################

# random values with uniform distribution in range [a,b]
def rand_uniform(a=0,b=1,size=1):
    if size==1: return np.float64(np.random.rand()*(b-a) + a)
    return np.random.rand(size)*(b-a) + a
    
# generic random Try and Catch algorithm
def rand_TAC(pdf,a,b,params=[],size=1):
	f = lambda x: pdf(x,*params)
	ymax = f(maximum(f,a,b))
	l = np.empty(size,dtype=np.float64)
	count = 0
	while count < size:
		x = rand_uniform(a,b)
		y = rand_uniform(0,ymax)
		if y < f(x):
			l[count] = x
			count += 1
	return l[0] if size == 1 else l

# using inverse CDF
def rand_expon(t0,size=1):
	if t0 == 0: return np.zeros(size)
	return -1/t0*np.log(1-np.random.rand(size))

# random poisson distribution using toy esperiment
def rand_poisson(mu, size = 1):
	def toy(mu): #t0 = 1, tM = mu
		N = -1
		while mu > 0:
			N += 1
			mu -= rand_expon(1)
		return N

	if size==1: return toy(mu)
	else: return np.array([toy(mu) for _ in range(size)])
    
    
def loglikelyhood(f,X,tau):
	return np.sum(np.log(f(X,tau)))

def likelyhood(f,X,tau):
	return np.prod(f(X,tau))
  
######################################################## IMINUIT EXAMPLES #############################################################################################
## EXTENDED BINNED NNL (unknown Nsignal, Nbackground)
def extended_binned_NNL_example():

	def parabola(x,a,b,c):
	    return a*x**2 + b*x + c

	area_para = crude_montecarlo(lambda x: -0.2*x**2+0.8*x+1,-1,5)[0]
	area_gauss = crude_montecarlo(lambda x: 2*sc.norm.pdf(x,loc=2,scale=0.25),-1,5)[0]
	# area_tot = crude_montecarlo(lambda x: 2*norm.pdf(x,loc=2,scale=0.25),-1,5)[0]
	area_tot = area_para + area_gauss

	scale_para = area_para/area_tot
	scale_gauss = area_gauss/area_tot

	# PDF
	pdf_parabola = lambda x: parabola(x,-0.2,0.8,1)/area_para #* theta(x,-1,5) # normalized
	pdf_gauss = lambda x: sc.norm.pdf(x,loc=2,scale=0.25) #* theta(x,-1,5)

	def pdf(x):
		val = scale_para*pdf_parabola(x) + scale_gauss*pdf_gauss(x)
		if type(val) == np.ndarray:
			return np.array([i if i>0 else 0 for i in val])
		return val if val>=0 else 0
	    
	N = 10000
	data = rand_TAC(pdf,-1,5,size=N)

	bin_edges = np.linspace(np.min(data),np.max(data),sturges(N))
	bin_content = np.array([len(data[(data>i)*(data<=g)]) for i,g in zip(bin_edges[:-1],bin_edges[1:])])
	bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])

	def g_total(x, N_background, a, b, c, N_signal, n, mu, sigma): # CDF!!!!!!!!!!!!!!
		return N_background * (a * x**3 * (1./3.) + b * x**2 * 0.5 + c * x) + \
			N_signal * n * sc.norm.cdf(x, loc = mu, scale = sigma)


	my_cost_func = ExtendedBinnedNLL(bin_content, bin_edges, g_total)

	# ---------------- BACKGROUND ------------------------------------------
	my_minuit = Minuit(my_cost_func, 
		           N_background = N, a=-0.1, b=1., c=1., 
		           N_signal = N, n=1., mu=np.mean(data), sigma=np.std(data)) # arbitrary starting values with the right signs
	my_minuit.limits['N_background', 'b', 'c', 'N_signal', 'n', 'mu', 'sigma'] = (0, None)
	my_minuit.limits['a'] = (None, 0)

	my_minuit.values['N_signal'] = 0
	my_minuit.fixed['N_signal', 'n', 'mu', 'sigma'] = True
	my_cost_func.mask = (bin_centres < 1.5) | (2.5 < bin_centres) # mask guessed looking at a preliminary histogram
	my_minuit.migrad()
	#print(my_minuit.valid)
	#display(my_minuit)
	# ----------------------------------------------------------------------

	# ---------- fix BKG and search for signal -----------------------------
	my_cost_func.mask = None
	my_minuit.fixed = False
	my_minuit.fixed['N_background', 'a', 'b', 'c'] = True
	my_minuit.values['N_signal'] = N - my_minuit.values['N_background']
	my_minuit.migrad()
	#print(my_minuit.valid)
	#display(my_minuit)
	#------------------------------------------------------------------------
	#------ all starting estimation are done, now search for the bests: -----
	my_minuit.fixed = False
	my_minuit.migrad()
	print(my_minuit.valid)
	print(my_minuit)
	
	'''
	
	# draw data and fitted line
	plt.errorbar(data_x, data_y, data_yerr, fmt="ok", label="data")
	plt.plot(data_x, line(data_x, *m.values), label="fit")

	# display legend with some fit info
	fit_info = [
		f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}",
	]
	for p, v, e in zip(m.parameters, m.values, m.errors):
		fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

	plt.legend(title="\n".join(fit_info), frameon=False)
	plt.xlabel("x")
	plt.ylabel("y");
	
	'''
	
## BINNED NNL
def binned_unbinned_NNL_example():
	N = 10000
	gauss_data = np.random.normal(size=N,loc=5,scale=10)
	# binned

	def mod_total_bin (bin_edges, mu, sigma):
		return sc.norm.cdf(bin_edges, mu, sigma)

	bin_edges = np.linspace(min(gauss_data),max(gauss_data),sturges(N))
	bin_content = np.array([len(gauss_data[(gauss_data>i)*(gauss_data<=g)]) for i,g in zip(bin_edges[:-1],bin_edges[1:])])

	my_cost_func_bin = BinnedNLL(bin_content, bin_edges, mod_total_bin)

	binned_minuit = Minuit (my_cost_func_bin, mu = np.mean(gauss_data), sigma = np.std(gauss_data))
	binned_minuit.migrad()
	print(binned_minuit)


	# unbinned
	def mod_signal_unb (x, mu, sigma) :
		return sc.norm.pdf(x, mu, sigma)

	my_cost_func_unb = UnbinnedNLL (gauss_data, mod_signal_unb)

	unbinned_minuit = Minuit (my_cost_func_unb, mu = np.mean(gauss_data), sigma = np.std(gauss_data))
	unbinned_minuit.migrad()
	print(unbinned_minuit)
	
def example_leasts_square_not_testable():

	# h è la pdf
	# NOTA: to fit least squares on gaussian fit N*binsize*pdf
	least_squares = LeastSquares(realX, realY, ystd, h)
	my_minuit = Minuit(least_squares, h0 = 0, g = expg)  # starting values for t1 and t2
	my_minuit.migrad()  # finds minimum of least_squares function
	my_minuit.hesse()   # accurately computes uncertainties
	display(my_minuit)

	is_valid = my_minuit.valid
	print ('success of the fit: ', is_valid)
	Q_squared = my_minuit.fval
	print ('value of the fit Q-squared', Q_squared)
	N_dof = my_minuit.ndof
	print ('value of the number of degrees of freedom', N_dof)

	for par, val, err in zip (my_minuit.parameters, my_minuit.values, my_minuit.errors) :
		print(f'{par} = {val:.3f} +/- {err:.3f}') # formatted output

if __name__ == '__main__':
	# print(sturges(1000))
	
	# f = lambda x: np.sin(x)
	
	# print(maximum(f,0,np.pi))
	# print(minimum(f,-np.pi/2,np.pi/2))
	
	# print(integral_hit_or_miss_N(f,0,np.pi))
	# print(crude_montecarlo(f,0,np.pi))
	
	# X = np.random.normal(size=1000, loc=3,scale=4)
	# print(Stats(X))
	
	# plt.scatter(rand_uniform(0,1,100),rand_uniform(0,1,100),marker='.',color='red')
 	# plt.show()
 	
 	'''
 	pdf_cauchy = lambda x,M,G: 1/np.pi * G / ((x-M)**2 + G**2)
 	fig, ax = plt.subplots(nrows = 1,ncols = 3)
 	ax[0].hist(rand_expon(1/5,1000),bins=sturges(1000),edgecolor='black',color='gold',density=True)
 	ax[1].hist(rand_poisson(4,1000),bins=np.arange(0,12),edgecolor='black',color='cornflowerblue',density=True)
 	ax[2].hist(rand_TAC(pdf_cauchy,3-8,3+8,params=[3,4],size=1000),bins=sturges(1000),edgecolor='black',color='mediumorchid',density=True)
 	plt.tight_layout()
 	plt.show()
 	'''
	
 	# extended_binned_NNL_example()
 	# binned_unbinned_NNL_example()
	



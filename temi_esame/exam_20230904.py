# Exam made using text editor
from CarminatiGiovanniLib import *

p0,p1,p2,p3 = 2,0.5,0.78,0.8

f = lambda x,p0,p1,p2,p3: p0*np.sin(p1*x+p2) + p3 
fix_f = lambda x: p0*np.sin(p1*x+p2) + p3


sigma = 0.3 # float(input('sigma: '))
X = np.array([0.5, 2.5, 4.5, 6.5, 8.5, 10.5],dtype=np.float64)
# X = np.linspace(0,20,20)
N = len(X)
epsilon = np.random.normal(scale=sigma,size=N)
Y = fix_f(X) + epsilon

'''
plt.errorbar(X,Y,np.std(epsilon),capsize=2,ecolor='black',color='red',fmt='o')
x = np.linspace(min(X),max(X),100)
plt.plot(x,fix_f(x))
plt.show()
'''

############ Q2 FIT ###################
'''
try_p0 = (np.max(Y) - np.min(Y)) / 2
try_p1 = 2*np.pi / 10 # looking at the spatial period in the graph
try_p2 = 4.5-np.pi / try_p1
try_p3 = np.max(Y) - np.min(Y)

least_squares = LeastSquares(X, Y, np.std(epsilon), f)
my_minuit = Minuit(least_squares, p0=try_p0,p1=try_p1,p2=try_p2,p3=try_p3)  # starting values for t1 and t2
my_minuit.migrad()  # finds minimum of least_squares function
my_minuit.hesse()   # accurately computes uncertainties
print(my_minuit)

is_valid = my_minuit.valid
print ('success of the fit: ', is_valid)
Q_squared = my_minuit.fval
print ('value of the fit Q-squared', Q_squared)
N_dof = my_minuit.ndof
print ('value of the number of degrees of freedom', N_dof)

for par, val, err in zip (my_minuit.parameters, my_minuit.values, my_minuit.errors) :
	print(f'{par} = {val:.3f} +/- {err:.3f}') # formatted output
'''
	
######### MORE ERRORSSSS !!!!! #############################

N = 20
X = np.random.rand(N)*10
def toy():
	e = np.random.normal(0,0.3,N)
	d = np.random.normal(0,0.5,N)
	s = e + d #np.random.normal(0,np.sqrt(0.5**2 + 0.3**2),N)
	Y = fix_f(X) + s
	try_p0 = (np.max(Y) - np.min(Y)) / 2
	try_p1 = 2*np.pi / 10 # looking at the spatial period in the graph
	try_p2 = 4.5-np.pi / try_p1
	try_p3 = np.max(Y) - np.min(Y)
	least_squares = LeastSquares(X, Y, np.sqrt(0.3**2 + 0.5**2), f)
	m = Minuit(least_squares, p0=try_p0,p1=try_p1,p2=try_p2,p3=try_p3)
	m.migrad()  # finds minimum of least_squares function
	# print(m)
	return m.fval
	
Q2 = np.array([toy() for _ in range(1000)])
plt.hist(Q2,bins=sturges(1000),edgecolor='black',color='navy',density=True)
x = np.linspace(np.min(Q2),np.max(Q2),100)
plt.plot(x,sc.chi2.pdf(x,N-4))
plt.show()




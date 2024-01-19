from CarminatiGiovanniLib import *

def pdf_parabola(x,x0):
	c = x0**2
	area = crude_montecarlo(lambda x: -x**2 + c,-x0,x0)[0]
	y = (-x**2 + c) / area
	if type(x) != np.ndarray: return y if y >=0 else 0
	return np.array([i if i >=0 else 0 for i in y]) # characteristic function
	
# x = np.linspace(-5,5,100)
# print(crude_montecarlo(lambda x: pdf_parabola(x,4),-5,5)[0])
# plt.plot(x,pdf_parabola(x,4))
# plt.show()

a,b,x0 = -1,1,0.5
# N = 1000
# data = rand_TAC(pdf_parabola,a,b,params=[x0],size=N)
# np.savetxt('rand_parabolic', data, delimiter='\n'

data = np.loadtxt('rand_parabolic')
N = len(data)


x = np.linspace(a,b,100)
plt.hist(data,bins=sturges(N),edgecolor='black',color='yellowgreen',density=True)
plt.plot(x,pdf_parabola(x,x0))
plt.show()




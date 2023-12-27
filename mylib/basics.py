import numpy as np

# returns fibonacci sequence up to N-th term
def fibonacciToN(N)->list:
    f = [1,1]
    for _ in range(0,N - 2):
        f.append(f[-1] + f[-2])
    return f

#return a list containing the factors of n
def factorize(n:int)->list:
    i = 2
    prime_factors = []
    while(i <= n):
        if n % i == 0:
            prime_factors.append(i)
            n = n/i
        else: i += 1
    return prime_factors

# return the median in an array
def median(a: np.ndarray) -> int:
    a = np.sort(a)
    return a[int(len(a)/2)-1]

# given an array return the value below lies the 25% of the elements
def barrier_below(a: np.ndarray, barrier: int) -> int: 
    a = np.sort(a)
    return a[int((len(a)/100)*barrier)-1]

def barrier_above(a: np.ndarray, barrier: int) -> int: 
    return barrier_below(a,100-barrier)

# mean, var, std, stdm, 

def mean(a):
    return np.sum(a)/len(a)

def variance(a):
    m = mean(a)
    return np.sum([(i-m)**2 for i in a])/len(a)

def std_dev(a):
    return np.sqrt(variance(a))

def std_dev_mean(a):
    return std_dev(a)/np.sqrt(len(a))

# best number of bins depending on the number of data:
def sturges (N_events) :
     return int( np.ceil( 1 + 3.322 * np.log(N_events) ) )

def data_reader_plotter(filename):
    import matplotlib.pyplot as plt
    data = np.loadtxt(filename,dtype=np.float64)
    bins = np.linspace(min(data),max(data),sturges(len(data)))
    plt.hist(data,bins = bins, color='springgreen',edgecolor='black')

    parameters = {
        'mean':data.mean(),
        'std':np.std(data),
        'var':np.std(data)**2,
        'std_mean':data.mean()/np.sqrt(len(data))
    }

    plt.show()
    print(parameters)

    return parameters
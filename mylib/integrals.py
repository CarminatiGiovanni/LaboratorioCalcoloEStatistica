import numpy as np

# from 0 to maxf
def integral_hit_or_miss_N(f,a,b,maxf,N = 1000):
    randX = a + np.random.rand(N) * (b - a)
    randY = 0 + np.random.rand(N) * (maxf - 0)
    hit = sum(map(lambda x: x[1] < f(x[0]),zip(randX,randY)))
    p = hit/N
    expected_value = p * (b-a) * (maxf - 0)
    variance = ((b-a) * (maxf - 0))**2 * p * (1-p) / N
    return expected_value, np.sqrt(variance)


def crude_montecarlo(f,a,b,N=1000000):
    X = a + np.random.rand(N) * (b-a)
    E = np.average(f(X))
    STD = np.std(f(X))
    return (b-a)*E, (b-a)*STD/np.sqrt(N)
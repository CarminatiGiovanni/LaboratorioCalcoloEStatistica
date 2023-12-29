import numpy as np

# find zero of a function f in the interval [a,b]
def zero(f,a,b,prec=0.0001):
    c = 0.5 * (b + a)
    if b - a < prec: return c
    if f(c) * f(a) > 0: return zero(f,c,b,prec)
    else: return zero(f,a,c,prec)

# return the minimum of a function in the interval [a,b]
# the additional argument x1,x2 should be:
# x1 = a + (1-0.618)*(b-a)
# x2 = a + 0.618*(b-a)
def minimum(f,a,x1,x2,b,prec = 0.0001):
    if np.abs(b - a) < prec: return x2
    if f(x1) < f(x2): return minimum(f,a,a+(1-0.618)*(x2-a),x1,x2,prec)
    else: return minimum(f,x1,x2,x1+0.618*(b-x1),b,prec)


# idem but finds maximum
def maximum(f,a,x1,x2,b,prec = 0.0001):
    if np.abs(b - a) < prec: return x2
    if f(x1) > f(x2): return maximum(f,a,a+(1-0.618)*(x2-a),x1,x2,prec)
    else: return maximum(f,x1,x2,x1+0.618*(b-x1),b,prec)

if __name__ == '__main__':
    print(maximum(lambda x: -(x-1)**2,-1,-1+(1-0.618)*3,-1+0.618*3,2))
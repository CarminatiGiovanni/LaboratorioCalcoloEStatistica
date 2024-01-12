import numpy as np
def sturges (N_events) :
     return int( np.ceil( 1 + 3.322 * np.log(N_events) ) )

def hist_style(N=None,color=None,bins=None):
    fav = ['tan','tomato','palegreen']
    if type(color) == int and color<len(fav) and color>=0:
        c = fav[color] 
    elif type(color) == str:
        c = color
    else:
        c = fav[np.random.random_integers(0,2)]

    if type(bins) == list or type(bins) == np.ndarray:
        bins = bins
    elif bins == None:
        bins = sturges(N)
    else:
        bins = bins

    return {'edgecolor':'black','density':True, \
            'bins':bins,'color':c}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data = np.random.normal(size=1000)
    plt.hist(data,**hist_style(len(data)))
    plt.show()
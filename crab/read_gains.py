import numpy as np
import matplotlib.pyplot as plt
import sys

def read_gains(x):
    g = np.zeros((len(x), 1024)) + 0j

    for i in xrange(len(x)):
        g[i] = ((x[i])[1])[0]
    
    return abs(g)

x = np.load(sys.argv[1])
g = read_gains(x)

plt.plot(g.T)
plt.show()

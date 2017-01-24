#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
import astropy.stats.funcs as ast

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: %s foldspec icounts" % sys.argv[0]
        sys.exit(1)
    f = np.load(sys.argv[1]) # Folded spectrum
    ic = np.load(sys.argv[2] if len(sys.argv) == 3 else sys.argv[1].replace('foldspec', 'icount'))

    t_obs = float(sys.argv[1].split('+')[-1].split('sec')[0])

    if f.shape[-1] == 4:
        fall = f[..., (0,3)].sum(-1)
    else:
        fall = f

    nall = fall/(ic+1e-15)

    ntavg = nall[:, :, :].mean(-1,keepdims=True).mean(axis=0, keepdims=True)
    nall = nall / ntavg - 1 #Divide by time averaged mean to normalize. This puts the intensity in units of T 
    #nall = nall - ntavg

    n2 = nall
    #n2 = nall.sum(0)
    n2c = n2.sum(0)

    vmin = n2c.mean()-2.*n2c.std()
    vmax = n2c.mean()+5.*n2c.std()

    plt.imshow(n2c, cmap=cm.Greys, interpolation='nearest', vmin=vmin, vmax=vmax, aspect='auto') #plot spectrum in bin units

    #plt.plot(n2c.sum(0))

    plt.xlabel("Phase")
    plt.ylabel("Frequency")

    plt.show()

#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
from spectools import *

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: %s waterfall.npy" % sys.argv[0]
        sys.exit(1)

    t_obs = float(sys.argv[1].split('+')[-1].split('sec')[0])
    name = str(sys.argv[1].split('+')[0])

    w = np.load(sys.argv[1]) # Waterfall file

    if w.shape[-1] == 4:
        w2 = w[..., (0,3)].sum(-1)
    else:
        w2 = w

    ntavg = w2.mean(axis=0, keepdims=True)
    nall = w2 / ntavg - 1

    taxis = nall.shape[1]*60./1000

    nallc = nall
    nallc = nall[1:nall.shape[0]-1]
    pulse_index = np.argmax(nallc.sum(1))

    vmin = nallc.mean()-2*nallc.std()
    vmax = nallc.mean()+5*nallc.std()

    plt.imshow(nallc.T, aspect='auto', interpolation='nearest', cmap=cm.Greys, vmin=vmin, vmax=vmax, extent=[0, t_obs, 0, w2.shape[1]])

    #plt.plot(nallc.sum(-1))
    plt.ylabel("frequency (bin)")
    plt.xlabel("time bin [%s microsecond]" % (taxis))

    plt.show()

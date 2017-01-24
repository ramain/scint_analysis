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

    #t_obs = float(sys.argv[1].split('+')[-1].split('sec')[0])
    t_obs = 6
    name = str(sys.argv[1].split('+')[0])

    w = np.load(sys.argv[1]) # Waterfall file

    if w.shape[-1] == 4:
        #w2 = abs(w[..., (0,3)]).sum(-1)
        w2 = abs(w).sum(-1)
    else:
        w2 = w

    ntavg = w2.mean(axis=0, keepdims=True)
    nall = w2 / ntavg - 1
    #nall = w2

    print nall.shape[0]
    taxis = 3000*nall.shape[1]*60./1000

    nallc = nall
    nallc = nall[1:-1]
    pulse_index = np.argmax(nallc.sum(1))

    # Bin time by a factor of 2
    #nallc = nallc.reshape(-1, 6, nallc.shape[1]).sum(axis=1)
    # Bin frequency by a factor of 4
    #n_clean = n_clean.reshape(n_clean.shape[0], -1, 4).sum(axis=-1)

    vmin = nallc.mean()-2*nallc.std()
    vmax = nallc.mean()+5*nallc.std()
    #np.save('Background',w[20000:30000].sum(0))
    #plt.imshow(nallc.T, aspect='auto', interpolation='nearest', cmap=cm.Greys, vmin=vmin, vmax=vmax)
    plt.imshow(nallc[pulse_index-1000:pulse_index+2000].T, aspect='auto', interpolation='nearest', cmap=cm.Greys, vmin=vmin, vmax=vmax, extent=[(pulse_index-10)*64*60e-9, (pulse_index+20)*64*60e-9, 0, w2.shape[1]])
    #x = 1e6*np.linspace((pulse_index-10)*64*60e-9,(pulse_index+20)*64*60e-9, num=nallc.shape[0])

    #plt.plot(nallc.sum(1))

    #plt.imshow(nallc[pulse_index-60:pulse_index+60].sum(-1)
    #plt.plot(np.linspace(0,6,nallc.shape[0]),nallc.sum(1))
    plt.ylabel("Frequency")
    plt.xlabel("time")

    plt.show()

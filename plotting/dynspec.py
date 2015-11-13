#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm

f_start = 400 # Edge frequency of used frequency range in MHz
bandwidth = 400 # Bandwidth in MHz
f_range = slice(0,1024) # Include all frequencies
pol_select = (0, 3)

def clip(n, sig):
    threshold = np.mean(n) + sig*np.std(n)
    bad = abs(n) > threshold
    n[bad] = np.mean(n)
    return n

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: %s foldspec icounts" % sys.argv[0] 
        # Run the code as eg: ./dynspec.py data_foldspec.npy data_icounts.py
        sys.exit(1)

    # Folded spectrum axes: time, frequency, phase, pol=4 (XX, XY, YX, YY).
    f = np.load(sys.argv[1]) # Folded spectrum
    ic = np.load(sys.argv[2] if len(sys.argv) == 3 else sys.argv[1].replace('foldspec', 'icount'))

    name = sys.argv[1].split('_')[-1].split('sec')[0]
    t_obs = float(sys.argv[1].split('+')[-1].split('sec')[0])

    if f.shape[-1] == 4:
        fall = f[..., pol_select].sum(-1)
    else:
        fall = f

    nall = fall/ic
    ntavg = nall[:, :, :].mean(-1,keepdims=True).mean(axis=0, keepdims=True)
    nall = nall / ntavg # Divide by time averaged mean to normalize. This puts the intensity in units of Tsys

    # Sum over time and frequency to get the averaged pulse profile, used only to find suitable background gates
    pprof_1D = nall[:,f_range,:].sum(0).sum(0)  
    pmax = np.argmax(pprof_1D)

    # Be careful - the default background ranges can contain pulse fulx 
    n_back1 = nall[:, f_range, pmax-10:pmax-5].mean(axis=-1, keepdims=True)
    n_back2 = nall[:, f_range, pmax+15:pmax+25].mean(axis=-1, keepdims=True)

    n_back = (n_back1+n_back2)/2.
    n_clean = (nall[:, f_range, :]-n_back) # Subtract average of background gates

    pprof_3D = n_clean[:,f_range,:].sum(1,keepdims=True).sum(0,keepdims=True)

    n_clean = n_clean * pprof_3D # Weight the dynamic spectrum by the time-averaged pulse profile.
    n_clean = n_clean.sum(-1) # Sum over phase bins

    # Bin time by a factor of 2
    #n_clean = n_clean.reshape(-1, 2, n_clean.shape[1]).sum(axis=1)
    # Bin frequency by a factor of 4
    #n_clean = n_clean.reshape(n_clean.shape[0], -1, 4).sum(axis=-1)

    vmin = n_clean.mean()-2*n_clean.std()
    vmax = n_clean.mean()+5.*n_clean.std()

    plt.imshow(n_clean.T, aspect='auto', cmap=cm.Greys, interpolation='nearest', extent=[0, t_obs/60.0, f_start+f_range.start*bandwidth/fall.shape[1], f_start+f_range.stop*bandwidth/fall.shape[1]], vmax=vmax, vmin=vmin, origin='lower') #plot dynamic spectrum with physical units


    plt.colorbar()
    plt.xlabel("time [minutes]")
    plt.ylabel("frequency [MHz]")

    plt.show()

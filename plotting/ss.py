#!/usr/bin/env python

from __future__ import division
import sys
import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
import astropy.stats.funcs as ast

##############
f_start = 400 # Lowest frequency in MHz
bandwidth = 400 # Bandwidth in MHz
f_range = slice(0,1024) # Run foldspec first to find best pars.
##############

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: %s foldspec icounts" % sys.argv[0] # Run the code as eg: ./plotspec.py data_foldspec.npy data_icounts.py
        sys.exit(1)
    f = np.load(sys.argv[1]) # Folded spectrum
    ic = np.load(sys.argv[2]) # Icounts

    t_obs = float(sys.argv[1].split('+')[-1].split('sec')[0])

    fall = f[:,:,:,(0,3)].sum(-1)
    nall = fall/ic

    ntavg = nall.mean(-1,keepdims=True).mean(axis=0, keepdims=True)
    ntvar = nall.std(-1,keepdims=True).mean(axis=0, keepdims=True)
    nall = nall / ntavg / ntvar #Divide by time averaged mean to normalize. This puts the intensity in units of T

    #nall = np.roll(nall, 100, axis=2) # For Crab Pulsar

    pprof_1D = nall[:,f_range,:].sum(0).sum(0)
    pprof_3D = nall[:,f_range,:].sum(1,keepdims=True).sum(0,keepdims=True)

    pmax = np.argmax(pprof_1D)

    n_back1 = nall[:, f_range, pmax-20:pmax-10].mean(axis=-1, keepdims=True)
    n_back2 = nall[:, f_range, pmax+10:pmax+20].mean(axis=-1, keepdims=True)
    n_back = (n_back1+n_back2)/2
    n_clean = (nall[:, f_range, :]-n_back).sum(axis=-1)

    n_clean = nall * pprof_3D   
    n_clean = n_clean[:,:,95:100].sum(-1)  # MP
    #n_clean = n_clean[:,:,16:21].sum(-1)  # IP

    # Perform sigma clipping on the data
    #n_clean = ast.sigma_clip(n_clean, sig=5, iters=1)

    # sum over 6 time bins
    #n_clean = n_clean.reshape(-1, 6, n_clean.shape[1]).sum(axis=1)
    # sum over 8 frequency bins
    #n_clean = n_clean.reshape(n_clean.shape[0], -1, 4).sum(axis=-1)

    fspec = np.fft.fft2(n_clean.T)
    fspec = np.fft.fftshift(fspec)

    fspec_top = fspec[0:(fspec.shape[0]//2),:]
    #fspec_top = fspec

    u_fdopp = np.fft.fftfreq(n_clean.shape[0],t_obs/n_clean.shape[0]) * 1000 # 1000/s = mHz
    u_tdelay = np.fft.fftfreq(n_clean.shape[1],bandwidth/n_clean.shape[1]) # 1 / MHz = microseconds

    vmax = 2*( np.log10(abs(fspec_top)).mean() + 5*np.log10(abs(fspec_top)).std() )
    vmin = 2*( np.log10(abs(fspec_top)).mean() - 2*np.log10(abs(fspec_top)).std() )
    plt.imshow(np.log10(abs(fspec_top)**2.0), aspect='auto', cmap=cm.Greys, interpolation='nearest', vmin=vmin,vmax=vmax, extent=[min(u_fdopp),max(u_fdopp),0,max(u_tdelay)] )

    plt.colorbar()
    plt.xlabel("doppler frequency [mHz]")
    plt.ylabel(r"time delay [$\mu$s]")

    plt.show()

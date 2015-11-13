#!/usr/bin/env python

"""
Perform Lyne 1984 method of counting crossings of 
dynamic spectrum to determine scintillation rate
"""

from __future__ import division
import sys
import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
import astropy.stats.funcs as ast
from astropy.time import Time
from scipy.optimize import curve_fit

##############
f_start = 318
bandwidth = 16
f_range = slice(0,512) # Run foldspec first to find best pars.
P_orb = 24.7 # Orbital period in hours
tbin = 180 # Time binning in seconds
##############

def mean_crossings(x, n_bin, t_bin):
    # Calculate the time-rate of scintillation in n_bin number of time bins of length t_bin
    x_binned = break_data(x, n_bin)
    m = np.zeros(shape=x_binned.shape)
    cross = []
    for i in range(0, x_binned.shape[0]):
        m[i] = x_binned[i] - x_binned[i].mean(keepdims=True)
        cross.append( len( np.where(np.diff( np.sign (m[i]-m[i].mean()) ))[0] ) )
    return np.array(cross) / t_bin # Returns rate of scintillation in each time bin

def srate(theta, i, a, vx, vy):
    y = (87.5/ (np.sin(i) * a) ) * np.sqrt( np.sin(i)**2.0 * (vx*vx+vy*vy)/ 87.5**2.0 + (2-np.sin(i)**2.0)/2 - 2*(vx*np.sin(theta)-vy*np.cos(theta)*np.cos(i))*np.sin(i)/87.5 - (np.sin(i)**2.0 *np.cos(2*theta) / 2) )
    return y

def dynspec(f,ic,d):    

    f = f # Folded spectrum
    ic = ic # Icounts
    t_obs = d/60

    fall = f[:,:,:,(0,3)].sum(-1)
    nall = fall/ic   

    ntavg = nall[:, :, :].mean(-1,keepdims=True).mean(axis=0, keepdims=True)
    nall = nall / ntavg #Divide by time averaged mean to normalize. This puts the intensity in units of T

    pprof_1D = nall[:,f_range,:].sum(0).sum(0)
    pmax = np.argmax(pprof_1D)

    n_back1 = nall[:, f_range, pmax-30:pmax-20].mean(axis=-1, keepdims=True)
    n_back2 = nall[:, f_range, pmax+20:pmax+30].mean(axis=-1, keepdims=True)
    n_back = (n_back1+n_back2)/2.
    n_clean = (nall[:, f_range, :]-n_back)
    
    pprof_3D = n_clean[:,f_range,:].sum(1,keepdims=True).sum(0,keepdims=True)

    n_clean = n_clean * pprof_3D

    n_clean = n_clean.sum(-1)

    # Perform sigma clipping on the data
    #n_clean = ast.sigma_clip(n_clean, sig=10, iters=1)

    # sum over 6 time bins
    #n_clean = n_clean.reshape(-1, 6, n_clean.shape[1]).sum(axis=1)
    # sum over 8 frequency bins
    #n_clean = n_clean.reshape(n_clean.shape[0], -1, 4).sum(axis=-1)

    return n_clean

if __name__ = '__main__':

    f1 = np.load("jbdadaB0655+64_512chan200ntbinfoldspec_2014-06-11T21:33:52.000+036000.0sec.npy")
    f2 = np.load("jbdadaB0655+64_512chan62ntbinfoldspec_2014-06-12T08:30:39.000+011160.0sec.npy")
    f3 = np.load("jbdadaB0655+64_512chan67ntbinfoldspec_2014-06-12T11:39:59.000+012060.0sec.npy")
    f4 = np.load("jbdadaB0655+64_512chan83ntbinfoldspec_2014-06-15T10:31:29.000+014940.0sec.npy")

    ic1 = np.load("jbdadaB0655+64_512chan200ntbinicount_2014-06-11T21:33:52.000+036000.0sec.npy")
    ic2 = np.load("jbdadaB0655+64_512chan62ntbinicount_2014-06-12T08:30:39.000+011160.0sec.npy")
    ic3 = np.load("jbdadaB0655+64_512chan67ntbinicount_2014-06-12T11:39:59.000+012060.0sec.npy")
    ic4 = np.load("jbdadaB0655+64_512chan83ntbinicount_2014-06-15T10:31:29.000+014940.0sec.npy")

    flist = [f1,f2,f3,f4]
    iclist = [ic1,ic2,ic3,ic4]
    dlist = [36000,11160,12060,14940]
    tlist = ['2014-06-11T21:33:52.0', '2014-06-12T08:30:39.0', '2014-06-12T11:39:59.0', '2014-06-15T10:31:29.0']

    for i in xrange(len(flist)):
        if i==0:
            n = dynspec(flist[i],iclist[i],dlist[i]).T
        else:
            n = np.column_stack( (n,dynspec(flist[i],iclist[i],dlist[i]).T) )
        if i < len(flist)-1:
            t = Time(tlist[i+1], scale='utc').unix - Time(tlist[i], scale='utc').unix
            dt = t - dlist[i]
            length = int(dt)/tbin
            nz = np.zeros((n.shape[0],length))
            n = np.column_stack((n,nz))    

        #ntavg = n.sum(-1,keepdims=True)
        #n = n / ntavg #Divide by time averaged mean to normalize. This puts the intensity in units of T

    mc = []

    print n.shape

    for i in xrange(8):
        if i==0:
            mc = mean_crossings(n[64*i:64*(i+1),0:360].sum(0),20,3)
        else:
            mc = np.column_stack( (mc,mean_crossings(n[64*i:64*(i+1),0:360].sum(0),20,3)) )

    mcmean=[]
    mcstd=[]
    theta=[]
    for i in xrange(mc.shape[0]):
        mcmean.append(np.mean(mc[i]))
        mcstd.append(np.std(mc[i]))
        theta.append(i*2*np.pi / 24.7)

    y = np.array(mcmean)
    yerr = np.array(mcstd)/np.sqrt(len(mc))
    theta = np.array(theta)

    theta_plot = np.linspace(0,max(theta),100)

    p0 = [1.5,40,20,5]

    popt, pcov = curve_fit(srate, theta, y, maxfev=10000)

    i = popt[0]*360/(2*np.pi)
    vd = np.sqrt( popt[2]**2.0 + popt[3]**2.0 )

    print("i [deg], a [km], vd [km/s] = %s, %s, %s" % (i, popt[1], vd))

    plt.errorbar(theta,y,xerr=0,yerr=yerr,linestyle='none')
    plt.plot(theta,y,'bo')
    plt.plot(theta_plot, srate(theta_plot,popt[0],popt[1],popt[2],popt[3]) )
    #plt.plot(theta, srate(theta,p0[0],p0[1],p0[2],p0[3]) )
    plt.ylim(-0.5,4.0)

    plt.xlabel('orbital phase')
    plt.ylabel(r'scintillation rate [hr$^{-1}$]')
    plt.show()


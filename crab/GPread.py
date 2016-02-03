#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import astropy.units as u
from scipy.stats import binned_statistic as bs
from astropy.time import Time

GPlist = np.loadtxt('/home/ramain/packages/scint_analysis/crab/data/gp_sorted.txt', dtype='string').T

PulseMean = np.load('data/PulseRespFull.npy')
# mask for bad frequencies
mask = np.load('data/mask.npy')

GPrange = slice(0,77700)
frange = slice(0,1024)
SN_cut = 100
Pbin = 2**5
dummy_spec = np.zeros(1024)[frange]
animate = 0 # Binary, whether to animate through MPs
#dummy_spec = np.delete(dummy_spec, mask)

Times = Time(GPlist[0], format='isot', scale='utc')[GPrange]
SN = GPlist[1].astype(float)[GPrange]
phase = GPlist[2].astype(float)[GPrange]

MPTimes = Times[abs(phase-0.2) < 0.05] #main pulse
IPTimes = Times[abs(phase-0.6) < 0.05] #interpulse
MPSN = SN[abs(phase-0.2) < 0.05] #main pulse
IPSN = SN[abs(phase-0.6) < 0.05] #interpulse

MPTimes = MPTimes[MPSN > SN_cut] #Only take high S/N pulses
MPdT = MPTimes[1:] - MPTimes[:-1]
MPTimes = MPTimes[1:][MPdT > 0*u.s]

IPTimes = IPTimes[IPSN > SN_cut] #Only take high S/N pulses
IPdT = IPTimes[1:] - IPTimes[:-1]
IPTimes = IPTimes[1:][IPdT > 0*u.s]

n = len(MPTimes)
m = len(IPTimes)

MPs=np.zeros((n,len(dummy_spec)))
IPs=np.zeros((m,len(dummy_spec)))
MMcorr=np.zeros(n**2)
MMdt=np.zeros(n**2)
IMcorr=np.zeros(n*m)
IMdt=np.zeros(n*m)
IIcorr=np.zeros(m**2)
IIdt=np.zeros(m**2)

# Construct an array with all main pulses in it
for i in xrange(n):
    x = np.load('/home/ramain/data/crab/GPlist-pulses/ARO-GPs/GP{0}.npy'.format(MPTimes[i].isot))
    MP = x[frange,(0,3)].sum(-1)

    MPsmooth = MP.reshape(len(MP)/Pbin, Pbin).mean(-1)
    MPsmooth = np.repeat(MPsmooth, Pbin)
    #MP -= MPsmooth

    #MP = np.delete(MP / PulseMean, mask)
    #MPs[i]=(MP-np.mean(MP))/np.std(MP)
    MPs[i] = MP

# Construct an array with all interpulses in it
for i in xrange(m):
    x = np.load('/home/ramain/data/crab/GPlist-pulses/ARO-GPs/GP{0}.npy'.format(IPTimes[i].isot))
    IP = x[frange,(0,3)].sum(-1)

    IPsmooth = IP.reshape(len(IP)/Pbin, Pbin).mean(-1)
    IPsmooth = np.repeat(IPsmooth, Pbin)


    #IP = np.delete(IP / PulseMean, mask)
    #IPs[i]=(IP-np.mean(IP))/np.std(IP)
    IPs[i] = IP

# Final calibrations:

#MPs = np.delete(MPs, mask, axis=1)
#IPs = np.delete(IPs, mask, axis=1)

if animate:

    ims = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in xrange(MPs.shape[0]):
        im = plt.plot( (MPs[i] - np.mean(MPs[i])) / np.mean(MPs[i]), 'k')
        #n = ax.annotate('N: ' + str(i),(25,50))
        ims.append(im)   
    
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=0)
#        ani.save('scint_spec.mp4')

    plt.ylim(-5,5)
    plt.show()


#!/usr/bin/env python

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import astropy.units as u
from astropy.time import Time

def plaw(x, a, b):
    return a*x**b

def line(x, a, b):
    return a*x + b

def key_event(e):
    global curr_pos

    if e.key == "right":
        curr_pos = curr_pos + 1
    elif e.key == "left":
        curr_pos = curr_pos - 1
    else:
        return
    curr_pos = curr_pos % len(plots)

    ax.cla()
    ax.plot(freq, plots[curr_pos])
    ax.plot(freq, plots2[curr_pos], 'r')
    ax.annotate('Pulse %s of %s' % (curr_pos, MPs.shape[0]) ,(650,20000))
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Flux [Jy]')
    plt.xscale('log')
    plt.xlim(400,800)
    plt.yscale('log')
    plt.ylim(1,1e5)
    fig.canvas.draw()

#GPlist = np.loadtxt('/home/ramain/packages/scint_analysis/crab/data/gp_sorted.txt', dtype='string').T
GPlist = np.loadtxt('/home/ramain/packages/scint_analysis/crab/gp_sorted.txt', dtype='string').T

PulseMean = np.load('data/PulseRespFull.npy')
# mask for bad frequencies
mask = np.load('data/mask.npy')

GPrange = slice(0,77700)
fitrange = slice(0,1024)
frange = slice(0,1024) 
MPSN_cut = 5
IPSN_cut = 5
binf = 1
dummy_spec = np.zeros(1024)[frange]
animate = 0 
calibrate = 1
fit = 0
cube = 0
#dummy_spec = np.delete(dummy_spec, mask)

Times = Time(GPlist[0], format='isot', scale='utc')
SN = GPlist[1].astype(float)
phase = GPlist[2].astype(float)

MPTimes = Times[abs(phase-0.2) < 0.05] #main pulse
IPTimes = Times[abs(phase-0.6) < 0.05] #interpulse
MPSN = SN[abs(phase-0.2) < 0.05] #main pulse
IPSN = SN[abs(phase-0.6) < 0.05] #interpulse
MPphase = phase[abs(phase-0.2) < 0.05]
IPphase = phase[abs(phase-0.6) < 0.05]

MPTimes = MPTimes[MPSN > MPSN_cut] #Only take high S/N pulses
MPdT = MPTimes[1:] - MPTimes[:-1]
MPTimes = MPTimes[1:][MPdT > 0.0*u.s]
MPphase = MPphase[MPSN > MPSN_cut]
MPSN = MPSN[MPSN > MPSN_cut]
MPSN = MPSN[1:][MPdT > 0.0*u.s]
MPdT = MPdT[MPdT > 0.0*u.s]

IPTimes = IPTimes[IPSN > IPSN_cut] #Only take high S/N pulses
IPdT = IPTimes[1:] - IPTimes[:-1]
IPTimes = IPTimes[1:][IPdT > 0.0*u.s]
IPphase = IPphase[IPSN > IPSN_cut]
IPSN = IPSN[IPSN > IPSN_cut]
IPSN = IPSN[1:][IPdT > 0.0*u.s]
IPdT = IPdT[IPdT > 0.0*u.s]

n = len(MPTimes)
m = len(IPTimes)

MPs=np.zeros((n,len(dummy_spec)))
IPs=np.zeros((m,len(dummy_spec)))

# Construct an array with all main pulses in it
for i in xrange(n):
    #x = np.load('/home/ramain/data/crab/GPlist-pulses/ARO-GPs/GP{0}.npy'.format(MPTimes[i].isot))
    x = np.load('/media/drive2/PulseSpectra/GP{0}.npy'.format(MPTimes[i].isot))
    MP = x[frange,(0,3)].sum(-1)
    MPs[i] = MP

# Construct an array with all interpulses in it
for i in xrange(m):
    #x = np.load('/home/ramain/data/crab/GPlist-pulses/ARO-GPs/GP{0}.npy'.format(IPTimes[i].isot))
    x = np.load('/media/drive2/PulseSpectra/GP{0}.npy'.format(IPTimes[i].isot))
    IP = x[frange,(0,3)].sum(-1)
    IPs[i] = IP


if calibrate:
    nebmodel = np.load('data/nebmodel.npy')
    fluxcal = np.load('data/fluxmodel.npy')
    mask = np.load('data/mask.npy')

    MPs /= nebmodel[np.newaxis]
    IPs /= nebmodel[np.newaxis]
    MPs *= 168*fluxcal
    IPs *= 168*fluxcal

MPs = MPs.reshape(-1, MPs.shape[-1] // binf, binf).mean(-1)
IPs = IPs.reshape(-1, IPs.shape[-1] // binf, binf).mean(-1)
freq = np.linspace(400, 800, 1024 // binf)

if fit:
    MPnorm = []
    MPindex = []
    MPflux = []
    MPerr = []
    IPnorm = []
    IPindex = []
    IPflux = []
    IPerr = []
    p0 = [1, -3]

    for i in xrange(MPs.shape[0]):
        #n_cal = (MPSN[i])**2.
        spec = MPs[i]
        spec = np.delete(spec, mask)
        freqfit = np.delete(freq, mask) / 1000.

        try:
            p, cov = curve_fit(plaw, freqfit[fitrange], spec[fitrange], p0=p0)    
            #print(p)

        except:
            print("MP Fit %s did not converge" % (i) )
            p = np.array([np.nan,np.nan])        
            cov = np.array([[np.nan,np.nan],[np.nan,np.nan]])

        MPnorm.append(p[0])
        MPindex.append(p[1])
        MPerr.append(cov[1,1])
        MPflux.append(np.mean(spec))

    for i in xrange(IPs.shape[0]):
        #n_cal = 10**(np.sqrt(MPSN[i]) / 2.)
        spec = IPs[i]
        spec = np.delete(spec, mask)
        freqfit = np.delete(freq, mask) / 1000.

        try:
            p, cov = curve_fit(plaw, freqfit[fitrange], spec[fitrange], p0=p0)

        except:
            print("IP Fit %s did not converge" % (i) )
            p = np.array([np.nan,np.nan])        
            cov = np.array([[np.nan,np.nan],[np.nan,np.nan]])

        IPnorm.append(p[0])
        IPindex.append(p[1])
        IPerr.append(cov[1,1])
        IPflux.append(np.mean(spec))


    #plt.hist(MPindex, bins=22, range=(-7,4), normed=1, histtype='step', color='b', label='Main Pulses')
    #plt.hist(IPindex, bins=22, range=(-7,4), normed=1, histtype='step', color='r', label='Interpulse')
    #plt.xlabel('Best Fit Spectral Index')
    #plt.legend()
    #plt.show()

if animate:

    ims = []
    imfits = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = fig.add_subplot(111)
    for i in xrange(MPs.shape[0]):
        im, = ax.plot( freq, MPs[i] , 'k' )
        im2, = ax2.plot( freq, plaw(freq, MPnorm[i], MPindex[i]), 'r' )
        n = ax.annotate('Pulse %s of %s' % (i, MPs.shape[0]) ,(650,20000))
        ims.append([im, im2, n])
        #imfits.append(imfit)   
    
    ani = animation.ArtistAnimation(fig, ims, interval=250, blit=False, repeat_delay=1000)
    #ani.save('scint_spec.mp4')

    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Flux [Jy]')
    plt.xscale('log')
    plt.xlim(400,800)
    plt.yscale('log')
    plt.ylim(1,1e5)
    plt.show()

if cube:

    plots = []
    plots2 = []
    for i in xrange(MPs.shape[0]):
        plots.append(MPs[i])
        plots2.append(plaw(freq, MPnorm[i], MPindex[i]))
 
    curr_pos = 0

    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', key_event)
    ax = fig.add_subplot(111)
    ax.plot(freq, plots[i])
    ax.plot(freq, plots2[i] ,'r')

    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Flux [Jy]')
    plt.xscale('log')
    plt.xlim(400,800)
    plt.yscale('log')
    plt.ylim(1,1e5)
    plt.show()


#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import astropy.units as u
from scipy.stats import binned_statistic as bs
from astropy.time import Time

GPlist = np.loadtxt('/home/ramain/packages/scint_analysis/crab/data/gp_sorted.txt', dtype='string').T

GPrange = slice(0,77700)
frange = slice(128,896)
SN_cut = 50
dummy_spec = np.zeros(1024)[frange]
binning = 20

Times = Time(GPlist[0], format='isot', scale='utc')[GPrange]
SN = GPlist[1].astype(float)[GPrange]
phase = GPlist[2].astype(float)[GPrange]

MPTimes = Times[abs(phase-0.2) < 0.05] #main pulse
IPTimes = Times[abs(phase-0.6) < 0.05] #interpulse
MPSN = SN[abs(phase-0.2) < 0.05] #main pulse
IPSN = SN[abs(phase-0.6) < 0.05] #interpulse

MPTimes = MPTimes[MPSN > SN_cut] #Only take high S/N pulses
IPTimes = IPTimes[IPSN > SN_cut] #Only take high S/N pulses

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
    MPs[i]=(MP-np.mean(MP))/np.std(MP)

# Construct an array with all interpulses in it
for i in xrange(m):
    x = np.load('/home/ramain/data/crab/GPlist-pulses/ARO-GPs/GP{0}.npy'.format(IPTimes[i].isot))
    IP = x[frange,(0,3)].sum(-1)
    IPs[i]=(IP-np.mean(IP))/np.std(IP)

for j in xrange(n):
    MP_temp=MPs[j]
    MP_temp=MP_temp[np.newaxis,:]
    norm = np.sqrt(np.sum(MP_temp*MP_temp) * (MPs*MPs).sum(1))
    MMcorr[j*n:(j+1)*n] = (MP_temp*MPs).sum(1) / norm
    MMdt[j*n:(j+1)*n] = MPTimes.unix-MPTimes.unix[j]

for j in xrange(m):
    IP_temp=IPs[j]
    IP_temp=IP_temp[np.newaxis,:]
    norm = np.sqrt(np.sum(IP_temp*IP_temp) * (IPs*IPs).sum(1))
    IIcorr[j*m:(j+1)*m] = (IP_temp*IPs).sum(1) / norm
    IIdt[j*m:(j+1)*m] = IPTimes.unix-IPTimes.unix[j]

    norm = np.sqrt(np.sum(IP_temp*IP_temp) * (MPs*MPs).sum(1))
    IMcorr[j*n:(j+1)*n] = (IP_temp*MPs).sum(1) / norm
    IMdt[j*n:(j+1)*n] = MPTimes.unix-IPTimes.unix[j]
    

MMcorr = MMcorr[MMdt>0]
MMdt = MMdt[MMdt>0]
IIcorr = IIcorr[IIdt>0]
IIdt = IIdt[IIdt>0]
IMcorr = IMcorr[IMdt>0]
IMdt = IMdt[IMdt>0]

if binning:
    tbins = np.linspace(-2,4,binning)
    dt = tbins[1] - tbins[0]
    tbins += dt/2
    MMavg=np.zeros(binning-1)
    MMerr=np.zeros(binning-1)
    IIavg=np.zeros(binning-1)
    IIerr=np.zeros(binning-1)
    IMavg=np.zeros(binning-1)
    IMerr=np.zeros(binning-1)

    for i in xrange(len(tbins)-1):
        MMavg[i] = np.mean( MMcorr[abs(np.log10(MMdt) - tbins[i]) < dt/2] )
        MMerr[i] = np.std( MMcorr[abs(np.log10(MMdt) - tbins[i]) < dt/2] ) / np.sqrt( len(MMcorr[abs(np.log10(MMdt) - tbins[i]) < dt/2]) )
        IMavg[i] = np.mean( IMcorr[abs(np.log10(IMdt) - tbins[i]) < dt/2] )
        IMerr[i] = np.std( IMcorr[abs(np.log10(IMdt) - tbins[i]) < dt/2] ) / np.sqrt( len(IMcorr[abs(np.log10(IMdt) - tbins[i]) < dt/2]) )
        IIavg[i] = np.mean( IIcorr[abs(np.log10(IIdt) - tbins[i]) < dt/2] )
        IIerr[i] = np.std( IIcorr[abs(np.log10(IIdt) - tbins[i]) < dt/2] ) / np.sqrt( len(IIcorr[abs(np.log10(IIdt) - tbins[i]) < dt/2]) )

    MMerr[MMerr<1e-5] = MMavg[MMerr<1e-5]
    IMerr[IMerr<1e-5] = IMavg[IMerr<1e-5]
    IIerr[IIerr<1e-5] = IIavg[IIerr<1e-5]

    plt.errorbar(tbins[:-1], MMavg, yerr=MMerr, fmt='--o',label='Main to Main')
    plt.errorbar(tbins[:-1], IMavg, yerr=IMerr, fmt='--o',label='Main to Inter')
    plt.errorbar(tbins[:-1], IIavg, yerr=IIerr, fmt='--o',label='Inter to Inter')
    plt.legend(loc=1)

    plt.xlabel('Time [log10(s)]')
    plt.ylabel('Correlation Coefficient')
    plt.ylim((0.2,1))

else:
    plt.plot(np.log10(MMdt),MMcorr,'bx',label='Main to Main')
    plt.plot(np.log10(IMdt),IMcorr,'gx',label='Main to Inter')
    plt.plot(np.log10(IIdt),IIcorr,'rx',label='Inter to Inter')


    plt.legend(loc=4)
    plt.xlabel('Time[s]')
    plt.ylabel('Correlation Coefficient')
    plt.ylim((-0.3,1))
    plt.xlim((-2,4))

plt.show()

#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import astropy.units as u
from astropy.time import Time

PulseWeighting = 1

GPlist = np.loadtxt('/home/ramain/packages/scint_analysis/crab/data/gp_sorted.txt', dtype='string').T

frange = slice(256,820)
nchan = len(np.zeros(1024)[frange])

Times = Time(GPlist[0], format='isot', scale='utc')[0:12000]
SN = GPlist[1].astype(float)[0:12000]
phase = GPlist[2].astype(float)[0:12000]

Times = Times[abs(phase-0.2) < 0.05] #main pulse
#Times = Times[abs(phase-0.6) < 0.05] #interpulse
SN = SN[abs(phase-0.2) < 0.05] #main pulse
#SN = SN[abs(phase-0.6) < 0.05] #interpulse

Times = Times[SN > 10] #Only take high S/N pulses

t = (Times.unix[-1] - Times.unix[0]) * u.s
t0 = Times[0]
dt = 30.*u.s

ntbin = int(t / dt)

dynspec = np.zeros((ntbin,1024))

for i in xrange(ntbin):
    tstep = Times[abs(Times+dt/2-t0) < dt/2]
    dyn_temp = np.zeros(1024)
    for j in xrange(len(tstep)):
        x = np.load('/home/ramain/data/crab/GPlist-pulses/ARO-GPs/GP{0}.npy'.format(tstep[j].isot))
        if PulseWeighting:
            dyn_temp += x[:,(0,3)].sum(-1)  * x[:,(0,3)].sum()  #Pulse weighting
        else:
            dyn_temp += x[:,(0,3)].sum(-1)

    dynspec[i] = dyn_temp

    if dynspec[i].mean() > 0:
        dynspec[i] /= dynspec.std()
    t0 += dt

dyn = (dynspec-dynspec.mean(axis=0,keepdims=True) ).T

# mask bad frequencies
dyn[60] = 0
dyn[155:157] = 0
dyn[330:370] = 0
dyn[470:485] = 0

vmin = dyn.mean()-2.0*dyn.std()
vmax = dyn.mean()+5.0*dyn.std()

dyn = dyn[frange]
# Bin freq by a factor of 4
dyn = dyn.reshape(-1, 4, dyn.shape[1]).mean(axis=1)

plt.imshow(dyn,interpolation='nearest',vmin=vmin,vmax=vmax,cmap=cm.Greys,aspect='auto',extent=[0,ntbin*dt/u.s,700,500])
plt.xlabel('time [s]')
plt.ylabel('frequency [MHz]')
#plt.savefig('/home/ramain/IP_dynspec.png', format='png', dpi=1000)
plt.show()

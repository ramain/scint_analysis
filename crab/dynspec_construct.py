#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import astropy.units as u
from astropy.time import Time

GPlist = np.loadtxt('/home/ramain/gp_sorted.txt', dtype='string').T

Times = Time(GPlist[0], format='isot', scale='utc')[20000:21000]
SN = GPlist[1].astype(float)[20000:21000]
phase = GPlist[2].astype(float)[20000:21000]

Times = Times[abs(phase-0.2) < 0.05] #main pulse
#Times = Times[abs(phase-0.6) < 0.05] #interpulse
SN = SN[abs(phase-0.2) < 0.05] #main pulse
#SN = SN[abs(phase-0.6) < 0.05] #interpulse

#Times = Times[SN > 10] #Only take high S/N pulses

t = (Times.unix[-1] - Times.unix[0]) * u.s
t0 = Times[0]
dt = 2.*u.s

ntbin = int(t / dt)

dynspec = np.zeros((ntbin,1024))

for i in xrange(ntbin):
    tstep = Times[abs(Times+dt/2-t0) < dt/2]
    dyn_temp = np.zeros(1024)
    for j in xrange(len(tstep)):
        x = np.load('GP{0}.npy'.format(tstep[j].isot))
        dyn_temp += x
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

vmin = dyn.mean()-dyn.std()
vmax = dyn.mean()+1.5*dyn.std()

plt.imshow(dyn,interpolation='nearest',vmin=0,vmax=vmax,cmap=cm.Greys,aspect='auto',extent=[0,ntbin*dt/u.s,800,400])
plt.xlabel('time [s]')
plt.ylabel('frequency [MHz]')
plt.savefig('/home/ramain/MP_dynspec1.png', format='png', dpi=1000)
plt.show()

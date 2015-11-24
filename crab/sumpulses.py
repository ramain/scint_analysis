#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import astropy.units as u
from astropy.time import Time
import faraday_tools as ft

def SumPulses(tlist,nfreq,npol):
    PulseAvg = np.zeros((nfreq,npol))
    for x in tlist:
        Pulse = np.load('/home/ramain/data/crab/GPlist-pulses/ARO-GPs/GP{0}.npy'.format(x.isot))
        PulseAvg += Pulse
    return PulseAvg

def StokesPars(Pulse):
    I = Pulse[:,(0,3)].sum(-1) / 2
    Q = ( Pulse[:,0] - Pulse[:,3] ) / 2
    U = Pulse[:,1]
    V = Pulse[:,2]
    return I, Q, U, V

def RMfitting(Q, RM):
    x = ft.FaradayTools(Q)
    pars = x.fit_RM(RM)
    vecFR = pars[0] * np.exp(2j*(pars[1]*x.freq**2.0 + 
                                  pars[2])) + pars[3]
    return vecFR

def CDfitting(U, guess):
    x = ft.FaradayTools(U)
    pars = x.fit_RM_cabdelay(guess)
    vecCD = pars[0] * np.exp(-2j*np.pi*(np.arange(x.nfreq)/x.nfreq * 
                                  pars[1]+pars[2]))+pars[3]
    return vecCD

if __name__ == "__main__":

    GPlist = np.loadtxt('/home/ramain/packages/scint_analysis/crab/gp_sorted.txt', dtype='string').T

    RM = 42 # Starting guess
    guess = 10 # Not sure what this is, 10 gives good result...

    Times = Time(GPlist[0], format='isot', scale='utc')
    SN = GPlist[1].astype(float)
    phase = GPlist[2].astype(float)

    Times = Times[abs(phase-0.2) < 0.05] #main pulse
    SN = SN[abs(phase-0.2) < 0.05] #main pulse

    Times = Times[SN > 10] #Only take high S/N pulses
    SN = SN[SN > 10]

    t0 = Times[0]
    dt = 30.*u.minute
    freq = np.linspace(400, 800, 1024)

    t_pulses = Times[Times < t0+dt]

    PulseAvg = SumPulses(t_pulses,1024,4)

    StokesArray = np.zeros_like(PulseAvg)
    I, Q, U, V = StokesPars(PulseAvg)

    #vecFR = RMfitting(Q, RM)
    #vecCD = CDfitting(U, guess)

    # Hardcoded - testing right now
    # GP = np.load('/home/ramain/data/crab/GPlist-pulses/ARO-GPs/GP2015-07-24T08:04:25.216.npy')
    # GP = np.load('/home/ramain/data/crab/GPlist-pulses/ARO-GPs/GP2015-07-24T08:04:41.046.npy')
    #I, Q, U, V = StokesPars(GP)
    vecFR = np.load('vecFR.npy')
    vecCD = np.load('vecCD.npy')

    P_array=[]
    Pang_array=[]
    for x in t_pulses:
        GP = np.load('/home/ramain/data/crab/GPlist-pulses/ARO-GPs/GP{0}.npy'.format(x.isot))
        I, Q, U, V = StokesPars(GP)
        Q = Q*vecFR
        U = U*vecCD
        P = Q + 1j*U
        P_array.append(sum(abs(P)) / sum(I))
        Pang_array.append(np.angle(sum(P)))

    plt.plot(Pang_array, SN[Times < t0+dt], 'bx')
    #plt.scatter(Pang_array, P_array, marker='x', c=SN[Times < t0+dt], cmap=cm.coolwarm)

    plt.show()

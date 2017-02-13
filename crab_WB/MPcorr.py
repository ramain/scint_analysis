import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import astropy.units as u
from astropy.time import Time
import sys

Tcut = 30
SNcut = 30

def ReadPulselist(f, SNcut=1, phase0=0.5, phasewidth=0.5):
    T, SN, phase = np.loadtxt(f, dtype='string').T
    
    T = Time(T, precision=9)

    Tu, indeces = np.unique(T.isot, return_index=True)

    T = Time(Tu, precision=9)
    SN = SN[indeces].astype('float')
    phase = phase[indeces].astype('float')

    T = T[SN>SNcut]
    phase = phase[SN>SNcut]
    SN = SN[SN>SNcut]

    T = T[abs(phase-phase0) < phasewidth]
    SN = SN[abs(phase-phase0) < phasewidth]
    phase = phase[abs(phase-phase0) < phasewidth]

    return T, SN, phase

def ExtractPulse(tgp):
    dchan = np.load('/media/drive2/WbGPs/GP{0}.npy'.format(tgp.isot))
    dR = np.concatenate((dchan[:,::-1,2][:,:-1], dchan[...,0][:,1:]), axis=1)
    dL = np.concatenate((dchan[:,::-1,3][:,:-1], dchan[...,1][:,1:]), axis=1)
    x = (abs(dR)**2 + abs(dL)**2)
    pulse = x[49:53].mean(0)
    noise = x[1:5].mean(0)
    bg = x[10:30].mean(0)
    GP = (pulse - bg)  
    n = (noise - bg)
    return GP[16:-16], noise[16:-16]

def CorrPulses(GP1, N1, GP2, N2):
    autoc = (GP1 - np.mean(GP1))  * (GP2 - np.mean(GP2))

    std1 = np.std(GP1)**2.0 - np.std(N1)**2.0
    std2 = np.std(GP2)**2.0 - np.std(N2)**2.0

    corr = np.mean( autoc / np.sqrt(std1 * std2))
    return corr


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print "Usage: %s gplist" % sys.argv[0] 

    gplist = str(sys.argv[1])

    T, SN, phase = ReadPulselist(gplist, SNcut=SNcut, phase0=0.73, phasewidth=0.02)

    l = len(T)
    pairDT = np.zeros(l*l)
    corr = np.zeros(l*l)
    Tu = T.unix

    for i in range(l):
        for j in range(l):
            diff = np.abs(Tu[i] - Tu[j])
            pairDT[i*l+j] = diff

    for i in range(l):
        print (i)
        GP1, N1 = ExtractPulse(T[i])
        for j in range(l):
            if pairDT[i*l+j] < Tcut:
                GP2, N2 = ExtractPulse(T[j])
                corr[i*l+j] = CorrPulses(GP1, N1, GP2, N2)

    DTplot = []
    Cplot = []
    Cerr = []

    # Bin every cycle
    #for i in range(1000):  
    #    DTrange = DTu[abs(DTu-i*0.0337 < 0.01]
    #    crange = corru[abs(DTu-i*0.0337) < 0.01]
    #    DTplot.append(i)
    #    Cplot.append(np.mean(crange))
    #    Cerr.append(np.std(crange))

    # Bin every second
    for i in range(30):  
        DTrange = DTu[abs(DTu-(i+0.5)) < 0.498]
        crange = corru[abs(DTu-(i+0.5)) < 0.498]
        DTplot.append(i+1)
        crange = crange[~np.isnan(crange)]
        Cplot.append(np.mean(crange) )
        Cerr.append(np.std(crange)/np.sqrt(len(crange)))

plt.errorbar(DTplot, Cplot, yerr=Cerr, fmt='kx')

plt.plot(pairDT[corr != 0], corr[corr != 0 ], 'bx')
plt.show()  

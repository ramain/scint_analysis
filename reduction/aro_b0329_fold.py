from __future__ import division
from baseband import vdif
import numpy as np
import astropy.units as u
import glob
import sys
import matplotlib.pyplot as plt
from astropy.time import Time
from pulsar.predictor import Polyco

#dm = 555. * u.pc / u.cm**3  # Frb121011
#dm = 26.7641 * u.pc / u.cm**3  # b1919 / LGM1
dm = 71.0227 * u.pc / u.cm**3  # b1937

#P = 0.71451 * u.s
P = 1.33730216019 * u.s

size = 2 ** 16  # there are 2**16 time samples per file
sample_rate = 800/1024. * u.MHz
dt1 = 2.56e-6 * u.s

fedge = 800 * u.MHz
dispersion_delay_constant = 4149. * u.s * u.MHz**2 * u.cm**3 / u.pc

# Set up frequencies for de-dispersion
freq = np.linspace(800,400,1025)[:-1]*u.MHz
fref = freq
fcoh = freq - np.fft.fftfreq(size, 1024 / (400*u.MHz))[:, np.newaxis]

# Pre-calculate de-dispersion delays (coherent phases + incoherent offsets)
dang = (dispersion_delay_constant * dm * fcoh * (1./fref-1./fcoh)**2) * u.cycle
with u.set_enabled_equivalencies(u.dimensionless_angles()):
    dd_coh = np.exp(dang * 1j).conj().astype(np.complex64);

# Dispersion delays relative to 800MHz
dt = dispersion_delay_constant * dm * (1./freq**2 - 1./(800*u.MHz)**2)

T = int(sys.argv[1])*u.s
ngate = int(sys.argv[3])

tfile = dt1*size
Nfiles = int(T // tfile + 1)

# Begin at time offset specified in command (seconds from start of obs)
tstart = int(sys.argv[2]) * u.s
dstart = int(tstart // tfile)

# Initialize empty array for waterfall
#f = np.zeros((Nfiles, ngate, 2, 1024))
#ic = np.zeros((Nfiles, ngate, 2, 1024))
f = np.zeros((1, ngate, 2, 1024))
ic = np.zeros((1, ngate, 2, 1024))

filelist = glob.glob('./b1919/*vdif')
filelist = np.sort(filelist)

for i in range(dstart, dstart+Nfiles):

    print("on %s of %s" % (i, Nfiles))

    print("opening file")
    fn = filelist[i]
    fh = vdif.open(fn, mode='rs', sample_rate=sample_rate)

    d = np.conj(fh.read(size))

    print("de-dispersing")
    #d_ft = np.fft.fft(d, axis=0)
    #d_ft *= dd_coh[:,np.newaxis]
    #d = np.fft.ifft(d_ft, axis=0)
 
    for j in xrange(len(dt)):
        d[...,j] = np.roll(d[...,j], -int(dt[j]//dt1), axis=0)
   
    if i == 0:
        t0 = fh.tell("time")
    t = fh.tell("time")
    dt0 = (t - t0).to(u.s)

    # Calculate phase from phase polynomial
    #phasepol = polyco.phasepol(t0, rphase='fraction', t0=t0, time_unit=u.second, convert=True)
    #phase = phasepol(dt0 + np.arange(size) * dt1.to(u.s).value)
    #phase = np.remainder(phase, 1)

    # For slow pulsars, can just use the period
    phase = ((dt0 + np.arange(size) * dt1) % P).value
    phase *= ngate
    phase = phase.astype('int')

    print("folding")
    for j in range(ngate):
        p = d[phase==j]
        for k in range(p.shape[0]):
            f[0, j] += abs(p[k])**2.0
            ic[0, j] += 1

n = f / (ic+1e-5)
n2 = n[0]
n2 = n2.sum(1)
n2 = n2 - np.median(n2,axis=0)
n2 = n2.reshape(8,1024//8,ngate).sum(0)

plt.ion()
plt.imshow(n2.T, aspect='auto')

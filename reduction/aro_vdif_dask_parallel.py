from __future__ import division
from baseband import vdif
import numpy as np
import astropy.units as u
#import matplotlib.pylab as plt
import glob
import sys
from astropy.time import Time
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
Csize = comm.Get_size()
rank = comm.Get_rank()

import dask.array as da

#dm = 556.5 * u.pc / u.cm**3  # Frb121011
dm = 12.4370 * u.pc / u.cm**3  # b1919 / LGM1
#dm = 71.0227 * u.pc / u.cm**3  # b1937

size = 2 ** 16  # there are 2**26 time samples per file
nstep = int(size // size)
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

if len(sys.argv) < 3:
    print("Usage: aro_vdif_waterfall.py Tintegrate Tstart PathtoData")


# Dispersion delays relative to 800MHz
dt = dispersion_delay_constant * dm * (1./freq**2 - 1./(800*u.MHz)**2)


T = int(sys.argv[1])*u.s
binf = 1024
tbin = dt1*binf
ntbin = size // binf
tfile = dt1*size
tpad = int(max(dt) // tbin + 1)

# Begin at time offset specified in command (seconds from start of obs)
tstart = int(sys.argv[2]) * u.s
dstart = int(tstart // tfile)

# Initialize empty array for waterfall
w = np.zeros((int(T // tbin) + tpad, 2, 1024)) + 0j
path = sys.argv[3]

filelist = glob.glob('%s*vdif' % (path))
filelist = np.sort(filelist)

N = int(T//tfile)+1
Npcore = int(T // (tfile*Csize)) + 1

for i in range(rank*Npcore + dstart, (rank+1)*Npcore + dstart):
    if (i - dstart) > N:
        print("Rank %s exiting" % (rank))
        break

    print("Rank %s on %s of %s" % ( rank, (i - rank*Npcore - dstart), Npcore) )
    fn = filelist[i]
    fh = vdif.open(fn, mode='rs', sample_rate=sample_rate)

    print("reading")
    d = fh.read(size)
    # Turn into chunked dask array
    print("de-dispersing")
    d = da.from_array(d, chunks=(d.shape[0],d.shape[1], 32))
    d = da.fft.fft(d, axis=0)
    d *= dd_coh[:,np.newaxis]
    d = da.fft.ifft(d, axis=0)
    # De-Chunk the array, to allow efficient rechaping
    print("reshaping and forming waterfall")
    d = d.rechunk(d.shape)
    d = abs(d).reshape(-1, binf, 2, 1024).mean(1)
    w[(i-dstart)*ntbin:(i-dstart+1)*ntbin] = d

for i in xrange(len(dt)):
    w[...,i] = np.roll(w[...,i], -int(dt[i]//tbin), axis=0)

w = abs(w).sum(1)
w2 = np.zeros_like(w)
print("Rank %s Comm reduce" % (rank))
comm.Reduce(w, w2, op=MPI.SUM, root=0)

if rank == 0:
    np.save('waterfall-t%s-%ssec.npy' % (str(tstart.value), str(T.value)), w2)

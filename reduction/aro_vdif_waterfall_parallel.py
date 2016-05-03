from __future__ import division
from baseband import vdif
import numpy as np
import astropy.units as u
import matplotlib.pylab as plt
import glob
import sys
from astropy.time import Time
from mpi4py import MPI

comm = MPI.COMM_WORLD
Csize = comm.Get_size()
rank = comm.Get_rank()

size = 2 ** 16  # there are 2**26 time samples per file
sample_rate = 800/1024. * u.MHz
dt1 = 2.56e-6 * u.s

fedge = 800 * u.MHz
dispersion_delay_constant = 4149. * u.s * u.MHz**2 * u.cm**3 / u.pc
#dm = 555. * u.pc / u.cm**3  # Frb121011
dm = 12.4370 * u.pc / u.cm**3  # b1919 / LGM1

# Set up frequencies for de-dispersion
freq = np.linspace(400,800,1025)[:-1]*u.MHz
freq = freq[::-1]
fref = freq
fcoh = freq - np.fft.fftfreq(size, 1024 / (400*u.MHz))[:, np.newaxis]

# Pre-calculate de-dispersion delays (coherent phases + incoherent offsets)
dang = (dispersion_delay_constant * dm * freq * (1./fref-1./fcoh)**2) * u.cycle
with u.set_enabled_equivalencies(u.dimensionless_angles()):
    dd_coh = np.exp(dang * 1j).conj().astype(np.complex64);

# Dispersion delays relative to 800MHz
dt = dispersion_delay_constant * dm * (1./freq**2 - 1./(800*u.MHz)**2)


T = 4.0*u.s
binf = 512
tbin = dt1*binf
ntbin = size // binf
tfile = dt1*size
tpad = int(max(dt) // tbin + 1)

# Begin at time offset specified in command (seconds from start of obs)
tstart = int(sys.argv[1]) * u.s
dstart = int(tstart // tfile)

# Initialize empty array for waterfall
w = np.zeros((int(T // tbin) + tpad, 2, 1024)) + 0j

#filelist = glob.glob('./data/20160422T132947Z_aro_raw/*vdif')
filelist = glob.glob('/media/drive2/b1919/*vdif')
filelist = np.sort(filelist)

N = int(T//tfile)+1
Npcore = max(N // Csize, 1)

for i in range(rank*Npcore + dstart, (rank+1)*Npcore + dstart):

    if (i - dstart) > N:
        print("Rank %s exiting" % (rank))
        break

    print("Rank %s on %s of %s" % ( rank, (i - rank*Npcore - dstart), Npcore) )
    fn = filelist[i]
    fh = vdif.open(fn, mode='rs', sample_rate=sample_rate)
    try:
        d = fh.read(size)
        d_ft = np.fft.fft(d, axis=0)
        d_ft *= dd_coh
        d = np.fft.ifft(d_ft, axis=0)
        d = d.reshape(-1, binf, 2, 1024).mean(1)

        w[i*ntbin:(i+1)*ntbin] = d
    except:
        print("file %s failed assertion" % (str(i)))

for i in xrange(len(dt)):
    w[...,i] = np.roll(w[...,i], int(dt[i]//tbin), axis=0)

w = abs(w.sum(1))
w2 = np.zeros_like(w)
comm.Reduce(w, w2, op=MPI.SUM, root=0)

if rank == 0:
    np.save('waterfall-t%s-%ssec.npy' % (str(tstart.value), str(T.value)), w2)
#    plt.imshow(abs(w2[:,0]).T, aspect='auto', interpolation='nearest')
#    plt.show() 

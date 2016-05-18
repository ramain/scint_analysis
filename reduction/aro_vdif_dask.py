from __future__ import division
from baseband import vdif
import numpy as np
import astropy.units as u
import matplotlib.pylab as plt
import glob
import dask.array as da
from astropy.time import Time

fn = './data/20160422T132947Z_aro_raw/0000000.vdif'
size = 2 ** 16  # 1.048576 s
sample_rate = 800/1024. * u.MHz
dt1 = 2.56e-6 * u.s

fedge = 800 * u.MHz
dispersion_delay_constant = 4149. * u.s * u.MHz**2 * u.cm**3 / u.pc
#dm = 555. * u.pc / u.cm**3
dm = 12.4370 * u.pc / u.cm**3  # b1919 / LGM1

# Set up frequencies for de-dispersion
freq = np.linspace(800,400,1025)[:-1]*u.MHz
fref = freq
fcoh = freq - np.fft.fftfreq(size, 1024 / (400*u.MHz))[:, np.newaxis]

# Pre-calculate de-dispersion delays (coherent phases + incoherent offsets)
dang = (dispersion_delay_constant * dm * freq * (1./fref-1./fcoh)**2) * u.cycle
with u.set_enabled_equivalencies(u.dimensionless_angles()):
    dd_coh = np.exp(dang * 1j).conj().astype(np.complex64)

dt = dispersion_delay_constant * dm * (1./freq**2 - 1./(800*u.MHz)**2)

T = 2.0*u.s
binf = 1024
tbin = dt1*binf
ntbin = size // binf
tfile = dt1*size
tpad = int(max(dt) // tbin + 1)

# Initialize empty array for waterfall
w = np.zeros((int(T // tbin) + tpad, 2, 1024))

#filelist = glob.glob('./data/20160422T132947Z_aro_raw/*vdif')
filelist = glob.glob('/media/drive2/b1919/*vdif')
filelist = np.sort(filelist)

N = int(T//tfile)+1

for i in xrange(int(T//tfile)+1):

    print("File %s of %s" % ( i, int(T//tfile) ) )
    fn = filelist[i]
    print("Reading...")
    fh = vdif.open(fn, mode='rs', sample_rate=sample_rate)

    d = fh.read(size)
    # Turn into a chunked dask array
    dc = da.from_array(d, chunks=(d.shape[0],d.shape[1], 32))
    print("De-dispersing...")
    dc = da.fft.fft(dc, axis=0)
    dc *= dd_coh[:,np.newaxis]
    dc = da.fft.ifft(dc, axis=0)
    print("Re-shaping and squaring...")
    d = dc.rechunk(d.shape)
    d = abs(d).reshape(-1, binf, 2, 1024).mean(1)
    print("Adding to Waterfall...")
    w[i*ntbin:(i+1)*ntbin] = d

print("Done, incoherently shifting waterfall")
for i in xrange(len(dt)):
    w[...,i] = np.roll(w[...,i], -int(dt[i]//tbin), axis=0)
plt.imshow(abs(w[:,0]).T, aspect='auto')
plt.show() 

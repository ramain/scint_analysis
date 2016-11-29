import numpy as np
import astropy.units as u
from astropy.time import Time
from reduction.folding import fold

"""
File Params

fn: Name of file to reduce
polyco: Name of polyco file
dm: dispersion measure
dtype: one of vdif, mark4, mark5b, dada
datapars: Associated channel, frequency definitions

Folding parameters

Tint: How long to integrate over
tbin: The size of each time bin
nchan: Number of channels per IF
ngate: Number of phase bins
size: Samples to read in each chunk

"""


fn = 'evn/Ar/gp052a_ar_no0016'
polyco = 'polycob1957+20_ao.dat'
dtype = 'mark4'

sample_rate = 32 * u.MHz
thread_ids = [0, 1]
fedge = 311.25 * u.MHz + (np.array(thread_ids) // 2) * 16. * u.MHz
dm = 29.11680 * 1.00007 * u.pc / u.cm**3

foldtype = 'sp'
Tint = 2*u.s
tbin = 1*u.s
nchan = 256
ngate = 32
size = 2 ** 25

if __name__ == '__main__':

    foldspec, icount = fold(foldtype, fn,polyco,dtype,Tint,tbin,nchan,ngate,size,sample_rate,thread_ids,fedge,dm)

    np.save('FOLD.npy', foldspec)# % (t0.unix), foldspec)
    np.save('IC.npy', icount)# % (t0.unix), ic)

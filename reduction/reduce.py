import argparse
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.io import fits
from reduction.folding import fold
from astropy.extern.configobj import configobj

parser = argparse.ArgumentParser(description='generalized reduction pipeline for scintillometry')
parser.add_argument("-ft", "--foldtype", nargs='?', default='fold')
parser.add_argument("-fn", "--filename")
parser.add_argument("-t0","--tstart", nargs='?')
parser.add_argument("--polyco", nargs='?')
parser.add_argument("--dtype", nargs='?')
parser.add_argument("-dt", "--Tint", type=float)
parser.add_argument("-tb", "--tbin", nargs='?', default=1, type=float)
parser.add_argument("-cn", "--nchan", nargs='?', default=256, type=int)
parser.add_argument("-ng", "--ngate", nargs='?', default=128, type=int)
parser.add_argument("--size", nargs='?', default=2**25, type=int)
a = parser.parse_args()

obs = {}
conf = configobj.ConfigObj(r"obs.conf")
for key, val in conf.iteritems():
    print val
    obs[key] = val

foldspec, icount = fold(a.foldtype, a.filename, a.tstart, a.polyco, a.dtype, a.Tint*u.s, a.tbin*u.s, a.nchan, a.ngate, a.size, **obs)

# Beginning with most basic fits file
n = foldspec / icount[...,np.newaxis]

header = fits.Header(['filename', a.filename])

if a.tstart:
    header.set('T0', a.tstart)
else:
    header.set('T0', 'beginning of file')

header.set('polyco', a.polyco)
header.set('Tint', a.Tint)

for key, val in conf.iteritems():
    header.set('{0}'.format(key), '{0}'.format(val))

hdu = fits.PrimaryHDU(n, header)
hdu.writeto('foldspec_{0}_{1}s_{2}chan_{3}gate.fits'
            .format(a.tstart, a.Tint, a.nchan, a.ngate))

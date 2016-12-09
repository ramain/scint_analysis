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
parser.add_argument("-nc", "--nchan", nargs='?', default=256, type=int)
parser.add_argument("-ng", "--ngate", nargs='?', default=128, type=int)
parser.add_argument("--size", nargs='?', default=2**25, type=int)
parser.add_argument("-dd", "--dedisperse", nargs='?', default='incoherent')
parser.add_argument("-o", "--obsfile", default='obs.conf')
parser.add_argument("-gp", "--pulsedetect", default=False, action='store_true')
a = parser.parse_args()

foldspec, icount = fold(a.foldtype, a.filename, a.obsfile, a.tstart, a.polyco, a.dtype, a.Tint*u.s, a.tbin*u.s, a.nchan, a.ngate, a.size, a.dedisperse, a.pulsedetect)

# Write very basic fits file
# Arrange to [time, pol, freq, phase]
n = foldspec / icount[...,np.newaxis]

if a.foldtype == 'fold':
    n = np.swapaxes(n, 1, 3)
    n = np.swapaxes(n, 2, 3)

header = fits.Header(['filename', a.filename])

if a.tstart:
    header.set('T0', a.tstart)
else:
    header.set('T0', 'beginning of file')

header.set('polyco', a.polyco)
header.set('Tint', a.Tint)
header.set('dedisperse', a.dedisperse)

obs = {}
conf = configobj.ConfigObj(r"{0}".format(a.obsfile))

for key, val in conf.iteritems():
    header.set('{0}'.format(key), '{0}'.format(val))

hdu = fits.PrimaryHDU(n, header)
hdu.writeto('{0}_{1}_{2}s_{3}chan_{4}gate.fits'
            .format(a.foldtype, a.tstart, a.Tint, a.nchan, a.ngate))

import argparse
import numpy as np
import astropy.units as u
from astropy.time import Time
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

n = foldspec / icount[...,np.newaxis]
np.save('Fold.npy', n)

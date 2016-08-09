from __future__ import division
from baseband import vdif
import numpy as np
import astropy.units as u
import glob
import sys
from astropy.time import Time
from pulsar.predictor import Polyco
from mpi4py import MPI

comm = MPI.COMM_WORLD
Csize = comm.Get_size()
rank = comm.Get_rank()

#dm = 555. * u.pc / u.cm**3  # Frb121011
dm = 26.7641 * u.pc / u.cm**3  # b0329
#dm = 71.0227 * u.pc / u.cm**3  # b1937

P = 0.71452 * u.s # b0329
#P = 1.33730216019 * u.s

size = 2 ** 16  # there are 2**16 time samples per file
# not 800 / 1024, since channelized data are complex
sample_rate = 400 / 1024. * u.MHz  
dt1 = 2.56e-6 * u.s

fedge = 800 * u.MHz
dispersion_delay_constant = 4149. * u.s * u.MHz**2 * u.cm**3 / u.pc


### INPUTS (currently hardcoded) ###
T = 60.0*u.s  # Integration Time
ngate = 512  # Number of phase bins
nbin = 10     # Number of time bins, choose such that T / nbin >> P
dedisperse = 'incoherent'
####################################

# Set up frequencies for de-dispersion
freq = np.linspace(800,400,1025)[:-1]*u.MHz

# Pre-calculate de-dispersion delays (coherent phases + incoherent offsets)
if dedisperse == 'by-channel':
    fref = freq
    fcoh = freq - np.fft.fftfreq(size, 1024 / (400*u.MHz))[:, np.newaxis]
    dang = (dispersion_delay_constant * dm * fcoh * (1./fref-1./fcoh)**2) * u.cycle
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        dd_coh = np.exp(dang * 1j).conj().astype(np.complex64);

# Dispersion delays relative to 800MHz
dt = dispersion_delay_constant * dm * (1./freq**2 - 1./(800*u.MHz)**2)


tfile = dt1*size
Nfiles = int(T // tfile + 1)
Npcore = max(Nfiles // Csize, 1)

# Begin at time offset specified in command (seconds from start of obs)
tstart = 0*u.s

dstart = int(tstart // tfile)

# Initialize empty array for foldspec and icounts
f = np.zeros((nbin, ngate, 2, 1024))
ic = np.zeros((nbin, ngate, 2, 1024))

filelist = glob.glob('/mnt/scratch-lustre/simard/B0329_080416/20160805T202001Z_aro_raw/*vdif')
filelist = np.sort(filelist)

# A bit clunky, get starting time from first file in all cores
fn = filelist[0]
fh = vdif.open(fn, mode='rs', sample_rate=sample_rate)
t0 = fh.tell("time")

for i in range(rank*Npcore + dstart, (rank+1)*Npcore + dstart):

    print("rank %s on %s of %s" % (rank, i, (rank+1)*Npcore + dstart))

    print("opening file %s pf %s" % (i, Nfiles))
    fn = filelist[i]
    fh = vdif.open(fn, mode='rs', sample_rate=sample_rate)

    d = fh.read(size)

    if dedisperse == 'by-channel':
        print("by-channel de-dispersing")
        # By-channel coherent de-dispersion to center of channel frequency
        # Only necessary when dispersion smearing > phase bin
        
        # Conjugate from being in second Nyquist zone
        d = np.conj(d)
        d_ft = np.fft.fft(d, axis=0)
        d_ft *= dd_coh[:,np.newaxis]
        d = np.fft.ifft(d_ft, axis=0)

    t = fh.tell("time")
    dt0 = (t - t0).to(u.s)

    # Calculate phase from phase polynomial
    #phasepol = polyco.phasepol(t0, rphase='fraction', t0=t0, time_unit=u.second, convert=True)
    #phase = phasepol(dt0 + np.arange(size) * dt1.to(u.s).value)
    #phase = np.remainder(phase, 1)

    # For slow pulsars, can just use the period
    phase = ((dt0 + np.arange(size) * dt1) / P).value
    phase = np.remainder(phase, 1) * ngate
    phase = phase.astype('int')

    # Put each file into a specific time bin, can cause slight bleed over at edges
    tbin = int((dt0 / T) * nbin)

    print("folding")
    for j in range(ngate):
        p = d[phase==j]
        for k in range(p.shape[0]):
            f[tbin, j] += abs(p[k])**2.0
            ic[tbin, j] += 1

if dedisperse == 'incoherent':
    gate_dt = P / ngate
    incoh_shift = ((dt / gate_dt).value).astype('int')
    # de-disperse along phase axis. Ideally do this on time axis in raw data,
    # bur need enormous chunks of data in memory to do properly
    for j in range(len(incoh_shift)):
        f[...,j] = np.roll(f[...,j], -incoh_shift[j], axis=1)
        ic[...,j] = np.roll(ic[...,j], -incoh_shift[j], axis=1)

f2 = np.zeros_like(f)
ic2 = np.zeros_like(ic)
comm.Reduce(f, f2, op=MPI.SUM, root=0)
comm.Reduce(ic, ic2, op=MPI.SUM, root=0)

if rank == 0:
    print("Comm Reduce...")
    np.save('foldspec_%s_%ss.npy' % (t0.isot, T.value), f2)
    np.save('icount_%s_%ss.npy' % (t0.isot, T.value), ic2)
    print("...done")

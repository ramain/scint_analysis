from __future__ import division

import os
import sys
import glob
import numpy as np
import astropy.units as u
from astropy.time import Time
from pulsar.predictor import Polyco
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

MAX_RMS = 1.5 # max RMS of frequency channels for RFI filter
tsamp = 2.56e-4 * u.second # timestep in seconds
thresh = 6. # detection threshold
buff = 1100 # bins lost to dedispersion at start of waterfall
writeGP = 1

def rfi_filter_raw(raw):
    #Simple RFI filter which zeros channels with high std dev
    #Sum polarizations to get total intensity
    if raw.shape[-1] == 4:
        rawbins = raw[:, :, (0,3)].sum(-1, keepdims=True)
    else:
        rawbins = raw

    std = rawbins.std(axis=0).mean()
    raw_norm = rawbins/std
    std_norm = raw_norm.std(axis=0,keepdims=True)
    ok = std_norm < MAX_RMS
    raw *= ok
    return raw, ok

def rfi_filter_power(power, t0, phase_pol):

    # Detect and store giant pulses in each block through simple S/N test
    if power.shape[-1] == 4:
        freq_av = power[:, :, (0,3)].sum(1).sum(-1)
    else:
        freq_av = power.sum(1).squeeze()

    sn = (freq_av-freq_av.mean()) / freq_av.std()
    # Loop to remove outliers from S/N calculation
    for i in xrange(3):        
        back = np.argwhere(sn < thresh)
        sn = (freq_av-freq_av[back].mean()) / freq_av[back].std()

    peaks = np.argwhere(sn > thresh)

    # Remove duplicates caused by pulse duration
    pdiff = peaks - np.roll(peaks,1)
    peaks = peaks[pdiff != 1]

    f = open('giant_pulses.txt', 'a')

    for peak in peaks:
        tsr = t0 + tsamp * peak
        phase = np.remainder(phase_pol(tsr.mjd), 1)
        f.writelines('{0} {1} {2}\n'.format(tsr.isot, sn[peak], phase))

        if writeGP == 1:
            if peak > 20:
                pulse = power[peak] - power[peak-20:peak-2,:].mean(axis=0)
            else:
                pulse = power[peak] - power[peak+2:peak+20,:].mean(axis=0)
            np.save('/scratch/m/mhvk/ramain/ARO-GPs/GP{0}'.format(tsr.isot),pulse)

        if sn[peak] >= 100:
            print('Bright pulse detected!!!  t={0} with S/N={1}'.format(tsr.isot, sn[peak]) )

    print('{0} Pulses detected'.format(len(peaks)))

    return power, sn

if __name__ == '__main__':
    
    files = np.array(glob.glob('/scratch/m/mhvk/ramain/waterfalls/arochime*waterfall*'))
    nbin = len(files) // size
    files = files[0:nbin*size].reshape(size, -1)
    psr_polyco = Polyco('/home/m/mhvk/ramain/trials/crab-aro/data/polycob0531+21_aro.dat')    

    for i in xrange(files.shape[-1]):
        obs = files[rank,i]
        t0 = Time(obs.split('_')[-1].split('+')[0], format='isot', scale='utc')
        phase_pol = psr_polyco.phasepol(t0)
        print("Rank {0} running pulse finder on {1} ({2}/{3})"
              .format(rank,t0,i,files.shape[-1]))

        w = np.load(obs)

        # Remove edges of waterfall which are lost to de-dispersion
        w=w[buff:-3*buff]
        t0 += buff*tsamp

        w, ok = rfi_filter_raw(w)
        w, sn = rfi_filter_power(w, t0, phase_pol)

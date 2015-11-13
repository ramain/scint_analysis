#!/usr/bin/env python
from __future__ import division, print_function

import sys
import numpy as np
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt

MAX_RMS = 2. # max RMS of frequency channels for RFI filter
tsamp = 2.56e-4 * u.second # timestep in seconds
thresh = 6. # detection threshold
buff = 1100 # bins lost to dedispersion at start of waterfall

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


def rfi_filter_power(power, t0):

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
        f.writelines('time={0} snr={1}\n'.format(tsr.isot, sn[peak]))

        if sn[peak] >= 50:
            print('\nBright pulse detected!!!  t={0} with S/N={1}'.format(tsr.isot, sn[peak]) )

    print('\n{0} Pulses detected'.format(len(peaks)))

    return power, sn

if __name__ == '__main__':
    w = np.load(sys.argv[1])

    t0 = Time(sys.argv[1].split('_')[-1].split('+')[0], format='isot', scale='utc')

    # Remove edges of waterfall which are lost to de-dispersion
    w=w[buff:-3*buff]
    # Quick workaround to ensure times are still correct
    t0 += buff*tsamp

    w, ok = rfi_filter_raw(w)
    w, sn = rfi_filter_power(w, t0)
    plt.plot(sn)
    plt.xlabel('time')
    plt.ylabel('intensity')
    plt.show()

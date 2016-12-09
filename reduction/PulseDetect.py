import numpy as np

def PulseDetect(power, t, phase_pol, thresh=6, outfile='gp.txt'):
    """Flag peaks above S/N threshold"""

    if power.shape == 3:
        power = power.sum(-1)
    freq_av = power.sum(-1)

    sn = (freq_av-freq_av.mean()) / freq_av.std()
    # Loop to remove outliers from S/N calculation
    for i in xrange(3):        
        back = np.argwhere(sn < thresh)
        sn = (freq_av-freq_av[back].mean()) / freq_av[back].std()

    peaks = np.argwhere(sn > thresh)

    f = open(outfile, 'a')

    for peak in peaks:
        t_gp = t[peak]
        phase = np.remainder(phase_pol(t_gp.mjd), 1)
        f.writelines('{0} {1} {2}\n'.format(t_gp.isot, sn[peak], phase))

        if sn[peak] >= 100:
            print('Bright pulse detected!!!  t={0} with S/N={1}'.format(t_gp.isot, sn[peak]) )

    print('{0} Pulses detected'.format(len(peaks)))

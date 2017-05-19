from baseband import mark5b
from baseband import vdif
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.time import Time
from pulsar.predictor import Polyco

#fn = 'ef_data/ek036a_ef_no0020.m5a'
#t_gp = Time('2015-10-19T02:02:36.763878')
#t_gp = Time('2015-10-19T02:03:12.329898')
tel = 'ef'
fn = 'datamount/{0}/ek036a_{0}_no0002.m5a'.format(tel)

size = 2 ** 25
step = 32000000
sample_rate = 32 * u.MHz
dt1 = 1/sample_rate
thread_ids = [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15]
fedge = 1610.49 * u.MHz + ((np.linspace(0,15,16) % 8) // 2) * 32. * u.MHz
fref = fedge.mean() + sample_rate / 4
nchan = 128
tsamp = dt1 * 2 * nchan

thresh = 6

# October DM from JB ephemeris (1e-2 is by eye correction)
dm = (56.7957 + 1e-2) * u.pc / u.cm**3

import pyfftw

print('planning FFT')
a = pyfftw.empty_aligned((size,16), dtype='float32', n=16)
b = pyfftw.empty_aligned((size//2+1,16), dtype='complex64', n=16)

print('...')
fft_object_a = pyfftw.FFTW(a,b, axes=(0,), direction='FFTW_FORWARD',
                planning_timelimit=10.0, threads=8 )
print('...')
fft_object_b = pyfftw.FFTW(b,a, axes=(0,), direction='FFTW_BACKWARD', 
                planning_timelimit=10.0, threads=8 )

def rfi_filter_power(dL, dR, t0):

    power = np.abs(dL)**2.0 + np.abs(dR)**2.0

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

    f = open('giant_pulses.txt', 'a')

    for peak in peaks:
        tsr = t0 + tsamp * peak[0]
        f.writelines('{0} {1}\n'.format(tsr.isot, sn[peak]))
        np.save('GPs/LGP%s.npy' % (tsr.isot), dL[peak-50:peak+50])
        np.save('GPs/RGP%s.npy' % (tsr.isot), dR[peak-50:peak+50])

    print('\n{0} Pulses detected'.format(len(peaks)))
    return power, sn

class DispersionMeasure(u.Quantity):

    dispersion_delay_constant = 4149. * u.s * u.MHz**2 * u.cm**3 / u.pc
    _default_unit = u.pc / u.cm**3

    def __new__(cls, dm, unit=None, dtype=None, copy=True):
        if unit is None:
            unit = getattr(dm, 'unit', cls._default_unit)
        self = super(DispersionMeasure, cls).__new__(cls, dm, unit,
                                                     dtype=dtype, copy=copy)
        if not self.unit.is_equivalent(cls._default_unit):
            raise u.UnitsError("Dispersion measures should have units of "
                               "pc/cm^3")
        return self

    def __quantity_subclass__(self, unit):
        if unit.is_equivalent(self._default_unit):
            return DispersionMeasure, True
        else:
            return super(DispersionMeasure,
                         self).__quantity_subclass__(unit)[0], False

    def __call__(self, f, fref, kind='complex'):
        d = self.dispersion_delay_constant * self
        if kind == 'delay':
            return d * (1./f**2 - 1./fref**2)
        else:
            dang = (d * f * (1./fref-1./f)**2) * u.cycle
            if kind == 'phase':
                return dang
            elif kind == 'complex':
                return np.exp(dang.to(u.rad).value * 1j)

        raise ValueError("kind not one of 'delay', 'phase', or 'complex'")

dm = DispersionMeasure(dm)


if __name__ == '__main__':

    fh = mark5b.open(fn, mode='rs', nchan=16, sample_rate=sample_rate,
                     thread_ids=thread_ids, ref_mjd=57000)

    for i in range(745):
        print('step %s of %s' % (i, 745))
        fh.seek(i*step)
        t0 = fh.tell("time")
        print(t0.isot)

        d = pyfftw.empty_aligned((size,16), dtype='float32')
        d[:] = fh.read(size)
        print('First FFT')
        ft = fft_object_a(d)

        #d = fh.read(size)
        #ft = np.fft.rfft(d, axis=0)

        # Second half of IFs have Fedge at top, need to subtract frequencies, 
        # and not conjugate coherent phases
        f = fedge + np.fft.rfftfreq(d.shape[0], dt1)[:, np.newaxis]
        f[:,8:] = fedge[8:] - np.fft.rfftfreq(d.shape[0], dt1)[:, np.newaxis]
        ft[:,:8] *= dm(f[:,:8], fref, kind='complex').conj()
        ft[:,8:] *= dm(f[:,8:], fref, kind='complex')

        #d = np.fft.irfft(ft, axis=0)

        d = pyfftw.empty_aligned((size//2+1,16), dtype='complex64')
        d[:] = ft
        d = fft_object_b(d)

        # Channelize the data
        dchan = np.fft.rfft(d.reshape(-1, 2*nchan, 16), axis=1)
        # Horribly inelegant, but works for now. 
        # Channels are not in order, and polarizations are separate
        dR = np.concatenate((dchan[:,::-1,8], dchan[...,0], dchan[:,::-1,10], dchan[...,2], dchan[:,::-1,12], dchan[...,4], dchan[:,::-1,14], dchan[...,6]), axis=1)
        dL = np.concatenate((dchan[:,::-1,9], dchan[...,1], dchan[:,::-1,11], dchan[...,3], dchan[:,::-1,13], dchan[...,5], dchan[:,::-1,15], dchan[...,7]), axis=1)
        power, sn = rfi_filter_power(dR, dL, t0)

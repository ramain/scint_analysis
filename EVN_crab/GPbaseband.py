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

fn = 'ef_data/%s/ek036a_%s_no0008.m5a' % (tel, tel)
t_gp = Time('2015-10-19T00:17:47.415')

size = 2 ** 22
sample_rate = 32 * u.MHz
dt1 = 1/sample_rate
thread_ids = [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15]
fedge = 1610.49 * u.MHz + ((np.linspace(0,15,16) % 8) // 2) * 32. * u.MHz
fref = fedge.mean() + sample_rate / 4
nchan = 512
# October DM from JB ephemeris (1e-2 is by eye correction)
dm = (56.7957 + 1e-2) * u.pc / u.cm**3
#dm = 0

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

    if tel == 'sr' or tel == 'mc':
        fh = vdif.open(fn, mode='rs', sample_rate=sample_rate)
    else:
        fh = mark5b.open(fn, mode='rs', nchan=16,
                    sample_rate=sample_rate, thread_ids=thread_ids, ref_mjd=57000)
    offset_gp = ((t_gp - fh.tell(unit='time')).to(u.s).value *
                 fh.frames_per_second * fh.samples_per_frame)
    fh.seek(int(offset_gp) - size // 2)
    d = fh.read(size)
    if tel == 'sr' or tel == 't6' or tel =='tr':
        d = np.roll(d, 8, axis=-1)
    if tel == 'mc' or tel =='sr':
        d = d[:,thread_ids]        
    ft = np.fft.rfft(d, axis=0)
    # Second half of IFs have Fedge at top, need to subtract frequencies, 
    # and not conjugate coherent phases
    f = fedge + np.fft.rfftfreq(d.shape[0], dt1)[:, np.newaxis]
    f[:,8:] = fedge[8:] - np.fft.rfftfreq(d.shape[0], dt1)[:, np.newaxis]
    ft[:,:8] *= dm(f[:,:8], fref, kind='complex').conj()
    ft[:,8:] *= dm(f[:,8:], fref, kind='complex')
    d = np.fft.irfft(ft, axis=0)

    # Channelize the data
    dchan = np.fft.rfft(d.reshape(-1, 2*nchan, 16), axis=1)
    # Horribly inelegant, but works for now. 
    # Channels are not in order, and polarizations are separate
    dR = np.concatenate((dchan[:,::-1,8], dchan[...,0], dchan[:,::-1,10], dchan[...,2], dchan[:,::-1,12], dchan[...,4], dchan[:,::-1,14], dchan[...,6]), axis=1)
    dL = np.concatenate((dchan[:,::-1,9], dchan[...,1], dchan[:,::-1,11], dchan[...,3], dchan[:,::-1,13], dchan[...,5], dchan[:,::-1,15], dchan[...,7]), axis=1)

    dorder = np.concatenate((dchan[...,0],dchan[...,1],dchan[...,2],dchan[...,3],dchan[...,4],dchan[...,5],dchan[...,6],dchan[...,7],dchan[...,8],dchan[...,9],dchan[...,10],dchan[...,11],dchan[...,12],dchan[...,13],dchan[...,14],dchan[...,15]), axis=1)
    power = abs(dR)**2 + abs(dL)**2
    plt.ion()
    plt.imshow((power).T, aspect='auto')
 

from baseband import mark4
import numpy as np
import astropy.units as u
import matplotlib.pylab as plt
from astropy.time import Time

# sshfs bgqdev:/scratch/r/rein/arachkov/DataFromJohnsDir evn

# Filename, time of giant pulse, and amount of samples of data to be read
fn = 'evn/Ar/gp052a_ar_no0006'
t_gp = Time('2014-06-13T06:01:21.91125')
size = 2 ** 25  # 1.048576 s

# Frequency setup for our observation, for data reading and de-dispersion
sample_rate = 32 * u.MHz
dt1 = 1/sample_rate
thread_ids = [0, 1, 2, 3, 4, 5]
fedge = 311.25 * u.MHz + (np.array(thread_ids) // 2) * 16. * u.MHz
fref = fedge.mean() + sample_rate / 4
dm = 29.11680 * 1.00007 * u.pc / u.cm**3
nchan = 256

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
    fh = mark4.open(fn, mode='rs', decade=2010, ntrack=64,
                    sample_rate=sample_rate, thread_ids=thread_ids)

    # Compute offset to have giant pulse in middle of data chunk
    offset_gp = ((t_gp - fh.tell(unit='time')).to(u.s).value *
                 fh.frames_per_second * fh.samples_per_frame)

    # Go to that offset, read "size" number of samples
    fh.seek(int(offset_gp) - size // 2)
    d = fh.read(size)

    # For coherent de-dispersion, FT to frequency space 
    # apply de-dispersion delay as phase rotation
    ft = np.fft.rfft(d, axis=0)
    f = fedge + np.fft.rfftfreq(d.shape[0], dt1)[:, np.newaxis]
    ft *= dm(f, fref, kind='complex').conj()
    d = np.fft.irfft(ft, axis=0)

    # Channelize to your desired number of channels
    dchan = np.fft.rfft(d.reshape(-1, 2*nchan, 6), axis=1)

    # Split into R and L circular polarization, append IFs together
    dR = np.concatenate((dchan[:,:nchan,1], dchan[:,:nchan,3], dchan[:,:nchan,5]), axis=1)
    dL = np.concatenate((dchan[:,:nchan,0], dchan[:,:nchan,2], dchan[:,:nchan,4]), axis=1)

    # Compute power
    power = abs(dL)**2.0 + abs(dR)**2.0

    #plt.ion()

    plt.imshow(power.T, aspect='auto', interpolation='nearest')
    plt.show()

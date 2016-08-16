from baseband import mark4
from baseband import vdif
import numpy as np
import astropy.units as u
import matplotlib.pylab as plt
from astropy.time import Time

# sshfs bgqdev:/scratch/r/rein/arachkov/DataFromJohnsDir evn
fn = 'evn/Ar/gp052a_ar_no0006'
t_gp = Time('2014-06-13T06:01:21.91125')
#t_gp = Time('2014-06-13T07:00:02.13597')
#fn = 'evn/Ar/gp052a_ar_no0015'
#t_gp = Time('2014-06-13T06:50:42.3')
fn = 'evn/Ar/gp052a_ar_no0016'
t_gp = Time('2014-06-13T07:09:27.16265')
size = 2 ** 25  # 1.048576 s
sample_rate = 32 * u.MHz
dt1 = 1/sample_rate
thread_ids = [0, 1, 2, 3, 4, 5]
fedge = 311.25 * u.MHz + (np.array(thread_ids) // 2) * 16. * u.MHz
fref = fedge.mean() + sample_rate / 4
dm = 29.11680 * 1.00007 * u.pc / u.cm**3
nchan = 256

import pyfftw

print('planning FFT')
a = pyfftw.empty_aligned((size,6), dtype='float32', n=16)
b = pyfftw.empty_aligned((size//2+1,6), dtype='complex64', n=16)
c1 = pyfftw.empty_aligned((size//(2*nchan), 2*nchan, 6), dtype='float32', n=16)
c2 = pyfftw.empty_aligned((size//(2*nchan), nchan+1, 6), dtype='complex64', n=16)

print('...')
fft_object_a = pyfftw.FFTW(a,b, axes=(0,), direction='FFTW_FORWARD',
                planning_timelimit=1.0, threads=8 )
print('...')
fft_object_b = pyfftw.FFTW(b,a, axes=(0,), direction='FFTW_BACKWARD', 
                planning_timelimit=1.0, threads=8 )
print('...')
fft_object_c = pyfftw.FFTW(c1,c2, axes=(1,), direction='FFTW_FORWARD',
                planning_timelimit=1.0, threads=8 )

f = fedge + np.fft.rfftfreq(size, dt1)[:, np.newaxis]

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


print("Pre-Calculating De-Dispersion Phases")
dm = DispersionMeasure(dm)
dd = dm(f, fref, kind='complex').conj()

if __name__ == '__main__':
    print('Reading...')
    fh = mark4.open(fn, mode='rs', decade=2010, ntrack=64,
                    sample_rate=sample_rate, thread_ids=thread_ids)
    offset_gp = ((t_gp - fh.tell(unit='time')).to(u.s).value *
                 fh.frames_per_second * fh.samples_per_frame)
    fh.seek(int(offset_gp) - size // 2)

    d = pyfftw.empty_aligned((2**25,6), dtype='float32')
    d[:] = fh.read(size)
    print('First FFT')
    ft = fft_object_a(d)
    print('De-Dispersing')
    ft *= dd
    print('Second FFT')
    d = pyfftw.empty_aligned((2**24+1,6), dtype='complex64')
    d[:] = ft
    data = fft_object_b(d)
    plt.ion()
    print('Channelize and form power')
    dchan = fft_object_c(data.reshape(-1, 2*nchan, 6))
    #dchan = np.fft.rfft(data.reshape(-1, 2*nchan, 6), axis=1)
    dR = np.concatenate((dchan[:,:nchan,1], dchan[:,:nchan,3], dchan[:,:nchan,5]), axis=1)
    dL = np.concatenate((dchan[:,:nchan,0], dchan[:,:nchan,2], dchan[:,:nchan,4]), axis=1)
    power = abs(dL)**2.0 + abs(dR)**2.0


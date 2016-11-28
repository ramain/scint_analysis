from baseband import mark4
import numpy as np
import astropy.units as u
import matplotlib.pylab as plt
import matplotlib.cm as cm
from astropy.time import Time
from pulsar.predictor import Polyco
from reduction.dm import DispersionMeasure
import pyfftw

fn = 'evn/Ar/gp052a_ar_no0016'
psr_polyco = Polyco('polycob1957+20_ao.dat')
size = 2 ** 25  # 2**25 = 1.048576 s 
sample_rate = 32 * u.MHz
#thread_ids = [0, 1, 2, 3, 4, 5, 6, 7]
thread_ids = [0, 1]
fedge = 311.25 * u.MHz + (np.array(thread_ids) // 2) * 16. * u.MHz
dm = 29.11680 * 1.00007 * u.pc / u.cm**3

"""
Folding parameters

Tint: How long to integrate over
tbin: The size of each time bin
nchan: Number of channels per IF
ngate: Number of phase bins

"""

Tint = 10*u.s
tbin = 2*u.s
nchan = 8
ngate = 256

# Derived values for folding

dt1 = 1/sample_rate
f = fedge + np.fft.rfftfreq(size, dt1)[:, np.newaxis]
fref = fedge + sample_rate // 2 # Reference to top of band
ntbin = int((Tint / tbin).value)
npol = len(thread_ids)

print("Pre-Calculating De-Dispersion Values")
dm = DispersionMeasure(dm)
dmloss = max( dm.time_delay(fref-sample_rate//2, fref) )
samploss = int(np.ceil( (dmloss * sample_rate).decompose() ).value)

# Step is reduced by DM losses, rounded to nearest power of 2
step = int(size -  2**(np.ceil(np.log2(samploss))))
Tstep = int(np.ceil( (Tint / (step*dt1)).decompose() ))

print("{0} and {1} samples lost to de-dispersion".format(dmloss, samploss))
print("Taking blocks of {0}, steps of {1} samples".format(size, step))

dd = dm.phase_factor(f, fref).conj()

print('planning FFT')
a = pyfftw.empty_aligned((size,npol), dtype='float32', n=16)
b = pyfftw.empty_aligned((size//2+1,npol), dtype='complex64', n=16)
c1 = pyfftw.empty_aligned((step//(2*nchan), 2*nchan, npol), dtype='float32', n=16)
c2 = pyfftw.empty_aligned((step//(2*nchan), nchan+1, npol), dtype='complex64', n=16)

print('...')
fft_object_a = pyfftw.FFTW(a,b, axes=(0,), direction='FFTW_FORWARD',
                planning_timelimit=10.0, threads=8 )
print('...')
fft_object_b = pyfftw.FFTW(b,a, axes=(0,), direction='FFTW_BACKWARD', 
                planning_timelimit=10.0, threads=8 )
print('...')
fft_object_c = pyfftw.FFTW(c1,c2, axes=(1,), direction='FFTW_FORWARD',
                planning_timelimit=10.0, threads=8 )


if __name__ == '__main__':

    fh = mark4.open(fn, mode='rs', decade=2010, ntrack=64,
                        sample_rate=sample_rate, thread_ids=thread_ids)

    t0 = fh.tell(unit='time')
    phase_pol = psr_polyco.phasepol(t0)
    phase0 = phase_pol(t0.mjd)
    phase1 = phase_pol((t0+Tstep*step*dt1).mjd)
    ncycle = int(np.ceil(phase1-phase0))

    for i in range(Tstep):
        print('On step {0} of {1}'.format(i, Tstep))
        print('Reading...')
        fh.seek(step*i)
        t0 = fh.tell(unit='time')
        if i == 0:
            print('starting at {0}'.format(t0.isot))

        phase_pol = psr_polyco.phasepol(t0)

        d = pyfftw.empty_aligned((size,npol), dtype='float32')
        d[:] = fh.read(size)

        print('First FFT')
        ft = fft_object_a(d)

        print('De-Dispersing')
        ft *= dd

        print('Second FFT')
        d = pyfftw.empty_aligned((size//2+1,npol), dtype='complex64')
        d[:] = ft
        data = fft_object_b(d)

        print('Channelize and form power')
        dchan = fft_object_c(data[:step].reshape(-1, 2*nchan, npol))
        power = (np.abs(dchan)**2).sum(1)

        print("Folding")
        tsamp = (2 * nchan / sample_rate).to(u.s)
        tsr = t0 + tsamp * np.arange(power.shape[0])
        phase = phase_pol(tsr.mjd)

        phase -= phase0
        ncycle = int(np.ceil(phase[-1] - phase[0]))
        iphase = np.remainder(phase*ngate, ncycle*ngate).astype(np.int)

        if i == 0:
            foldspec = np.zeros((ncycle*Tstep, ngate, npol))
            ic = np.zeros((ncycle*Tstep, ngate))   

        for pol in range(npol):
            foldspec[i*ncycle:(i+1)*ncycle, :, pol] += np.bincount(iphase, power[..., pol], minlength=ngate*ncycle).reshape(ncycle, ngate)
        ic[i*ncycle:(i+1)*ncycle] += np.bincount(iphase, power[..., 0] != 0, minlength=ngate*ncycle).reshape(ncycle, ngate)
                
        
        #ibin = np.floor( (ntbin * tsamp * (i * power.shape[0] + np.arange(power.shape[0]) ) 
        #                  // Tint).decompose() ).astype('int')
        #time_bins = np.unique(ibin)

        # Beautiful triply embeded for loop for folding
        #for bin in time_bins:
        #    pfold = power[ibin==bin]
        #    phasefold = phase[ibin==bin]
        #    for pol in range(npol):
        #        for kfreq in range(nchan):
        #            foldspec[bin,kfreq,:,pol] += np.bincount(phasefold, pfold[:,kfreq,pol], ngate)
        #            ic[bin,kfreq,:,pol] += np.bincount(phasefold, pfold[:,kfreq,pol] != 0., ngate)


    #np.save('FOLD{0}_{0}.npy' % (t0.unix), foldspec)
    #np.save('IC{0}_{0}.npy' % (t0.unix), ic)

    ic[ic==0] = 1
    n = foldspec / ic[...,np.newaxis]
    plt.ion()
    plt.imshow(n.sum(-1) - np.median(n.sum(-1), axis=-1, keepdims=True), aspect='auto', interpolation='nearest', cmap=cm.Greys, vmin=-200)

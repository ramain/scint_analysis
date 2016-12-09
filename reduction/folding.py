import numpy as np
import pyfftw
import astropy.units as u
from astropy.time import Time
from pulsar.predictor import Polyco
from reduction.dm import DispersionMeasure
from reduction.ReadDD import ReadDD as RD
from reduction.PulseDetect import PulseDetect

def fold(foldtype, fn, obsfile, tstart, polyco, dtype, Tint, tbin, nchan, ngate, size, dedisperse, pulsedetect):

    """
    Parameters
    ----------
    foldtype : string
        One of 'fold', 'sp'
    fn : file handle
        handle to file holding voltage timeseries
    tstart : astropy Time
        beginning time to reduce, defaults to start of file
    polyco : string
        path to file containing polyco solution for timing
    dtype : string
        one of 'vdif', 'mark4', 'mark5b'
    nchan : int
        number of frequency channels for FFT
    tbin : float
        size of time bins in seconds
    ngate : int
        number of phase bins to use for folded spectrum
        ntbin should be an integer fraction of nt
    size: int
        number of samples to reduce in each step
    dedisperse: string
        one of 'coherent', 'incoherent'
    **obs: parameters read from obs.conf
    """

    fh = RD(fname=fn, obsfile=obsfile, size=size)

    psr_polyco = Polyco(polyco)

    # Derived values for folding
    dt1 = fh.dt1
    sample_rate = fh.sample_rate
    ntbin = int((Tint / tbin).value)
    npol = len(fh.thread_ids)

    t0 = fh.fh.tell('time')
    if not tstart:
        tstart = t0
    else:
        tstart = Time(tstart)

    print("File begins at {0}, beginning at {1}".format(t0.isot, tstart.isot))
    offset = int( np.floor( ((tstart - t0) / dt1).decompose() ).value )
    print("Offset {0} samples from start of file".format(offset))

    # Step is reduced by DM losses, rounded to nearest power of 2
    step = fh.step
    Tstep = int(np.ceil( (Tint / (step*dt1)).decompose() ))

    print("Taking blocks of {0}, steps of {1} samples".format(size, step))

    if foldtype == 'fold':
        foldspec = np.zeros((ntbin, nchan, ngate, npol))
        ic = np.zeros((ntbin, nchan, ngate))
    
    # Folding loop
    for i in range(Tstep):
        print('On step {0} of {1}'.format(i, Tstep))
        print('Reading...')
        fh.seek(offset + step*i)
        t0 = fh.fh.tell('time')
        if i == 0:
            print('starting at {0}'.format(t0.isot))

        phase_pol = psr_polyco.phasepol(t0)

        if dedisperse == 'coherent':
            data = fh.readCoherent(size)
            dchan = np.fft.rfft(data.reshape(-1, 2*nchan, npol), axis=1)
            del data
        elif dedisperse == 'incoherent':
            dchan = fh.readIncoherent(size, nchan)

        power = (np.abs(dchan)**2)
        print("Folding")
        tsamp = (2 * nchan / sample_rate).to(u.s)
        tsr = t0 + tsamp * np.arange(power.shape[0])   

        if pulsedetect:
            PulseDetect(power, tsr, phase_pol)

        if foldtype == 'fold':
            phase = (np.remainder(phase_pol(tsr.mjd),1) * ngate).astype('int')
            ibin = np.floor( (ntbin * tsamp * (i * power.shape[0] + np.arange(power.shape[0]) ) // Tint).decompose() ).astype('int')
            time_bins = np.unique(ibin)
            time_bins = time_bins[time_bins<ntbin]

            # Beautiful triply embeded for loop for folding
            for bin in time_bins:
                pfold = power[ibin==bin]
                phasefold = phase[ibin==bin]
                for kfreq in range(nchan):
                    for pol in range(npol):
                        foldspec[bin,kfreq,:,pol] += np.bincount(phasefold, pfold[:,kfreq,pol], ngate)
                    ic[bin,kfreq,:] += np.bincount(phasefold, pfold[:,kfreq,0] != 0., ngate)

        if foldtype == 'sp':
            pfold = power.sum(1)
            phase = phase_pol(tsr.mjd)
            ncycle = int(np.ceil(phase[-1] - phase[0]))

            if i == 0:
                foldspec = np.zeros((ncycle*Tstep, ngate, npol))
                ic = np.zeros((ncycle*Tstep, ngate))
                phase0 = phase[0]

            phase -= phase0
            iphase = np.remainder(phase*ngate, ncycle*ngate).astype(np.int)
 
            for pol in range(npol):
                foldspec[i*ncycle:(i+1)*ncycle, :, pol] += np.bincount(iphase, pfold[..., pol], minlength=ngate*ncycle).reshape(ncycle, ngate)
            ic[i*ncycle:(i+1)*ncycle] += np.bincount(iphase, pfold[..., 0] != 0, minlength=ngate*ncycle).reshape(ncycle, ngate)

    ic[ic==0] = 1
    return foldspec, ic

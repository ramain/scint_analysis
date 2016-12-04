import numpy as np
import pyfftw
import astropy.units as u
from astropy.time import Time
from pulsar.predictor import Polyco
from reduction.dm import DispersionMeasure
from baseband import mark4, mark5b, vdif, dada

def fold(foldtype, fn, tstart, polyco, dtype, Tint, tbin, nchan, ngate, size, dedisperse, **obs):

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

    # Read from obs.conf
    sample_rate =  float(obs["srate"]) * u.MHz
    dm = float(obs["dm"]) * u.pc / u.cm**3
    thread_ids = list(obs["threads"])
    fedge = np.array(obs["fedge"]).astype('float') * u.MHz

    psr_polyco = Polyco(polyco)

    # Derived values for folding
    dt1 = 1/sample_rate
    fref = max(fedge) + sample_rate // 2 # Reference to top of band
    ntbin = int((Tint / tbin).value)
    npol = len(thread_ids)

    if dtype == 'mark4':
        fh = mark4.open(fn, mode='rs', decade=2010, ntrack=64,
                        sample_rate=sample_rate, thread_ids=thread_ids)
    elif dtype == 'mark5b':
        nIF = int(obs["nIF"])
        fh = mark5b.open(fn, mode='rs', nchan=nIF,
                    sample_rate=sample_rate, thread_ids=thread_ids, ref_mjd=57000)
    elif dtype == 'vdif':
        fh = vdif.open(fn, mode='rs', sample_rate=sample_rate)

    t0 = fh.tell('time')
    if not tstart:
        tstart = t0
    else:
        tstart = Time(tstart)

    print("File begins at {0}, beginning at {1}".format(t0.isot, tstart.isot))
    offset = int( np.floor( ((tstart - t0) / dt1).decompose() ).value )
    print("Offset {0} samples from start of file".format(offset))

    print("Pre-Calculating De-Dispersion Values")
    dm = DispersionMeasure(dm)
    dmloss = dm.time_delay(min(fedge), fref)
    samploss = int(np.ceil( (dmloss * sample_rate).decompose() ).value)

    # Step is reduced by DM losses, rounded to nearest power of 2
    step = int(size -  2**(np.ceil(np.log2(samploss))))
    Tstep = int(np.ceil( (Tint / (step*dt1)).decompose() ))

    print("{0} and {1} samples lost to de-dispersion".format(dmloss, samploss))
    print("Taking blocks of {0}, steps of {1} samples".format(size, step))

    if dedisperse == 'coherent':
        print('Planning coherent DD FFT')
        f = fedge + np.fft.rfftfreq(size, dt1)[:, np.newaxis]
        dd = dm.phase_factor(f, fref).conj()
        a = pyfftw.empty_aligned((size,npol), dtype='float32', n=16)
        b = pyfftw.empty_aligned((size//2+1,npol), dtype='complex64', n=16)
        fft_object_a = pyfftw.FFTW(a,b, axes=(0,), direction='FFTW_FORWARD',
                           planning_timelimit=10.0, threads=8 )
        fft_object_b = pyfftw.FFTW(b,a, axes=(0,), direction='FFTW_BACKWARD', 
                           planning_timelimit=10.0, threads=8 )
    
    elif dedisperse == 'incoherent':
        f = fedge + np.fft.rfftfreq(2*nchan, dt1)[:, np.newaxis]
        dm_delay = dm.time_delay(f, fref)
        dm_sample = np.floor( (dm_delay / (2*nchan*dt1)).decompose()).value.astype('int')

    c1 = pyfftw.empty_aligned((size//(2*nchan), 2*nchan, npol), dtype='float32', n=16)
    c2 = pyfftw.empty_aligned((size//(2*nchan), nchan+1, npol), dtype='complex64', n=16)

    print("planning FFT for channelization")
    fft_object_c = pyfftw.FFTW(c1,c2, axes=(1,), direction='FFTW_FORWARD',
                           planning_timelimit=10.0, threads=8 )

    if foldtype == 'fold':
        foldspec = np.zeros((ntbin, nchan, ngate, npol))
        ic = np.zeros((ntbin, nchan, ngate))
    
    # Folding loop
    for i in range(Tstep):
        print('On step {0} of {1}'.format(i, Tstep))
        print('Reading...')
        fh.seek(offset + step*i)
        t0 = fh.tell(unit='time')
        if i == 0:
            print('starting at {0}'.format(t0.isot))

        phase_pol = psr_polyco.phasepol(t0)

        data = pyfftw.empty_aligned((size,npol), dtype='float32')
        data[:] = fh.read(size)[...,thread_ids]
        
        if dedisperse == 'coherent':
            print('First FFT')
            ft = fft_object_a(data)
            print('Applying De-Dispersion Phases')
            ft *= dd
            print('Second FFT')
            d = pyfftw.empty_aligned((size//2+1,npol), dtype='complex64')
            d[:] = ft
            data = fft_object_b(d)

        print('Channelize and form power')
        dchan = fft_object_c(data.reshape(-1, 2*nchan, npol))
        if dedisperse == 'incoherent':
            for chan in range(nchan):
                for pol in range(npol):
                    dchan[:,chan,pol] = np.roll(dchan[:,chan,pol], -dm_sample[chan,pol], axis=0)

        power = (np.abs(dchan[:step//(2*nchan)])**2)
        print("Folding")
        tsamp = (2 * nchan / sample_rate).to(u.s)
        tsr = t0 + tsamp * np.arange(power.shape[0])   
        
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

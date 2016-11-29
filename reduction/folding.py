import numpy as np
import astropy.units as u
import matplotlib.pylab as plt
import matplotlib.cm as cm
import pyfftw
from astropy.time import Time
from pulsar.predictor import Polyco
from reduction.dm import DispersionMeasure
from baseband import mark4, mark5b, vdif, dada

def fold(foldtype, fn, polyco, dtype, dm, Tint, tbin, nchan, ngate, size, sample_rate, thread_ids, nIF, fedge):

    """
    Folding is done from the position the file is currently in

    Parameters
    ----------
    fn : file handle
        handle to file holding voltage timeseries
    comm: MPI communicator or None
        will use size, rank attributes
    samplerate : Quantity
        rate at which samples were originally taken and thus double the
        band width (frequency units)
    fedge : float
        edge of the frequency band (frequency units)
    fedge_at_top: bool
        whether edge is at top (True) or bottom (False)
    nchan : int
        number of frequency channels for FFT
    nt, ntint : int
        total number nt of sets, each containing ntint samples in each file
        hence, total # of samples is nt*ntint, with each sample containing
        a single polarisation
    ngate, ntbin : int
        number of phase and time bins to use for folded spectrum
        ntbin should be an integer fraction of nt
    ntw : int
        number of time samples to combine for waterfall (does not have to be
        integer fraction of nt)
    dm : float
        dispersion measure of pulsar, used to correct for ism delay
        (column number density)
    fref: float
        reference frequency for dispersion measure
    phasepol : callable
        function that returns the pulsar phase for time in seconds relative to
        start of the file that is read.

    """

    psr_polyco = Polyco(polyco)

    # Derived values for folding
    dt1 = 1/sample_rate
    f = fedge + np.fft.rfftfreq(size, dt1)[:, np.newaxis]
    fref = max(fedge) + sample_rate // 2 # Reference to top of band
    ntbin = int((Tint / tbin).value)
    npol = len(thread_ids)

    print(fref)

    print("Pre-Calculating De-Dispersion Values")
    dm = DispersionMeasure(dm)
    dmloss = dm.time_delay(fref-sample_rate//2, fref)
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

    if dtype == 'mark4':
        fh = mark4.open(fn, mode='rs', decade=2010, ntrack=64,
                        sample_rate=sample_rate, thread_ids=thread_ids)
    elif dtype == 'mark5b':
        fh = mark5b.open(fn, mode='rs', nchan=nIF,
                    sample_rate=sample_rate, thread_ids=thread_ids, ref_mjd=57000)


    if foldtype == 'fold':
        foldspec = np.zeros((ntbin, nchan, ngate, npol))
        ic = np.zeros((ntbin, nchan, ngate))
    
    # Folding loop
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
        power = (np.abs(dchan)**2)
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

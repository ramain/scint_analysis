import numpy as np
import astropy.units as u
import pyfftw
from reduction.dm import DispersionMeasure
from astropy.extern.configobj import configobj
from baseband import vdif, mark4, mark5b, dada

class ReadDD:
    """Reader class to return de-dispersed data directy from raw binaries

    Requires a configuration file with data type and frequency information
    of your observation

    Parameters
    ----------
    fname: filename of binary file to be opened
    obsfile: filename of config file for observation
    requires
        srate : int 
            sample_rate in MHz
        dtype : string
            one of 'vdif', 'mark4', 'mark5b', 'dada'
        dm : float
            dispersion measure, in pc/cm**3
        threads:
            IFs to be read
        fedge: list, int
            edge frequencies of IFs in MHz
    optional
        forder: list, int
            list of 1 or -1 indicating upper or lower sideband
            default assumption is upper
        ntrack: int
            ntrack parameter for mark4 data, default = 64
        nIF: int
            total number of IFs, required to open mark5b data
    size: int
        number of samples to read
    """

    def __init__(self, fname=None, obsfile=None, size=2**25):

        obs = {}
        conf = configobj.ConfigObj(r"{0}".format(obsfile))

        for key, val in conf.iteritems():
            obs[key] = val
        
        self.dtype = obs["dtype"]
        self.sample_rate =  float(obs["srate"]) * u.MHz
        dm = float(obs["dm"]) * u.pc / u.cm**3
        self.dm = DispersionMeasure(dm)
        self.thread_ids = list(obs["threads"])
        self.fedge = np.array(obs["fedge"]).astype('float') * u.MHz

        if 'forder' in obs:
            self.forder = np.array(obs["forder"]).astype('int')
        else:
            self.forder = np.ones(len(self.fedge))
        if 'ntrack' in obs:
            self.ntrack = int(obs["ntrack"])
        else:
            self.ntrack = 64
        if 'nIF' in obs:
            self.nIF = int(obs["nIF"])
 
        self.dt1 = 1/self.sample_rate
        self.size = size
        self.fref = max(self.fedge) + self.sample_rate // 2
        self.f = self.fedge + self.forder*np.fft.rfftfreq(size, self.dt1)[:, np.newaxis]
        self.dmloss = self.dm.time_delay(min(self.fedge), self.fref)
        self.samploss = int(np.ceil( (self.dmloss * self.sample_rate).decompose() ).value)
        self.step = int(size -  2**(np.ceil(np.log2(self.samploss))))
        self.npol = len(self.thread_ids)

        print("{0} and {1} samples lost to de-dispersion".format(self.dmloss, self.samploss))
        print("Taking blocks of {0}, steps of {1} samples".format(self.size, self.step))

        if fname:
            self.open(fname)


    def open(self, fname):
        """Open data with appropriate baseband reader"""

        if self.dtype == 'vdif':
            self.fh = vdif.open(fname, mode='rs', sample_rate=self.sample_rate)

        if self.dtype == 'mark4':
            self.fh = mark4.open(fname, mode='rs', decade=2010, ntrack=self.ntrack,
                                 sample_rate=self.sample_rate, thread_ids=self.thread_ids)

        if self.dtype == 'mark5b':
            self.fh = mark5b.open(fname, mode='rs', nchan=self.nIF, ref_mjd=57000,
                                  sample_rate=self.sample_rate, thread_ids=self.thread_ids)

    def seek(self, nsamples):
        """Seek as in a baseband stream"""
        self.fh.seek(nsamples)

    def readCoherent(self, size=2**25):
        """Return coherently dedispersed timestream

        Read size number of samples, coherently dedisperse, take first
        step samples to chop off wraparound
        """

        if size != self.size or not 'dd' in self.__dict__ :
            # Only compute dedispersion phases and fft plans once for given size
            print("Calculating de-dispersion phase factors for size {0}".format(size))
            self.f = self.fedge + self.forder*np.fft.rfftfreq(size, self.dt1)[:, np.newaxis]
            self.dd = self.dm.phase_factor(self.f, self.fref)
            for j in range(len(self.forder)):
                if self.forder[j] == 1:
                    self.dd[...,j] = np.conj(self.dd[...,j])
            self.size = size
            self.step = int(size -  2**(np.ceil(np.log2(self.samploss))))
            a = pyfftw.empty_aligned((self.size, self.npol), dtype='float32', n=16)
            b = pyfftw.empty_aligned((self.size//2+1, self.npol), dtype='complex64', n=16)
            print("planning FFTs for coherent dedispersion...")
            self.fft_ts = pyfftw.FFTW(a,b, axes=(0,), direction='FFTW_FORWARD',
                           planning_timelimit=1.0, threads=8 )
            print("...")
            self.ifft_ts = pyfftw.FFTW(b,a, axes=(0,), direction='FFTW_BACKWARD',
                           planning_timelimit=1.0, threads=8 )


        d = pyfftw.empty_aligned((size,self.npol), dtype='float32')
        #d = self.fh.read(size)

        # need better solution...
        if self.dtype == 'vdif':
            d[:] = self.fh.read(size)[self.thread_ids]
        else:
            d[:] = self.fh.read(size)

        #ft = np.fft.rfft(d, axis=0)
        ft = self.fft_ts(d)
        ft *= self.dd

        dift = pyfftw.empty_aligned((size//2+1,self.npol), dtype='complex64')
        dift[:] = ft
        d = self.ifft_ts(dift)
        #d = np.fft.irfft(ft, axis=0)[:self.step]
        return d

    def readIncoherent(self, size=2**25, nchan=512):
        """Return incoherently dedispersed, channelized data

        Read size number of samples, channelize to nchan, incoherently 
        dedisperse, take first step samples to chop off wraparound
        """

        self.f = self.fedge + self.forder*np.fft.rfftfreq(2*nchan, self.dt1)[:, np.newaxis]
        npol = len(self.fh.thread_ids)
        dm_delay = self.dm.time_delay(self.f, self.fref)
        dm_sample = np.floor( (dm_delay / (2*nchan*self.dt1)).decompose()).value.astype('int')

        d = self.fh.read(size)
        dchan = np.fft.rfft(d.reshape(-1, 2*nchan, npol), axis=1)
        for chan in range(nchan):
            for pol in range(npol):
                dchan[:,chan,pol] = np.roll(dchan[:,chan,pol], -dm_sample[chan,pol], axis=0)
        return dchan[:(self.step//(2*nchan))]

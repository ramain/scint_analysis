import numpy as np
import pfb
from .vdifSeq import vdifSeq
from pyfftw.interfaces.numpy_fft import rfft, irfft

class InvPFB(vdifSeq):

    def __init__(self, flist, sample_rate):
        """ARO data acquired with a CHIME correlator, PFB inverted.

        The PFB inversion is imperfect at the edges. To do this properly,
        need to read in multiple blocks at a time (not currently implemented).

        Also, this will ideally be read as sets of 2048 samples 
        (ie: read as dtype (2048,)4bit: 1024)
        """

        vdifSeq.__init__(self, flist, sample_rate)

        self.nblock = 2048
        self.h = pfb.sinc_hamming(4, self.nblock).reshape(4, -1)
        self.npol = 2

        # S/N for use in the Wiener Filter
        # Assume 8 bits are set to have noise at 3 bits, so 1.5 bits for FT.
        # samples off by uniform distribution of [-0.5, 0.5] ->
        # equivalent to adding noise with std=0.2887
        prec = (1 << 3) ** 0.5
        self.sn = prec / 0.2887

    def read(self, samples):

        raw = self.fh.read(samples)
        raw = np.swapaxes(raw, 1, 2)

        nyq_pad = np.zeros((raw.shape[0], 1, self.npol), dtype=raw.dtype)
        raw = np.concatenate((raw, nyq_pad), axis=1)
 
        # Get pseudo-timestream
        pd = irfft(raw, axis=1)
        # Set up for deconvolution
        fpd = rfft(pd, axis=0)
        del pd

        lh = np.zeros((raw.shape[0], self.h.shape[1]))
        lh[:self.h.shape[0]] = self.h
        fh = rfft(lh, axis=0).conj()
        del lh
        
        # FT of Wiener deconvolution kernel
        fg = fh.conj() / (np.abs(fh)**2 + (1/self.sn)**2)
        # Deconvolve and get deconvolved timestream
        rd = irfft(fpd * fg[..., np.newaxis],
                          axis=0).reshape(-1, self.npol)

        # view as a record array
        return rd.astype('f4')

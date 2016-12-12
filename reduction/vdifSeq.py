import numpy as np
from baseband import vdif

class vdifSeq:
    """class to seek and read freely from a series of 
    contiguous vdif files
    """
    def __init__(self, flist, sample_rate, filesamp=2**16):
        self.flist = flist
        self.filesamp = filesamp
        self.findex = 0
        self.offset = 0
        self.sample_rate = sample_rate
        self.open()

    def open(self):
        self.fh = vdif.open(self.flist[self.findex], 'rs', sample_rate=self.sample_rate)

    def seek(self, samples):
        self.findex = samples // self.filesamp
        self.offset = np.remainder(samples, self.filesamp)
        self.open()
        self.fh.seek(self.offset)

    def read(self, samples):
        samp_rem = self.filesamp - self.offset
        if samples <= samp_rem:
            d = self.fh.read(samples)
            self.offset += samples

        if samples > samp_rem:
            nfiles = int(1 + np.ceil((samples-samp_rem)/self.filesamp))
            d = self.fh.read(samp_rem)
            self.findex += 1
            samples -= samp_rem

            for i in range(nfiles):
                self.open()
                if samples > self.filesamp:
                    d2 = self.fh.read(self.filesamp)
                    self.findex += 1
                    d = np.concatenate((d,d2), axis=0)
                if samples <= self.filesamp:
                    d2 = self.fh.read(samples)
                    self.offset += samples
                    d = np.concatenate((d,d2), axis=0)

        if self.offset == self.filesamp:
            self.offset = 0
            self.findex += 1
            self.open()

        return d

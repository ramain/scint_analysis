import numpy as np
from reduction.dm import DispersionMeasure

class ReadCoherent:

    def __init__(self, fname, sample_rate, fedge, thread_ids, dm, size=2**24):
        self.sample_rate = sample_rate
        self.dt1 = 1/sample_rate
        self.fedge = fedge
        self.thread_ids = thread_ids
        self.dm = DispersionMeasure(dm)
        self.size = size
        self.fref = max(fedge) + samplerate // 2
        self.f = fedge + np.fft.rfftfreq(size, self.dt1)[:, np.newaxis]
        self.dd = self.dm.phase_factor(self.f, self.fref).conj()
        if fname:
            self.mark4open(fname)
        else:
            self.ts = None

    def mark4open(self, fname):
        from baseband import mark4
        fh = mark4.open(fname, mode='rs', decade=2010, ntrack=64,
                        sample_rate=self.samplerate, thread_ids=self.thread_ids)
        d = fh.read(self.size)
        ft = np.fft.rfft(d, axis=0)
        ft *= self.dd
        d = np.fft.irfft(ft, axis=0)
        del ft
        self.ts = d

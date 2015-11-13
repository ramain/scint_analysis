#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if len(sys.argv) != 2:
    print "Usage: %s dynspec" % sys.argv[0]
    sys.exit(1)

dyn = np.load(sys.argv[1]) # Dynamic spectrum

if sys.argv[1] == 'sim_dspec.npy':
    dyn = np.real(dyn)

#Optional binning
#dyn=dyn[0:750,:]
#dyn = dyn.reshape(dyn.shape[0], -1, 4).sum(axis=-1)
#dyn = dyn.reshape(-1, 3, dyn.shape[1]).sum(axis=1)
#################

nfreq=dyn.shape[0]
nt=dyn.shape[1]
corr=np.zeros(shape=(nt,nt))

for i in xrange(nt):
    
    d_temp = dyn[:,i]
    d_temp = d_temp[:,np.newaxis]

    norm = np.sqrt(np.sum(d_temp*d_temp) * (dyn*dyn).sum(0))
    corr[i,:] = (d_temp*dyn).sum(0) / norm

#norm = (dyn[:, :-1] * dyn[:, 1:]).sum(0)
#norm = (norm[:-1] + norm[1:])/2.
#corr = corr[1:-1, 1:-1] / np.sqrt(norm[:, np.newaxis] * norm[np.newaxis, :])

fig=plt.figure()

ax1 = fig.add_subplot(211)
ax1.imshow(corr,interpolation='nearest')
ax2 = fig.add_subplot(212)
ax2.imshow(dyn,interpolation='nearest',cmap=cm.Greys,aspect='auto')
plt.xlabel('time')
plt.ylabel('frequency')
#plt.imshow(corr,interpolation='nearest', vmin=0, vmax=1.0)
#plt.imshow(corr,interpolation='nearest')
#plt.colorbar()
plt.show()

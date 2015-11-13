#!/usr/bin/env python

""" 
Takes a 1-D array of many giant pulses and computes the
correlation coefficient between them.

Currently very inefficient - it may be best to only look
adjacent pairs, or perform a time cut
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

nfreq=1024 # Number of frequency bins, currently hardcoded.
files=glob.glob('*npy')

n=len(files)
t=np.zeros(n)
dt=(np.zeros(n**2))
GPs=np.zeros((n,nfreq))
corr=np.zeros(n**2)

i=0
for f in files:
    x=np.load(f)
    GP = x
    GPs[i]=GP/np.std(GP)
    t[i]=float(f.split('GP')[-1].split('.npy')[0])
    i+=1

for j in xrange(n):
    GP_temp=GPs[j]
    GP_temp=GP_temp[np.newaxis,:]
    norm = np.sqrt(np.sum(GP_temp*GP_temp) * (GPs*GPs).sum(1))
    corr[j*n:(j+1)*n] = (GP_temp*GPs).sum(1) / norm
    dt[j*n:(j+1)*n] = t-t[j]

dt=dt[dt>1E-4]
corr=corr[dt>1E-4]

#plt.plot(np.log10(dt),corr,'.')
plt.plot(GP[3])
plt.plot(GP[16])
plt.show()

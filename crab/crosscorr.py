'''
crosscorr.py

Calculate cross correlations of spectra, to get de-correlation bandwidths and rolling correlations between spectra
'''

import numpy as np
import matplotlib.pyplot as plt

#Array is 2**15 elements, 600-800MHz
size = 2**11
shift = 2**14

frange = slice(shift,shift+size)

def correlate(s1, s2):
    y = []
    for i in range(0 ,len(s1)-1):
        if i == 0:
            v1 = s1
            v2 = s2
        else:
            v1 = s1[:-i]
            v2 = s2[i:]
        s = np.mean( ( (v1 - v1.mean()) / v1.std() ) * ( (v2 - v2.mean()) / v2.std() ) )
        y.append(s)
    return np.array(y)


if __name__ == '__main__':

    snoise = np.load('data/s3.npy')[frange]
    s1 = np.load('data/s1.npy')[frange] - snoise
    s2 = np.load('data/s2.npy')[frange] - snoise
    s3 = np.load('data/s4.npy')[frange] - snoise



    csig_33 = correlate(s1, s2)
    csig_1s = correlate(s2, s3)
    cnoise = correlate(s2, snoise)
    a1 = correlate(s1, s1)
    a2 = correlate(s2, s2)
    a3 = correlate(s3, s3)
    anoise = correlate(snoise, snoise)

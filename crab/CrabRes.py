import numpy as np
import matplotlib.pyplot as plt

def scattering(x):
    return 4. * (x / 600.)**2.

def scint_BW(x):
    return 2300. * (x / 2330.)**4.

def scint_BW44(x):
    return 2300. * (x / 2330.)**4.4

freq = np.linspace(400,800, 100)

a = scattering(freq)
b = scint_BW(freq)
c = scint_BW44(freq)

plt.plot(freq, a, label='max res')
plt.plot(freq, b, label='scint bw ^4 scaling')
plt.plot(freq, c, label='scint bw ^4.4 scaling')
#plt.xscale('log')
plt.yscale('log')

plt.xlabel('Frequency [MHz]')
plt.ylabel('Freqnency [kHz]')

plt.legend()
plt.show()

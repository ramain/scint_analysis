import numpy as np
import matplotlib.pyplot as plt

def rechannelize(pulse, nchan):
    pulse = np.fft.irfft(pulse, axis=1)
    # Reshape to channels and keep polarization
    if pulse.shape[-1] == 2:
        pulse = pulse.reshape(-1, 2*nchan, 2)
    else:
        pulse = pulse.reshape(-1, 2*nchan)
    pulse = np.fft.rfft(pulse, axis=1)
    return pulse

# Stored each pulse as voltages with 4096 channels
# Array with dimensions [time, freq, pol]
p1 = np.load('/mnt/raid-cita/ramain/p1.npy')
p2 = np.load('/mnt/raid-cita/ramain/p2.npy')

p1_coarse = rechannelize(p1, 128)

p1fine = rechannelize(p1, 2**16)
p2fine = rechannelize(p2, 2**16)

p1_selfphased = (p1fine / p1fine) * abs(p1fine)
p1_polphased = (p1fine[...,0] / p1fine[...,1]) * abs(p1fine[...,1])
p1_phased = (p1fine / p2fine) * abs(p2fine)

p1_selfphased = rechannelize(p1_selfphased, 128)
p1_polphased = rechannelize(p1_polphased, 128)
p1_phased = rechannelize(p1_phased, 128)

plt.plot(abs(p1_coarse[...,0]).sum(1),'b', label='original signal')
plt.plot(abs(p1_selfphased[...,0]).sum(1), 'g', label='self phased')
plt.plot(abs(p1_polphased).sum(1), 'k', label='self pol phased')
plt.plot(abs(p1_phased[...,0]).sum(1), 'r', label='2 pulse phased')

plt.yscale('log')
plt.xlim(3500, 5500)
plt.ylim(80, 10000)
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.time import Time

def func(x,m,b):
    return m*x+b

def plaw(x,a0,a1):
    return 10**(a1)*x**a0

f=open('giant_pulses.txt','r')

flux=[]
phase=[]

for line in f:
    sig = float(line.split(' ')[2])
    p = float(line.split(' ')[-1].split(']')[0])
    flux.append(sig)
    phase.append(p)

flux=np.array(flux)
phase=np.array(phase)

print phase

mp=flux[abs(phase-0.88)<0.2]
ip=flux[abs(phase-0.28)<0.2]

mp = np.sort(np.array(mp))
flux=mp

dist=[]
distip=[]
xaxis=[]
for i in range(6,50):
    x = len(flux[flux>i])
    x_ip = len(ip[ip>i])
    dist.append(x)
    distip.append(x_ip)
    xaxis.append(i)

dist=np.array(dist)
distip=np.array(distip)
xaxis=np.array(xaxis)

a0, a1 = curve_fit(func, np.log10(xaxis[xaxis<38]), np.log10(dist[xaxis<38]))[0]

print a0, a1

plaw_fit = plaw(xaxis,a0,a1)

#plt.errorbar(xaxis,dist,xerr=0,yerr=np.sqrt(dist),linestyle='none')
#plt.errorbar(xaxis,distip,xerr=0,yerr=np.sqrt(distip),linestyle='none')
plt.plot(xaxis,dist,'bx')
plt.plot(xaxis,distip,'g+')
plt.plot(xaxis,plaw_fit,'r-')
plt.xscale('log')
plt.yscale('log')
plt.xlim([8,50])

plt.xlabel('Flux [S/N]')
plt.ylabel('N (Flux > F0)')

plt.show()

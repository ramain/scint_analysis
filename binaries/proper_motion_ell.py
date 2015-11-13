#!/usr/bin/env python

"""
Code to visualize elliptical orbits on the sky, 
find times of orbital intersection 
"""

import numpy as np
from PyAstronomy import pyasl
import matplotlib.pyplot as plt

def poscalc(PM,T,t):
    return PM*t*T

def crossings(x,y,t,a,nstep,norbit):
    #Determine the x,y,t values of crossing points. 
    #Currently quite inefficient for large nstep.
    x_p1 = np.append(x[1:len(x)],x[-1]-0.01)
    y_p1 = np.append(y[1:len(y)],y[-1]-0.01)
    dxdy = (x_p1-x)/(y_p1-y)

    xy_sep=2.*np.pi*a / nstep
    dxy_sep=0.1 #Currently arbitrary, but seems to work.

    for i in xrange(len(x)):
        for j in xrange(len(x)):
            if i != j:
                dist = np.sqrt((x[j]-x[i])**2.0 + (y[j]-y[i])**2.0)
                ddx = (dxdy[j]-dxdy[i])/(dxdy[j]+dxdy[i])
                if dist < xy_sep and ddx > dxy_sep:
                    print("Intersection at X,Y = : ",x[i],y[i]) 
                    print("At times: ",t[i],t[j])
                    print ddx
                 
        

PM_x = -3.82 / 365.242 #mas per day
PM_y = 2.13 / 365.242 #mas per day

a = 2.578e-3 #mas of semi-major axis
T = 0.10225 #Orbital period in days

incl = 88.1
e = 0.0877775
Om = 65.0

nstep = 1200 #Number of steps
norbit = 3 #Number of orbits
xlin = []
ylin = []
t_array = []

t = np.linspace(0,norbit,nstep)

ke = pyasl.KeplerEllipse(a,1,e=e,Omega=Om,i=incl)
ell_pos = ke.xyzPos(t)

for ti in t:
    xlin.append( poscalc(PM_x,T,ti) )
    ylin.append( poscalc(PM_y,T,ti) )

xlin=np.array(xlin)
ylin=np.array(ylin)

X=ell_pos[:,0]+xlin
Y=ell_pos[:,1]+ylin
print X.shape

plt.plot(X,Y)
#plt.plot(ell_pos[:,0],ell_pos[:,1])

#crossings(X,Y,t,a,nstep,norbit)
plt.xlabel('x (microarcsec)')
plt.xlabel('y (microarcsec)')

plt.show()

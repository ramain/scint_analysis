#!/usr/bin/env python

"""
Generates an animation of a scattering ring as a function of time
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

def rt(t):
    return 160*np.sqrt(t/40)

def b(x,y):
    r = rad(x, y)
    #return 1 + abs( y / r )
    #return ( 1 + abs( y / r ) ) / (1 + r/80)
    return ( 1 + abs( y / r ) ) * (1 - r/200)

def rad(x,y):
    return np.sqrt(x*x+y*y)

def mask(z,r,t1,t2):
    zdummy = 1.*z
    zdummy[r < rt(t1)] = 0
    zdummy[r > rt(t2)] = 0
    return zdummy

x = np.linspace(-200,200,1000)
y = np.linspace(-200,200,1000)
z = np.array(b(x[:,None],y[None,:]))
rvalue = np.array(rad(x[:,None],y[None,:]))

fig = plt.figure()
ims = []
ax = fig.add_subplot(111)

for i in xrange(100):
    Z = mask(z,rvalue,i,i+1)
    im = ax.imshow(Z,vmin=0,vmax=2,cmap=cm.afmhot)
    n = ax.annotate('t = %s microseconds' % (2.5*(i+1)) ,(80,80), color='w')
    ims.append([im, n])
    
ani = animation.ArtistAnimation(fig, ims, interval=10, blit=False, repeat_delay=1000)
ani.save('ringamimation.mp4')
plt.show()

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
    r = np.sqrt(x*x+y*y)
    #return 1 + abs( y / r )
    #return ( 1 + abs( y / r ) ) / (1 + r/80)
    return ( 1 + abs( y / r ) ) * (1 - r/200)

def mask(z,x,y,t1,t2):
    zdummy = 1*z
    for i in xrange(z.shape[0]):
        for j in xrange(z.shape[1]):
            if np.sqrt(x[i]*x[i]+y[j]*y[j]) < rt(t1) :
                zdummy[i][j] = 0
            if np.sqrt(x[i]*x[i]+y[j]*y[j]) > rt(t2) :
                zdummy[i][j] = 0
    return zdummy
 
x = np.linspace(-200,200,100)
y = np.linspace(-200,200,100)
z = np.array(b(x[:,None],y[None,:]))

fig = plt.figure()
ims = []
ax = fig.add_subplot(111)

for i in xrange(70):
    Z = mask(z,x,y,i,i+1)
    im = ax.imshow(Z,vmin=0,vmax=2,cmap=cm.afmhot)
    n = ax.annotate('t = %s' % (i+1) ,(80,80), color='w')
    ims.append([im, n])
    
ani = animation.ArtistAnimation(fig, ims, interval=30, blit=False, repeat_delay=1000)

plt.show()

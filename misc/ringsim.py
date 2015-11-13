#!/usr/bin/env python

"""
Generates an image of a scattering ring as a function of time
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
 
x = np.linspace(-200,200,1000)
y = np.linspace(-200,200,1000)
z = np.array(b(x[:,None],y[None,:]))

Z1 = mask(z,x,y,0,2)
Z2 = mask(z,x,y,19,21)
Z3 = mask(z,x,y,38,40)

fig=plt.figure()

ax1 = fig.add_subplot(131)
ax1.imshow(Z1,vmin=0,vmax=2,cmap=cm.afmhot)
ax1.text(0.02, 0.92, 't = 0ms', color='w',transform=ax1.transAxes)
ax1.axis('off')

ax2 = fig.add_subplot(132)
ax2.imshow(Z2,vmin=0,vmax=2,cmap=cm.afmhot)
ax2.text(0.02, 0.92, 't = 20ms', color='w',transform=ax2.transAxes)
ax2.axis('off')

ax3 = fig.add_subplot(133)
ax3.imshow(Z3,vmin=0,vmax=2,cmap=cm.afmhot)
ax3.text(0.02, 0.92, 't = 40ms', color='w',transform=ax3.transAxes)
ax3.axis('off')
#ax3.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
#ax3.tick_params(axis='y',which='both',bottom='off',top='off',labelbottom='off')

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import ICRS, Galactic, FK4, FK5

#Plots the UV coverage for a source 

#Sources
crab = SkyCoord.from_name('M1')
b0329 = SkyCoord('03h32m59.368s +54d34m43.57s', FK5)
b1937 = SkyCoord('19h39m38.558720s +21d34m59.13745s', FK5)
frb = SkyCoord('05h31m58s +33d08m04s', FK5)
Npole = SkyCoord('00h00m00s +90d00m00s', FK5)
Eq = SkyCoord('00h00m00s +00d00m00s', FK5)

#Observatories 
ARO = EarthLocation(lat=45.6*u.deg, lon=-78.04*u.deg, height=390*u.m)
DRAO = EarthLocation(lat=49.19*u.deg, lon=-119.37*u.deg, height=545*u.m)
Ef = EarthLocation(lat=50.5247*u.deg, lon=6.8828*u.deg, height=319*u.m)
JB = EarthLocation(lat=53.23*u.deg, lon=-02.3*u.deg, height=86*u.m)
GMRT = EarthLocation(lat=19.0965*u.deg, lon=74.0497*u.deg, height=100*u.m)
GBT = EarthLocation(lat=38.4331*u.deg, lon=-79.8397*u.deg, height=100*u.m)
VLA = EarthLocation(lat=34.1*u.deg, lon=-107.6*u.deg, height=100*u.m) # Height = guess?
Test = EarthLocation(lat=45.6*u.deg, lon=0*u.deg, height=390*u.m)
greenwich = EarthLocation(lat=51.48*u.deg, lon=0.0*u.deg, height=100*u.m)

"""
Inputs
"""
source = b1937
telescopes = [Ef, ARO, DRAO]
midnight = Time('2016-04-21 00:00:00')
npoints = 50
# b1937 is visible across all 3 telescopes from 7:00-12:00 UTC 
trange = np.linspace(7, 12, npoints)*u.hour

plt.ion()

uvw_mat = np.zeros((npoints, len(telescopes), 3))

# Loop over i, j to cover all baselines
for i in xrange(len(telescopes)):
    t1 = telescopes[i]

    dX = t1.x
    dY = t1.y
    dZ = t1.z
    Xvec = np.array([dX/(1*u.m), dY/(1*u.m), dZ/(1*u.m)])

 
    k=0
    for t in trange:
        # calculate sidereal times at both telescopes, 
        # average for use in hourangle
        ot=Time(midnight+t, scale='utc', location=t1)
        ot.delta_ut1_utc = 0.
        obst = ot.sidereal_time('mean')
           
        # I'm certain there's a better astropy way to get ot_avg in degrees
        h = obst.deg*u.deg - source.ra   
        dec = source.dec

        # matrix to transform xyz to uvw
        mat = np.array([(np.sin(h), np.cos(h), 0), (-np.sin(dec)*np.cos(h), np.sin(dec)*np.sin(h), np.cos(dec)), (np.cos(dec)*np.cos(h), -np.cos(dec)*np.sin(h), np.sin(dec))])

        uvw = np.dot(mat, Xvec)
        uvw_mat[k, i] = uvw
        k += 1

uvw_pdiff = np.zeros((npoints, len(telescopes)*(len(telescopes)-1)//2, 3))
uvw_ndiff = np.zeros((npoints, len(telescopes)*(len(telescopes)-1)//2, 3))

k=0
for i in xrange(len(telescopes)):
    for j in range(i+1, len(telescopes)):
        uvw_pdiff[:,k] = uvw_mat[:,i] - uvw_mat[:,j]
        uvw_ndiff[:,k] = uvw_mat[:,j] - uvw_mat[:,i]
        k += 1

plt.plot(uvw_pdiff[:,:,0] / 1000, uvw_pdiff[:,:,1] / 1000, 'b')
plt.plot(uvw_ndiff[:,:,0] / 1000, uvw_ndiff[:,:,1] / 1000, 'b')

plt.xlabel('u [km]')  
plt.ylabel('v [km]')
plt.xlim(-10000, 10000)
plt.ylim(-8000, 8000)
plt.title('b1937')
plt.show()

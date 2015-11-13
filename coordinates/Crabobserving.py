import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import ICRS, Galactic, FK4, FK5

#Plots the observability of the Crab pulsar from various observatories

crab = SkyCoord.from_name('M1')
#crab = SkyCoord("19h39m38.6s +21d34m59.1s", FK5)
#crab = SkyCoord.from_name('b1937+21')

#Observatories 
#ARO = EarthLocation(lat=45.6*u.deg, lon=-78.04*u.deg, height=390*u.m)
#DRAO = EarthLocation(lat=49.19*u.deg, lon=-119.37*u.deg, height=545*u.m)
#JB = EarthLocation(lat=53.23*u.deg, lon=-02.3*u.deg, height=86*u.m)
VLA = EarthLocation(lat=34.1*u.deg, lon=-107.6*u.deg, height=100*u.m) # Height = guess?

utcoffset = -6*u.hour  # Eastern Daylight Time
#utcoffset = -7*u.hour  # Pacific Time
#utcoffset = 0*u.hour
midnight = Time('2015-9-20 00:00:00') - utcoffset
delta_midnight = np.linspace(-10, 10, 100)*u.hour
crabaltazs = crab.transform_to(AltAz(obstime=midnight+delta_midnight, location=VLA))

plt.plot(delta_midnight, crabaltazs.alt)  
plt.xlim(-10, 10)  
#plt.ylim(1, 4)  
plt.xlabel('Hours from UTC Midnight')  
plt.ylabel('Altitude [degrees]')
plt.axhline(y=8,xmin=-10, xmax=10)
plt.show()

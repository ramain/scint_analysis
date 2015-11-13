import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import ICRS, Galactic, FK4, FK5

#Find Altitude and Azimuth at ARO of the zenith at DRAO

#Observatories 
ARO = EarthLocation(lat=45.9555*u.deg, lon=-78.0730*u.deg, height=390*u.m)
DRAO = EarthLocation(lat=49.321*u.deg, lon=-119.624*u.deg, height=545*u.m)

midnight = Time('2015-9-01 00:00:00')

zenith = SkyCoord(AltAz,az=358.2*u.deg,alt=50.0*u.deg, obstime=midnight, location=DRAO)

ARO_coords = ARO.to_geocentric()
DRAO_coords = DRAO.to_geocentric()

print DRAO_coords
print ARO_coords

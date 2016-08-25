import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import ICRS, Galactic, FK4, FK5

#Plots the observability of the Crab pulsar from various observatories

crab = SkyCoord.from_name('M1')
b0329 = SkyCoord('03h32m59.368s +54d34m43.57s', FK5)
b1937 = SkyCoord('19h39m38.558720s +21d34m59.13745s', FK5)
b1919 = SkyCoord('19h21m44.815s +21d53m02.25s', FK5)
frb = SkyCoord('05h31m58s +33d08m04s', FK5)
magnetar = SkyCoord('17h45m40.1662s -29d00m29.8958s', FK5)
J1723 = SkyCoord('17h23m23.1856s -28d37m57.17s', FK5)

J0518 = SkyCoord('05h18m05.1424740s +33d06m13.365060s', FK5)
J0530 = SkyCoord('05h30m12.5492240s +37d23m32.619700s', FK5)

""" Nearby Calibration Sources """
J0539p3308 = SkyCoord('05h39m09.6718850s +33d08m15.490870s', FK5)
J0533p3451 = SkyCoord('05h33m12.7651060s +34d51m30.336990s', FK5)

""" Nearby Fringe Finders """
J0555p3948 = SkyCoord('05h55m30.8056160s +39d48m49.164930s', FK5)  # Probably use this one
J0530p1331 = SkyCoord('05h30m56.4167490s +13d31m55.149440s', FK5)

#Observatories 
ARO = EarthLocation(lat=45.6*u.deg, lon=-78.04*u.deg, height=390*u.m)
DRAO = EarthLocation(lat=49.19*u.deg, lon=-119.37*u.deg, height=545*u.m)
Ef = EarthLocation(lat=50.5247*u.deg, lon=6.8828*u.deg, height=319*u.m)
JB = EarthLocation(lat=53.23*u.deg, lon=-02.3*u.deg, height=86*u.m)
VLA = EarthLocation(lat=34.1*u.deg, lon=-107.6*u.deg, height=100*u.m) # Height = guess?

#BR_VLBA = EarthLocation(-2112065.0155, -3705356.5107, 4726813.7669)
#FD_VLBA = EarthLocation(-1324009.1622 ,-5332181.9600, 3231962.4508)
#HN_VLBA = EarthLocation( 1446375.0685, -4447939.6572, 4322306.1267)
#KP_VLBA = EarthLocation(-1995678.6640, -5037317.7064, 3357328.1027)
#LA_VLBA = EarthLocation(-1449752.3988, -4975298.5819, 3709123.9044)
#MK_VLBA = EarthLocation(-5464075.0019, -2495248.9291, 2148296.9417)
#NL_VLBA = EarthLocation(-130872.2990, -4762317.1109, 4226851.0268)
#OV_VLBA = EarthLocation(-2409150.1629, -4478573.2045, 3838617.3797)
#PT_VLBA = EarthLocation(-1640953.7116, -5014816.0236, 3575411.8801)
#SC_VLBA = EarthLocation( 2607848.5431, -5488069.6574, 1932739.5702)

source = b0329
observatory = ARO
midnight = Time('2016-08-30 00:00:00')

delta_midnight = np.linspace(-24, 24, 200)*u.hour

altaz = source.transform_to(AltAz(obstime=midnight+delta_midnight, location=observatory))

plt.plot(delta_midnight, altaz.alt, label='ARO alt')  
plt.plot(delta_midnight, altaz.az, label='ARO az')  

#plt.xlim(7, 12)  
#plt.ylim(-90, 90)  
plt.xlabel('Hours from UTC Midnight')  
plt.ylabel('Altitude [degrees]')
plt.axhline(y=51)
plt.axhline(y=41)
plt.axhline(y=10)
plt.legend()
plt.title('J1723, ARO 08/30/2016')
#plt.title('%s at %s, %s' % (str(source), str(observatory), midnight.isot))
plt.show()

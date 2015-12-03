import numpy as np
import scipy.optimize as so

class FaradayTools():
     """ Class for RM fitting, Faraday de-rotation, 

     """

     def __init__(self, data):
          self.nfreq = 1024
          #self.nfreq = 500
          self.freq = np.linspace(400, 800, self.nfreq) * 1e6
          #self.freq = np.linspace(400, 400+195.3125, 500) * 1e6
          self.lam = 3e8 / self.freq
          self.data = data

     def fr_sin(self, lam, params):
          """ Sinusoidal farday rotation
          
          Parameters
          ----------
          lam : array_like
               Wavelength in meters
          params : list
               A, RM, phi, C 

          Returns
          -------
          Sinusoid with given parameters
          """

          A, RM, phi, C = params[0], params[1], params[2], params[3]

          return A * np.exp(2j * (RM * lam**2 + phi))

     def fr_cabdelay_exp(self, lam, params):
          """ Sinusoidal farday rotation
          
          Parameters
          ----------
          lam : array_like
               Wavelength in meters
          params : list
               A, RM, phi, pix, phi2, C

          Returns
          -------
          Sinusoid with given parameters
          """

          A, pix, phi2, C = params[0], params[1], params[2], params[3]

          fr_lam2 = -1.11081175*np.exp(2j * (-42.797459 * lam**2 + 0.54372979)) + 0.27691225

          fr_nu = A * np.exp(2*np.pi*1j*(np.arange(self.nfreq)/np.float(self.nfreq) * pix + phi2))
          #fr_nu = A * np.exp(2*np.pi*1j*((self.freq/1e6)/np.float(self.nfreq) * pix + phi2))
          
          #return fr_lam2 * fr_nu + C
          return fr_nu + C

     def diff(self, params, lam, data):
          """ Gets abs difference of data to fr_sin function

          Parameters
          ----------
          params : list
               A, RM, phi, C 

          lam : array_like
               Wavelength in meters

          data : array_like
               visibilities to fit, (nfreq)

          Returns
          -------
          Absolute value of difference 
          """
          diff = self.fr_sin(lam, params) - data

          return diff.real**2 + diff.imag**2

     def diff2(self, params, lam, data):
          """ Gets abs difference of data to fr_sin function

          Parameters
          ----------
          params : list
               A, RM, phi, C 

          lam : array_like
               Wavelength in meters

          data : array_like
               visibilities to fit, (nfreq)

          Returns
          -------
          Absolute value of difference 
          """
          diff = self.fr_cabdelay_exp(lam, params) - data

          return diff.real**2 + diff.imag**2

     def fit_RM(self, RM_o):
          """
          """
          A = np.std(self.data).real
          phi = 0.0
          C = np.median(self.data).real

          params_o = [A, RM_o, phi, C]

          params = so.leastsq(self.diff, params_o, args=(self.lam, self.data), maxfev=10000)[0]

          return params

     def fit_RM_cabdelay(self, pix_o):
          """
          """
          #C = np.median(self.data).real
          A = np.std(self.data).real

          params_o = [A, pix_o, -0.4, 0.0]

          params = so.leastsq(self.diff2, params_o, args=(self.lam, self.data), maxfev=10000)[0]

          return params


def run_RM_fits(data, RM_o):
     ntimes = data.shape[-1]
     nfreq = data.shape[0]

     params = []

     FT = FaradayTools(data[:, 0])
     
     for tt in range(ntimes):

          FT.data = data[:, tt]
          params.append(FT.fit_RM(RM_o))

     return params

def run_RM_fits2(data, pix_o):
     ntimes = data.shape[-1]
     nfreq = data.shape[0]

     params = []

     FT = FaradayTools(data[:, 0])
     
     for tt in range(ntimes):

          FT.data = data[:, tt]
          params.append(FT.fit_RM_cabdelay(pix_o))

     return params











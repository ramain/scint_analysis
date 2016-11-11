from __future__ import division, print_function, unicode_literals

import numpy as np
import astropy.units as u


class DispersionMeasure(u.Quantity):

    '''
    Marten's Dispersion Measure Class, added here for convenience
    '''

    dispersion_delay_constant = 4149. * u.s * u.MHz**2 * u.cm**3 / u.pc
    _default_unit = u.pc / u.cm**3

    def __new__(cls, dm, unit=None, **kwargs):
        if unit is None:
            unit = getattr(dm, 'unit', cls._default_unit)
        self = super(DispersionMeasure, cls).__new__(cls, dm, unit, **kwargs)
        if not self.unit.is_equivalent(cls._default_unit):
            raise u.UnitsError("Dispersion measures should have units "
                               "equivalent to pc/cm^3")
        return self

    def __quantity_subclass__(self, unit):
        if unit.is_equivalent(self._default_unit):
            return DispersionMeasure, True
        else:
            return super(DispersionMeasure,
                         self).__quantity_subclass__(unit)[0], False

    def time_delay(self, f, fref=None):
        """Time delay due to dispersion.

        Parameters
        ----------
        f : `~astropy.units.Quantity`
            Frequency at which to evaluate the dispersion delay.
        fref : `~astropy.units.Quantity`, optional
            Reference frequency relative to which the dispersion delay is
            evaluated.  If not given, infinite frequency is assumed.
        """
        d = self.dispersion_delay_constant * self
        fref_inv2 = 0. if fref is None else 1. / fref**2
        return d * (1./f**2 - fref_inv2)

    def phase_delay(self, f, fref=None):
        """Phase delay due to dispersion.

        Parameters
        ----------
        f : `~astropy.units.Quantity`
            Frequency at which to evaluate the dispersion delay.
        fref : `~astropy.units.Quantity`, optional
            Reference frequency relative to which the dispersion delay is
            evaluated.  If not given, infinite frequency is assumed.
        """
        d = self.dispersion_delay_constant * u.cycle * self
        fref_inv = 0. if fref is None else 1. / fref
        return d * (f * (fref_inv - 1./f)**2)

    def phase_factor(self, f, fref=None):
        """Complex exponential factor due to dispersion.

        This is just ``exp(1j * phase_delay)``.

        Parameters
        ----------
        f : `~astropy.units.Quantity`
            Frequency at which to evaluate the dispersion delay.
        fref : `~astropy.units.Quantity`, optional
            Reference frequency relative to which the dispersion delay is
            evaluated.  If not given, infinite frequency is assumed.
        """
        return np.exp(self.phase_delay(f, fref).to(u.rad).value * 1j)


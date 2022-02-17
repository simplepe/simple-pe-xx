from astropy.cosmology import Planck15 as cosmo
import astropy.units as unt
import numpy as np
from scipy import interpolate

##################################################################
# Cosmology
##################################################################

max_z = 200.
redshifts = np.linspace(0, max_z, 2000)
distances = cosmo.comoving_distance(redshifts)
dl = cosmo.luminosity_distance(redshifts)
z_interp = interpolate.interp1d(distances, redshifts)
zdl_interp = interpolate.interp1d(dl, redshifts)


def redshift_at_comoving_dist(distance):
    """
    return redshift at a given comoving distance
    """
    return z_interp(distance)

def redshift_at_lum_dist(distance):
    """
    return redshift at a given luminosity distance
    """
    return zdl_interp(distance)

def luminosity_distance(distance):
    """
    return luminosity distance for a given comoving distance
    """
    return (1 + z_interp(distance))*distance/unt.Mpc


def volume(z_list, ratio_list):
    return (4 * np.pi * ratio_list*cosmo.differential_comoving_volume(z_list).value/(1+z_list)).sum()
    


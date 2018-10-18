from astropy.cosmology import Planck15 as cosmo
import astropy.units as unt
import numpy as np
import detectors
from scipy import interpolate

##################################################################
# Cosmology
##################################################################

max_z = 10.
redshifts = np.linspace(0, 2*10., 2000)
distances = cosmo.comoving_distance(redshifts)
dl = cosmo.luminosity_distance(redshifts)
z_interp = interpolate.interp1d(distances, redshifts)
zdl_interp = interpolate.interp1d(dl, redshifts)

#mchs = np.arange(1, 50.0, .05)
#maxz = np.zeros_like(mchs)
#
#for i, mch in enumerate(mchs):
#    for z in redshifts:    # ets = .25 for equal mass binaries.
#        if cosmo.luminosity_distance(z)/unt.Mpc > detectors.read_dhr('ET', mch*(1+z), .25)[0][0]:
#            maxz[i] = z
#            break
#maxz_interp = interpolate.interp1d(mchs, maxz)

def redshift(distance):
    """
    return redshift at a given comoving distance
    """
    return z_interp(distance)

def red_dl(distance):
    """
    return redshift at a given luminosity distance
    """
    return zdl_interp(distance)

def luminosity_distance(distance):
    """
    return luminosity distance for a given comoving distance
    """
    return (1 + z_interp(distance))*distance/unt.Mpc

#def max_comoving_distance(mchirp, eta = .25):
#   """
#   return the maximum injectable distance for a given chirp mass and eta considering cosmology
#   """
#   return cosmo.comoving_distance(maxz_interp(mchirp))/unt.Mpc

def volume(z_list, ratio_list):
    return (4 * np.pi * ratio_list*cosmo.differential_comoving_volume(z_list).value/(1+z_list)).sum()
    


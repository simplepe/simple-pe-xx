import numpy as np
from numpy import *
from simple_pe import cosmology
import astropy.units as u

import scipy.integrate as integrate
from scipy.integrate import quad
from scipy.interpolate import interp1d

from astropy.cosmology import Planck15, z_at_value

BNSrate=1500 * u.Gpc**-3 * u.yr**-1 #  From BNS discovery paper

t_D_min = 0.02 #from Regimau Mock #0.05 - from Taylor and Gair 2012
t_D_max = 13.7

'''We are using Planck15 cosmological parameters:'''
# H0 = 67.74
# Om0 = 0.307
# Ode0 = 0.691

def sfr(z):
    '''
        equation 15 on p. 48 of Madau and Dickenson (2014)
        http://www.annualreviews.org/doi/pdf/10.1146/annurev-astro-081811-125615
        '''
    return 0.015*(1.+z)**2.7/(1.+(1.+z)/2.9)**5.6  #msun per yr per Mpc^3

def redshift_at_age(t):
    '''
        t in billions of years
        '''
    z = z_at_age_interp(t)
    return z

def age_at_redshift(z):
    '''
        t in billions of years
        '''
    t = Planck15.age(z)/u.Gyr
    return t

def delay_distribution(t, t_D_min=t_D_min, t_D_max=t_D_max):
    dist = array([1/t])
    dist[where(t_D_max<t_D)] = 0
    dist[where(t_D<t_D_min)] = 0
    return dist

def z_at_formation(z,t_D):
    age = age_at_redshift(z) - t_D
    if (age<0).any():
        raise ValueError('this time delay - redshift combination would require formation before the beginning of the universe')
    else: return redshift_at_age(age)

def rate_density_coalescence_integrand(t_D, z):
    zf = z_at_formation(z, t_D)
    return sfr(zf) * delay_distribution(t_D) / (1+zf)

def rate_density_coalescence(z, t_D_min=t_D_min, t_D_max=t_D_max):
    epsilon = 0.001 #otherwise rounding errors causes astropy to throw an error
    t_D_max = age_at_redshift(z) - epsilon
    return integrate.quad(rate_density_coalescence_integrand, t_D_min, t_D_max, args = (z), epsrel=1.49e-02)[0]

def integrand(z, R0 = BNSrate): # eq 31 in Regimbau 2012 https://journals.aps.org/prd/pdf/10.1103/PhysRevD.86.122001
    norm = R0/rate_density_coalescence(0)
    R_local = norm * rate_density_coalescence(z)
    dv_dz=Planck15.differential_comoving_volume(z).to(u.Gpc**3 / u.sr) #dv_dz_interp(z)
    return 4*np.pi *u.sr * dv_dz * R_local * u.yr #1/(1+z) factor accounted for in rate_density_coalescence_integrand

epsilon = 0.0005
ages = geomspace(epsilon, age_at_redshift(0)-epsilon,1000)
z_at_ages = array([z_at_value(Planck15.age, age*u.Gyr) for age in ages])
z_at_age_interp = interp1d(ages,z_at_ages)

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

# Time - redshift conversions

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
    return age_at_z_interp(z)
#interpolation functions to speed up:
redshifts = linspace(0.0, 20, 1000)
redshifts = append(redshifts, linspace(20,900,100))

age_at_z = array([Planck15.age(z)/u.Gyr for z in redshifts])
age_at_z_interp = interp1d(redshifts, age_at_z)

epsilon = 0.0005
ages = linspace(epsilon, age_at_redshift(0)-epsilon,1000)
z_at_ages = array([z_at_value(Planck15.age, age*u.Gyr) for age in ages])
ages = append(ages, age_at_redshift(0))
z_at_ages = append(z_at_ages, 0)
z_at_age_interp = interp1d(ages,z_at_ages)

def z_at_formation(z,t_D):
    '''takes the redshift of a source at merger and the delay since formation,
        outputs the redshift at which the binary was formed
        '''
    age = age_at_redshift(z) - t_D
    if (age<0).any():
        raise ValueError('this time delay - redshift combination would require formation before the beginning of the universe')
    else: return redshift_at_age(age)

# Star formation rate fits

def sfrMD(z):
    '''
        equation 15 on p. 48 of Madau and Dickenson (2014)
        http://www.annualreviews.org/doi/pdf/10.1146/annurev-astro-081811-125615
        uses (O_M, O_de, h) = (0.3, 0.7, 0.7) params
        '''
    sfrmd = 0.015*(1.+z)**2.7/(1.+((1.+z)/2.9)**5.6) #msun per yr per Mpc^3
    #below convert to Plan15 cosmology (essentially negligible)
    #(H0 constant out the fron will be normalized away later so we ignore it)
    return sfrmd #* sqrt((0.3075*(1+z)**3 + 0.691)/(0.3*(1+z)**3 + 0.7)) neglecting this cosmology convertion factor as it is essentially negligible

def sfrHB(z):
    '''
        SFR of Hopkins and Beacom (2006)
        http://iopscience.iop.org/article/10.1086/506610/pdf
        '''
    return 0.7 * (0.017 + 0.13*z) / (1.+(z/3.3)**5.3)

def sfr2_porciani_madau(z):
    '''
        SFR2 from porciani madau 2001
        http://stacks.iop.org/0004-637X/548/i=2/a=522
        '''
    return exp(3.4*z) / (exp(3.4*z)+22.) *Planck15.efunc(z) /((1+z)**(3/2.))

# Time Delay Distribution

def P(t_D, t_D_max = age_at_redshift(0), t_D_min = t_D_min):
    '''1/t delay time distribution'''
    norm_cnst = 1/log(t_D_max/t_D_min) # denominator = integral of 1/t from min to max delay times.
    try:
        test_dist = 1./t_D
        test_dist[where(t_D_max<t_D)] = 0
        test_dist[where(t_D<t_D_min)] = 0
        return norm_cnst*test_dist
    except TypeError: # when times are given as floats, instead of arrays
        if t_D <t_D_min: return 0
        elif t_D>t_D_max: return 0
        else: return norm_cnst*1/t_D

# CBC merger rate density

def rate_density_integrand(tb, t, sfr = sfrMD):
    '''eq 37 of Nakar (2007)
        http://www.sciencedirect.com/science/article/pii/S0370157307000476?via%3Dihub
        
        equivalent to eq 3 of LIGO GW150914 stochastic paper (2016)
        https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.116.131102
        
        also eq 1 from Anand (2017) https://arxiv.org/pdf/1710.04996.pdf
        
        also eq 3 of Ghirlanda (2016) https://www.aanda.org/articles/aa/full_html/2016/10/aa28993-16/aa28993-16.html
        
        also eq 1 of LIGO kilanova paper (2017) http://iopscience.iop.org/article/10.3847/2041-8213/aa9478/pdf
        
        also eq 18 in Taylor and Gair (2012) https://arxiv.org/abs/1204.6739
        '''
    t_D = t - tb
    zf = redshift_at_age(tb)
    return   sfr(zf) * P(t_D)


def rate_density(z, z_max = 20, sfr = sfrMD):
    '''
        computes the rate density at z by integrating over the contributions
        from all possible delay times from binary birth to merger
        and the attending SFR at the at the time of formation
        
        z_max: define a maximum redshift beyond which we assume there is no star formation
        
        sfr: specify a cosmic sfr function
        
        '''
    t = age_at_redshift(z)
    t_min = age_at_redshift(z_max)
    return  integrate.quad(rate_density_integrand, t_min, t,
                           args = (t, sfr), epsabs = 0)[0]
def integrand(z, R0 = BNSrate, z_max = 20, sfr = sfrMD):
    '''
        
    '''
    norm = R0/rate_density(0, z_max, sfr)
    R_local = norm * rate_density(z, z_max = z_max, sfr = sfr)
    dv_dz=Planck15.differential_comoving_volume(z).to(u.Gpc**3 / u.sr) #dv_dz_interp(z)
    return 4*np.pi *u.sr * dv_dz * R_local * u.yr * 1/(1+z) # 4*pi factor: dv/dz is given in per steridians and so we multiply by the whole sky

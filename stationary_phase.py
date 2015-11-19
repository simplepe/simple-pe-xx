import math
from scipy.constants import G, c, pi, mega, parsec
import pylab

def amplitude(freq, m1, m2, distance = 1):
  solar_mass = 1.98892e30
  mpc = mega *parsec
  eta = m1 * m2 / (m1 + m2)**2
  M = (m1 + m2)
  mu = M * eta
  amp_fac = (2 * G * solar_mass) / (c**2 * distance * mpc) * \
      (5. * mu / 96 )**(1./2) * \
      (M / math.pi**2)**(1./3) * \
      (G * solar_mass / c**3)**(-1./6) 
  fi = freq < isco(m1,m2)
  amp = pylab.zeros_like(freq)
  amp[fi] = freq[fi]**(-7./6) * amp_fac
  return amp

def stationary_phase(freq,m1, m2):
  M = (m1 + m2)* solar_mass * G / c**3
  fstar = 40.0
  pimf = math.pi * M * fstar
  eta = m1 * m2 / (m1 + m2)**2
  mchirp = (m1 + m2) * eta**(3./5)
  tau0 = 5./256./math.pi/fstar*pimf**(-5./3.)/eta
  tau2 = 5./192./math.pi/fstar/pimf/eta*(743./336. + 11./4.*eta)
  tau3 = 1./8./fstar*pimf**(-2./3.)/eta
  tau4 = 5./128./math.pi/fstar*pimf**(-1./3.)/eta * \
      (3058673./1016064. + 5429./1008.*eta + 617./144. * eta**2 )
  x = (freq/fstar)**(-1./3)

  phi = 2.*math.pi*fstar* ( 3. * tau0 / 5. * x**5 + tau2 * x**3 \
      - 3. * tau3 /2. * x**2 + 3. * tau4 * x)
  time = tau0 * x**8 + tau2 * x**6 + tau3 * x**5 + tau4 * x**4

  return time, phi

def spin_phase(freq, mchirp, eta, chi):
  solar_mass = 1.98892e30
  M = mchirp * eta**(-3./5) * solar_mass * G / c**3
  nu = (math.pi * M * freq)**(1./3)
  psi = 3./128./eta/nu**5 * ( 1 + nu**2 * (3715./756. + 55.*eta/9.) -
    nu**3 * ( 16*math.pi - (113. * chi/3.) ) )

  return psi


def isco(m1, m2):
  solar_mass = 1.98892e30
  M = (m1 + m2)* solar_mass * G / c**3
  f = 1./ (6 * math.sqrt(6) * math.pi * M)
  return f

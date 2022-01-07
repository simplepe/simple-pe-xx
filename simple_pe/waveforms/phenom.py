from __future__ import division
from scipy.constants import G, c, pi, mega, parsec
import math
import pylab

def frequencies(m1, m2):
  solar_mass = 1.98892e30
  M = (m1 + m2)* solar_mass * G / c**3
  eta = m1 * m2 / (m1 + m2)**2
  a = {}
  b = {}
  C = {}
  a["merge"]= 2.9740e-1
  b["merge"]= 4.4810e-2
  C["merge"]= 9.5560e-2
  a["ring"] = 5.9411e-1
  b["ring"] = 8.9794e-2
  C["ring"] = 1.9111e-1
  a["sigma"]= 5.0801e-1
  b["sigma"]= 7.7515e-2
  C["sigma"]= 2.2369e-2
  a["cut"]  = 8.4845e-1
  b["cut"]  = 1.2848e-1
  C["cut"]  = 2.7299e-1

  freq = {}
  for type in ["merge", "ring", "sigma", "cut"]:
    freq[type] = ( a[type] * eta * eta + b[type]*eta + C[type] ) / pi / M

  return freq

def amplitude(freq, m1, m2,distance=1):
  ff = frequencies(m1, m2)
  amp = pylab.zeros_like(freq)
  M = m1 + m2
  eta = m1 * m2 / M**2
  G = 6.67300e-11
  c = 299792458
  solar_mass = 1.98892e30
  mpc = mega *parsec 
  amp_fac = (5*eta/24)**(1./2)* (1./pi)**(2./3) * c/(distance*mpc) * \
       (G*M*solar_mass/c**3)**(5/6)* ff["merge"]**(-7./6)
  fi = freq < ff["merge"]
  fm = (freq > ff["merge"]) * (freq < ff["ring"])
  fr = (freq > ff["ring"]) * (freq < ff["cut"])
  amp[fi] = ( freq[fi] / ff["merge"] )**(-7./6)
  amp[fm] = ( freq[fm] / ff["merge"] )**(-2./3)
  w = pi * ff["sigma"] / 2 * ( ff["ring"] / ff["merge"] )**(-2./3)
  amp[fr] = w  / 2 / pi * ff["sigma"] / \
      ( (freq[fr] - ff["ring"])**2 + ff["sigma"]**2 / 4 )
  amp *= amp_fac
  return amp


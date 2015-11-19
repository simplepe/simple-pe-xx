from __future__ import division
import pylab
import numpy
import math
import psd
import stationary_phase
import phenom
from scipy import integrate

def calc_snr_vs_time(f, p, dt, m1,m2,toff=0):

  amp = stationary_phase.amplitude(f, m1, m2)
  norm, f_mean, f_band = psd.calculate_moments(f, amp, p)
  fac = 4 * amp**2 / p / norm

  overlap = numpy.zeros_like(dt, dtype = complex)

  # calculate the snr for the two phases:
  for i in xrange(len(dt)):
    overlap[i] = numpy.trapz(fac * numpy.exp(2 * math.pi * 1j *f * \
        (dt[i] - toff)) , f)

  return(overlap)

def calc_dh_vs_time(f, p, dt, m1,m2,):
  # calculate the overlap between dh (orthogonal to h) and h(t)
  # dh = (f - f_mean)/f_band * h
  amp = stationary_phase.amplitude(f, m1, m2)
  norm, f_mean, f_band = psd.calculate_moments(f, amp, p)
  fac = 4 * amp**2 / p / norm

  overlap = numpy.zeros_like(dt, dtype = complex)

  # calculate the snr for the two phases:
  for i in xrange(len(dt)):
    overlap[i] = numpy.trapz(fac * (f - f_mean)/f_band * \
        numpy.exp(2 * math.pi * 1j * f * dt[i]) , f)

  return(overlap)

def approx_snr_vs_time(f, p, dt, m1,m2):

  amp = stationary_phase.amplitude(f, m1, m2)
  norm, f_mean, f_band = psd.calculate_moments(f, amp, p)
  fac = 4 * amp**2 / p / norm

  cosfac = 1 - 2* math.pi**2 * dt**2 * (f_band**2 + f_mean**2)
  sinfac = 2 * math.pi * dt * f_mean
  overlap = cosfac + 1j * sinfac

  return(overlap)

def calc_snr_time_two_tmplt(f, p, dt, m1,m2,toff):

  amp = stationary_phase.amplitude(f, m1, m2)
  norm, f_mean, f_band = psd.calculate_moments(f, amp, p)
  fac = 4 * amp**2 / p / norm

  overlap_plus_toff = numpy.zeros_like(dt, dtype = complex)
  overlap_minus_toff = numpy.zeros_like(dt, dtype = complex)
  
  # calculate the overlap for waveforms at +toff and - toff
  for i in xrange(len(dt)):
    overlap_plus_toff[i] = numpy.trapz(fac * \
        numpy.exp(2 * math.pi * 1j * f * (dt[i] - toff)) , f)
    overlap_minus_toff[i] = numpy.trapz(fac * \
        numpy.exp(2 * math.pi * 1j * f * (dt[i] + toff)) , f)

  # now, calculate the orthogonal waveforms and their overlaps:
  overlap_2toff = numpy.trapz(fac * \
      numpy.exp(2 * math.pi * 1j * f * 2 * toff) , f)
  phase_off = 0.5 * numpy.angle(overlap_2toff)

  # first waveform is e(-i phase_off) h(t) + e(i phase_off) h(-t) 
  # second waveform is e(-i phase_off) h(t) - e(i phase_off) h(-t) 
  norm_1 = numpy.sqrt(2 + 2*abs(overlap_2toff))
  norm_2 = numpy.sqrt(2 - 2*abs(overlap_2toff))

  overlap_1 = (1./norm_1) * (overlap_plus_toff * numpy.exp(1j* phase_off) + 
                             overlap_minus_toff * numpy.exp(-1j* phase_off) )
  overlap_2 = (1./norm_2) * (overlap_plus_toff * numpy.exp(1j* phase_off) 
                           - overlap_minus_toff * numpy.exp(-1j* phase_off))

  return(overlap_plus_toff, overlap_minus_toff, overlap_1, overlap_2)


def calc_snr_vs_two_tmplt(f, p, m1, m2, n, match = 0.97, fudge = 1):
  
  # approximate offset for 97% match:
  # match squared between h and h exp(i a f^(n/3)) is approx
  # 1 - a^2 (j[7-2n] - j[7-n]^2)
  j = psd.all_moments(f, p, 1500)
  if n==0:
    # phase -- "forget" that we maximize over it
    a_97 = numpy.sqrt( (1 - match**2) / j[7-2*n])
  else:
    a_97 = numpy.sqrt( (1 - match**2) / (j[7-2*n] - j[7-n]**2) )

  a_97 *= fudge

  da = numpy.linspace(-2*a_97, 2*a_97, 101)
  overlap_plus = numpy.zeros_like(da, dtype = complex)
  overlap_minus = numpy.zeros_like(da, dtype = complex)
  
  amp = stationary_phase.amplitude(f, m1, m2)
  norm, f_mean, f_band = psd.calculate_moments(f, amp, p)
  fac = 4 * amp**2 / p / norm

  # calculate the overlap for waveforms at +a_97 and -a_97
  for i in xrange(len(da)):
    overlap_plus[i] = numpy.trapz(fac * \
        numpy.exp(1j * f**(n/3.) * (da[i] - a_97)) , f)
    overlap_minus[i] = numpy.trapz(fac * \
        numpy.exp(1j * f**(n/3.) * (da[i] + a_97)) , f)

  # now, calculate the orthogonal waveforms and their overlaps:
  overlap_2step = numpy.trapz(fac * \
      numpy.exp(1j * f**(n/3.) * 2 * a_97) , f)
  phase_off = 0.5 * numpy.angle(overlap_2step)

  if n == 0:
    overlap_2step = overlap_2step.real
    phase_off = 0

  # first waveform is e(-i phase_off) h(t) + e(i phase_off) h(-t) 
  # second waveform is e(-i phase_off) h(t) - e(i phase_off) h(-t) 
  norm_1 = numpy.sqrt(2 + 2*abs(overlap_2step))
  norm_2 = numpy.sqrt(2 - 2*abs(overlap_2step))
  norm_int = numpy.sqrt(2*(da**2 + a_97**2) + \
      2*abs(overlap_2step)*(-da**2 + a_97**2) )

  overlap_1 = (1./norm_1) * (overlap_plus * numpy.exp(1j* phase_off) + 
                             overlap_minus * numpy.exp(-1j* phase_off) )
  overlap_2 = (1./norm_2) * (overlap_plus * numpy.exp(1j* phase_off) 
                           - overlap_minus * numpy.exp(-1j* phase_off))
                           
  overlap_interp = (1./norm_int) * ((a_97 + da) * overlap_plus * numpy.exp(1j* phase_off)
                          + (a_97 - da) * overlap_minus * numpy.exp(-1j* phase_off))

  return(da, overlap_plus, overlap_minus, overlap_1, overlap_2, overlap_interp)



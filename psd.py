import numpy
import pylab
from scipy import integrate
import math
import numpy

data = {}
data["S6 H1"] = "/Users/stephenfairhurst/papers/cgp_fairhurst/ligo/localization/data/lho4k_15May2010_05hrs17min45secUTC_strain.txt"
data["No SRM"] = "/Users/stephenfairhurst/papers/cgp_fairhurst/ligo/localization/data/NO_SRM.txt"
data["Zero Det"] = "/Users/stephenfairhurst/papers/cgp_fairhurst/ligo/localization/data/ZERO_DET_high_P.txt"
data["Early aLIGO"] = "/Users/stephenfairhurst/papers/cgp_fairhurst/ligo/localization/data/early_aligo.dat.dat"
data["Mid aLIGO"] = "/Users/stephenfairhurst/papers/cgp_fairhurst/ligo/localization/data/mid_aligo.dat.dat"
data["Design aLIGO"] = "/Users/stephenfairhurst/papers/cgp_fairhurst/ligo/localization/data/final_aligo.dat.dat"
data["BBH Opt"] = "/Users/stephenfairhurst/papers/cgp_fairhurst/ligo/localization/data/BHBH_20deg.txt"
data["BNS Opt"] = "/Users/stephenfairhurst/papers/cgp_fairhurst/ligo/localization/data/NSNS_opt.txt"
data["Sigg aLIGO"] = "/Users/stephenfairhurst/papers/cgp_fairhurst/ligo/localization/data/aLIGO_sigg.txt"
data["Squeeze aLIGO"] = ".ata/aLIGO_sqz.txt"
data["KAGRA B"] = "/Users/stephenfairhurst/papers/cgp_fairhurst/ligo/localization/data/KAGRA_VRSEB.dat"
data["KAGRA D"] = "/Users/stephenfairhurst/papers/cgp_fairhurst/ligo/localization/data/KAGRA_VRSED.dat"
data["AdV Final DSR"] = "/Users/stephenfairhurst/papers/cgp_fairhurst/ligo/localization/data/AdV_Final_125W_DSR.txt"
data["AdV Final TSR"] = "/Users/stephenfairhurst/papers/cgp_fairhurst/ligo/localization/data/AdV_Final_125W_TSR.txt"
data["AdV NSR"] = "/Users/stephenfairhurst/papers/cgp_fairhurst/ligo/localization/data/AdV_Mid_25W_NSR.txt"

def psd(name):
  if name == "iLIGO":
    f = numpy.arange(30,8192,1./16)
    return ( f, ligo_psd( f ))
  if name == "Virgo":
    f = numpy.arange(30,8192,1./16)
    return ( f, virgo_psd( f ))
  if name == "aLIGO":
    f = pylab.arange(10,8192,1./16)
    return ( f, adligo_psd( f ))
  else:
    f, asd = read_asd(name)
    return(f, asd**2) 

def ligo_psd( f ):
  x = f / 150.0
  p = pow( 4.49 * x, -56.0 ) + 0.16 * pow( x, -4.52 ) + 0.52 + 0.32 * x * x 
  return p

def adligo_psd( f ):
  aligo = pylab.zeros(len(f))
  for i in xrange(len(f)):
    aligo[i] = bench_62_adv_ligo_psd( f[i] )
  return aligo

def read_asd(name):
  if name in data.keys():
    freq, amp_spec = numpy.loadtxt(data[name], unpack=True) 
    return(freq, amp_spec)
  else:
    print("Invalid input")
    return None
    
def bench_62_adv_ligo_psd( f ):
  if(f > 600): 
    p = 4.4e-12 * pow(f,2.2)
  elif(f > 330): 
    p = 0.0000468 * pow(f,-0.409 + pow(f-258, 1.96)/941192) - 7e-7
  elif (f > 85):
    p = 0.00004 * pow(f,-0.408)
  elif (f > 26):
    p = 16.9201  * pow(f,-3.86474) + 6.389e-6
  elif (f > 12): 
    p = 40 * pow(f,-4.100)
  else: 
    p = 1.0e11  * pow(f,-13 + (f-10)/8)
  return p

def virgo_psd( f ):
  x = f / 500.0
  p =  pow(7.87*x,-4.8) + 6./(17. *x)  + 1. + x*x
  return p

def calculate_moments(freq, amp, psd, name=None):
  J_0 = numpy.trapz( amp**2 / psd, freq)
  J_1 = numpy.trapz( freq * amp**2 / psd, freq)
  J_2 = numpy.trapz( freq**2 * amp**2 / psd, freq)
  if name:
    pylab.semilogy(freq, amp**2/psd / J_0, label=name)
  snrsq = 4 * J_0
  f_mean = J_1 / J_0
  f_var = J_2 / J_0 - (J_1)**2 / (J_0)**2
  f_band = pylab.sqrt(f_var)
  return snrsq, f_mean, f_band

def calculate_max_med(freq, amp, psd):
  """
  Calculate median frequency
  """
  j_0 = integrate.cumtrapz( amp**2 / psd, freq)
  j_0 /= j_0[-1]
  return(freq[numpy.argmax(amp**2/psd)], min(freq[j_0>0.5]))
  
def mchirp_moments(freq, amp, psd):
  J_0 = numpy.trapz( amp**2 / psd, freq)
  J_1 = numpy.trapz( freq**(-5./3) * amp**2 / psd, freq)
  J_2 = numpy.trapz( freq**(-10./3) * amp**2 / psd, freq)
  snrsq = 4 * J_0
  f53_mean = J_1 / J_0
  f53_var = J_2 / J_0 - (J_1)**2 / (J_0)**2
  f53_band = pylab.sqrt(f53_var)
  return snrsq, f53_mean, f53_band

def calculate_higher_moments(freq, amp, psd):
  J_0 = numpy.trapz( amp**2 / psd, freq)
  J_1 = numpy.trapz( freq * amp**2 / psd, freq)
  J_2 = numpy.trapz( freq**2 * amp**2 / psd, freq)
  J_3 = numpy.trapz( freq**3 * amp**2 / psd, freq)
  J_4 = numpy.trapz( freq**4 * amp**2 / psd, freq)
  return J_1/J_0, J_2/J_0, J_3/J_0, J_4/J_0 

def calculate_cut_moments(freq, psd):
  """
  Calculate the moments for a bunch of different fcuts
  """
  j_0 = integrate.cumtrapz( freq**(-7./3) / psd, freq)
  j_1 = integrate.cumtrapz( freq**(-4./3) / psd, freq)
  j_2 = integrate.cumtrapz( freq**(-1./3) / psd, freq)
  f_mean = j_1 / j_0
  f_var = j_2 / j_0 - (j_1)**2 / (j_0)**2
  f_band = pylab.sqrt(f_var)
  return j_0, f_mean, f_band

def chirp_moments(freq, psd):
  J = {}
  J7 = numpy.trapz( freq**(-7./3) / psd, freq)
  for x in [-17, -12, -9, -4, -1]:
    J[x] = numpy.trapz( freq**(x/3.) / psd, freq) / J7
  return J

def all_moments(freq, psd, fcut = None):
  if fcut:
    f = freq[freq < fcut]
    p = psd[freq < fcut]
  else:
    f = freq
    p = psd
  J = {}
  J7 = numpy.trapz( f**(-7./3) / p, f)
  for n in xrange(1,18):
    J[n] = numpy.trapz( f**(-n/3.) / p, f) / J7
  return J


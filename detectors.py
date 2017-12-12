from numpy import *
import lal

################################
# define the detectors
################################
def detectors(ifos, india="bangalore", south_africa="sutherland"):
  """
  Set up a dictionary of detector locations and responses. 
  Either put indigo in bangalore or a different site
  """
  location = {}
  response = {}
  # Use cached detectors for known sites:
  lho = lal.lal.CachedDetectors[lal.LALDetectorIndexLHODIFF]

  if "H1" in ifos:
    location["H1"] = lho.location
    response["H1"] = lho.response
  if "H2" in ifos:
    location["H2"] = lho.location
    response["H2"] = lho.response
  if "L1" in ifos:
    llo = lal.CachedDetectors[lal.LALDetectorIndexLLODIFF]
    location["L1"] = llo.location
    response["L1"] = llo.response
  if "V1" in ifos:
    virgo = lal.CachedDetectors[lal.LALDetectorIndexVIRGODIFF]
    location["V1"] = virgo.location
    response["V1"] = virgo.response
  if "XX" in ifos:
    response["XX"] = array([[0.5,0,0],[0,-0.5,0],[0,0,0]],  dtype=float32)
  if "K1" in ifos:
    # KAGRA location:
    # Here is the coordinates
    # 36.25 degree N, 136.718 degree E
    # and 19 degrees from North to West.
    location["K1"], response["K1"] = calc_location_response(136.718, 36.25, -19)
  if "I1" in ifos:
    if india == "bangalore":
      # Could you pl run us the Localisation Plot once more with
      # a location close to Bangalore that is seismically quiet
      # 14 deg 14' N
      # 76 deg 26' E?
      location["I1"], response["I1"] = \
          calc_location_response(76 + 26./60, 14 + 14./60, 270)
    elif india == "gmrt":
      # Here is the coordinates
      # location: 74deg  02' 59" E 19deg 05' 47" N 270.0 deg (W)
      location["I1"], response["I1"] = \
        calc_location_response(74 + 3./60, 19 + 6./60, 270)
  if "S1" in ifos:
    # SOUTH AFRICA
    # from Jeandrew
    # Soetdoring
    # -28.83926,26.098595
    # Sutherland
    # -32.370683,20.691833
    if south_africa == "sutherland":
      location["S1"], response["S1"] = \
        calc_location_response(20.691833, -32.370683, 270)
    elif south_africa == "soetdoring":
      location["S1"], response["S1"] = \
        calc_location_response(26.098595, -28.83926, 270)
  if "ETdet1" in ifos:
    location["ETdet1"], response["ETdet1"] = \
          calc_location_response(76 + 26./60, 14 + 14./60, 270)
  if "ETdet2" in ifos:
    location["ETdet2"], response["ETdet2"] = \
          calc_location_response(76 + 26./60, 14. + 14./60, 270-45)
  if "ETdet3" in ifos:
    location["ETdet3"], response["ETdet3"] = \
          calc_location_response(16 + 26./60, 84. + 14./60, 270)
   
  return( location, response )

def calc_location_response(longitude, latitude, arms):
  """
  Calculate the location and response for a detector with longitude, latitude in degrees
  The angle gives the orientation of the arms and is in degrees from North to East
  """
  phi = radians(longitude)
  theta = radians(latitude)
  angle = radians(arms)
  r = 6.4e6
  location = r * xyz(phi, theta)
  r_hat = location / linalg.norm(location)
  # Take North, project onto earth's surface...
  e_n = array([0,0,1])
  e_n = e_n - r_hat * inner(e_n, r_hat)
  # normalize
  e_n = e_n / linalg.norm(e_n)
  # and calculate east
  e_e = cross(e_n, r_hat)
  # Calculate arm vectors
  u_y = e_e * sin(angle) + e_n * cos(angle)
  u_x = e_e * sin(angle + pi/2) + e_n * cos(angle + pi/2)
  response = array(1./2 * (outer(u_x, u_x) - outer(u_y, u_y)), dtype=float32)
  return location, response
  

################################
# co-ordinate transformations 
################################
def xyz(phi, theta):
  """
  phi, theta -> x,y,z
  """
  x = cos(theta) * cos(phi)
  y = cos(theta) * sin(phi)
  z = sin(theta)
  loc = asarray([x,y,z])
  return(loc)

def phitheta(loc):
  """
  x,y,z -> phi, theta
  """
  x = loc[0]
  y = loc[1]
  z = loc[2]
  r = sqrt(x**2 + y**2 + z**2)
  theta = arcsin(z/r)
  phi = arctan2(y,x)
  return(phi, theta)

def range_8(configuration):
  """
  return the detector ranges for a given configuration
  """
  range_dict_all = {
    "design" : {'H1' : 197.5, 'L1' : 197.5, 'V1': 128.3 }, 
    "o3" : {'H1' : 150, 'L1' : 150, 'V1': 90 }, 
    "2016" : {'H1' : 108, 'L1' : 108, 'V1': 36 }, 
    "early" : {'H1' : 60., 'L1': 60.},
    "half_ligo" : {'H1' : 99, 'L1' : 99, 'V1': 128.3 }, 
    "half_virgo" : {'H1' : 197.5, 'L1' : 197.5, 'V1': 64 }, 
    "nosrm" : {'H1' : 159, 'L1' : 159, 'V1': 109 }, 
    "india" : {'H1' : 197.5, 'L1' : 197.5, 'V1': 128.3, "I1" : 197.5 }, 
    "kagra" : {'H1' : 197.5, 'L1' : 197.5, 'V1': 128.3, "I1" : 197.5 , \
        "K1" : 160.0}, 
    "aligoplus" : {'H1' : 300, 'L1' : 300, 'V1': 130, "K1" : 130.0}, 
    "bala" : {'H1' : 197.5, 'H2' : 197.5, 'L1' : 197.5, 'V1': 128.3, \
        "I1" : 197.5 , "K1" : 160.0}, 
    "sa" : {'H1' : 197.5, 'L1' : 197.5, 'V1': 128.3, "I1" : 197.5 , \
        "K1" : 160.0, "S1":197.5}, 
    "sa2" : {'H1' : 197.5, 'L1' : 197.5, 'V1': 128.3, "I1" : 197.5 , \
        "K1" : 160.0, "S1":197.5}, 
    "steve" : {'H1' : 160.0, 'L1' : 160.0, 'V1': 160.0, "I1" : 160.0 }, 
    "s6vsr2" : {'H1' : 20., 'L1' : 20., 'V1': 8. } ,
    "ET1" : {'H1' : 3*197.5, 'L1' : 3*197.5, 'V1': 3*128.3, 'ETdet1': 1500., 'ETdet2': 1500. }, # Triangular ET
    "ET2" : {'H1' : 3*197.5, 'L1' : 3*197.5, 'V1': 3*128.3, 'ETdet1': 1500., 'ETdet3': 1500. }, # L-shaped at 2 places
  }
  return(range_dict_all[configuration])

def bandwidth(configuration):
  """
  return the detector bandwidths for a given configuration
  """
  bandwidth_dict_all = {
    "design" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9 }, 
    "o3" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9 }, 
    "early"  : {'H1' : 123.7, 'L1' : 123.7 },
    "2016" : {'H1' : 115., 'L1' : 115, 'V1': 89. }, 
    "half_virgo" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9 }, 
    "half_ligo" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9 }, 
    "nosrm" : {'H1' : 43, 'L1' : 43, 'V1': 58 }, 
    "india" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9, "I1" : 117.4 }, 
    "kagra" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9, "I1" : 117.4, \
        "K1" : 89.0 }, 
    "aligoplus" : {'H1' : 150., 'L1' : 150., 'V1': 80., "K1" : 80.0 }, 
    "bala" : {'H1' : 117.4, 'H2' : 117.4, 'L1' : 117.4, 'V1': 148.9, \
        "I1" : 117.4, "K1" : 89.0 }, 
    "sa" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9, "I1" : 117.4, \
        "K1" : 89.0 , "S1": 117.4}, 
    "sa2" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9, "I1" : 117.4, \
        "K1" : 89.0 , "S1": 117.4}, 
    "steve" : {'H1' : 100.0, 'L1' : 100.0, 'V1': 100.0, "I1" : 100.0 }, 
    "s6vsr2" : {'H1' : 100., 'L1' : 100., 'V1': 120. } ,
    "ET1" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9, 'ETdet1': 117.4, 'ETdet2': 117.4 },
    "ET2" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9, 'ETdet1': 117.4, 'ETdet3': 117.4 },
  }
  return(bandwidth_dict_all[configuration])

def fmean(configuration):
  """
  return the detector mean frequencies for a given configuration
  """
  fmean_dict_all = {
    "steve" : {'H1' : 100.0, 'L1' : 100.0, 'V1': 100.0, "I1" : 100.0 }, 
    "early" : {'H1' : 100., 'L1' : 100.}, 
    "2016" : {'H1' : 118., 'L1' : 118., 'V1': 119. }, 
    "design" : {'H1' : 100., 'L1' : 100., 'V1': 130. }, 
    "o3" : {'H1' : 100., 'L1' : 100., 'V1': 130. }, 
    "india" : {'H1' : 100., 'I1' : 100., 'L1' : 100., 'V1': 130. }, 
    "kagra" : {'H1' : 100., 'I1' : 100., 'L1' : 100., 'V1': 130. , "K1":100}, 
    "aligoplus" : {'H1' : 120., 'K1' : 100., 'L1' : 120., 'V1': 100. }, 
    "s6vsr2" : {'H1' : 180., 'L1' : 180., 'V1': 150. },
    "ET1" : {'H1' : 100., 'L1' : 100., 'V1': 130., 'ETdet1':100., 'ETdet2':100 },
    "ET2" : {'H1' : 100., 'L1' : 100., 'V1': 130., 'ETdet1':100., 'ETdet3':100 },
  }
  return(fmean_dict_all[configuration])

def sigma_t(configuration):
  """
  return the timing accuracy.  We use SNR of 10 in LIGO, but scale the expected
  SNR in other detectors based on the range.
  It's just 1/(20 pi sigma_f for LIGO.  
  But 1/(20 pi sigma_f)(r_ligo/r_virgo) for Virgo; 
  5 seconds => no localization from det.
  """
  b = bandwidth(configuration)
  r = range_8(configuration)
  s = {}
  for ifo in r.keys():
    s[ifo] = 1./20/math.pi/b[ifo]*r["H1"]/r[ifo]
  return(s)

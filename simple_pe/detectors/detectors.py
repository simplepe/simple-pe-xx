import numpy as np
from pycbc import detector


################################
# define the detectors
################################

def detectors(ifos):
    """
    Set up a dictionary of detector locations and responses.

    Parameters
    ----------
    ifos: a list of IFOs

    Returns
    -------
    location: a dictionary of detector locations
    response: a dictionary of the detector responses
    """
    location = {}
    response = {}

    for ifo in ifos:
        det = detector.Detector(ifo)
        location[ifo] = det.location
        response[ifo] = det.response

    return location, response

# Need to add EL, EV (L-shaped ET equivalent)
# AL, AV, UL, UV (for 3-et config)
# For CE, use hanford/livingston
# for Voyager use existing sites


def calc_location_response(longitude, latitude, arms):
    """
    Calculate the location and response for a detector with longitude, latitude in degrees
    The angle gives the orientation of the arms and is in degrees from North to East

    Parameters
    ----------
    longitude: the longitude
    latitude: the latitude
    arms: the angle between the arms

    Returns
    -------
     location, response: the detector location and response
    """
    phi = np.radians(longitude)
    theta = np.radians(latitude)
    angle = np.radians(arms)
    r = 6.4e6
    location = r * xyz(phi, theta)
    r_hat = location / np.linalg.norm(location)
    # Take North, project onto earth's surface...
    e_n = np.array([0, 0, 1])
    e_n = e_n - r_hat * np.inner(e_n, r_hat)
    # normalize
    e_n = e_n / np.linalg.norm(e_n)
    # and calculate east
    e_e = np.cross(e_n, r_hat)
    # Calculate arm vectors
    u_y = e_e * np.sin(angle) + e_n * np.cos(angle)
    u_x = e_e * np.sin(angle + np.pi / 2) + e_n * np.cos(angle + np.pi / 2)
    response = np.array(1. / 2 * (np.outer(u_x, u_x) - np.outer(u_y, u_y)), dtype=np.float32)
    return location, response


################################
# co-ordinate transformations
################################
def xyz(phi, theta):
    """
    return cartesian co-ordinates on the unit sphere for a given theta and phi

    Parameters
    ----------
    phi: azimuthal angle
    theta: inclination angle

    Returns
    -------
    loc: array of (x, y, z) locations
    """
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    loc = np.asarray([x, y, z])
    return loc


def phitheta(loc):
    """
    return spherical co-ordinates for a given set of cartesian coordinates

    Parameters
    ----------
    loc: array of (x, y, z) locations

    Returns
    -------
    phi: azimuthal angle
    theta: inclination angle
    """
    x = loc[0]
    y = loc[1]
    z = loc[2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arcsin(z / r)
    phi = np.arctan2(y, x)
    return phi, theta


def range_8(configuration):
    """
    Provide the range for a set of detectors based upon the given configuration

    Parameters
    ----------
    configuration: the name of the network configuration

    Returns
    -------
    range_dict: dictionary of ranges for ifos in network
    """
    range_dict_all = {
        "design": {'H1': 197.5, 'L1': 197.5, 'V1': 128.3},
        "o3": {'H1': 150, 'L1': 150, 'V1': 90},
        "2016": {'H1': 108, 'L1': 108, 'V1': 36},
        "early": {'H1': 60., 'L1': 60.},
        "half_ligo": {'H1': 99, 'L1': 99, 'V1': 128.3},
        "half_virgo": {'H1': 197.5, 'L1': 197.5, 'V1': 64},
        "nosrm": {'H1': 159, 'L1': 159, 'V1': 109},
        "india": {'H1': 197.5, 'L1': 197.5, 'V1': 128.3, "I1": 197.5},
        "kagra": {'H1': 197.5, 'L1': 197.5, 'V1': 128.3, "I1": 197.5,
                  "K1": 160.0},
        # leaving the HLV numbers as Steve wrote, Tagoshi's group used 120 for HL and 60 for V.
        "kagra-o3-8": {'H1': 110, 'L1': 140, 'V1': 50, "K1": 8.0},
        "kagra-o3-15": {'H1': 110, 'L1': 140, 'V1': 50, "K1": 15.0},
        "kagra-o3-25": {'H1': 110, 'L1': 140, 'V1': 50, "K1": 25.0},
        "nokagra-o3": {'H1': 110, 'L1': 140, 'V1': 50},
        "aligoplus": {'H1': 300, 'L1': 300, 'V1': 130, "K1": 130.0},
        "bala": {'H1': 197.5, 'H2': 197.5, 'L1': 197.5, 'V1': 128.3,
                 "I1": 197.5, "K1": 160.0},
        "sa": {'H1': 197.5, 'L1': 197.5, 'V1': 128.3, "I1": 197.5,
               "K1": 160.0, "S1": 197.5},
        "sa2": {'H1': 197.5, 'L1': 197.5, 'V1': 128.3, "I1": 197.5,
                "K1": 160.0, "S1": 197.5},
        "steve": {'H1': 160.0, 'L1': 160.0, 'V1': 160.0, "I1": 160.0},
        "s6vsr2": {'H1': 20., 'L1': 20., 'V1': 8.},
        "ET1": {'H1': 3 * 197.5, 'L1': 3 * 197.5, 'V1': 3 * 128.3, 'ETdet1': 1500., 'ETdet2': 1500.},  # Triangular ET
        "ET2": {'H1': 3 * 197.5, 'L1': 3 * 197.5, 'V1': 3 * 128.3, 'ETdet1': 1500., 'ETdet3': 1500.},
        # L-shaped at 2 places
    }
    return range_dict_all[configuration]


def bandwidth(configuration):
    """
    Provide the bandwidth for a set of detectors based upon the given configuration

    Parameters
    ----------
    configuration: the name of the network configuration

    Returns
    -------
    bandwidth_dict: dictionary of bandwidths for ifos in network
    """
    bandwidth_dict_all = {
        "design": {'H1': 117.4, 'L1': 117.4, 'V1': 148.9},
        "o3": {'H1': 117.4, 'L1': 117.4, 'V1': 148.9},
        "early": {'H1': 123.7, 'L1': 123.7},
        "2016": {'H1': 115., 'L1': 115, 'V1': 89.},
        "half_virgo": {'H1': 117.4, 'L1': 117.4, 'V1': 148.9},
        "half_ligo": {'H1': 117.4, 'L1': 117.4, 'V1': 148.9},
        "nosrm": {'H1': 43, 'L1': 43, 'V1': 58},
        "india": {'H1': 117.4, 'L1': 117.4, 'V1': 148.9, "I1": 117.4},
        "kagra": {'H1': 117.4, 'L1': 117.4, 'V1': 148.9, "I1": 117.4,
                  "K1": 89.0},
        "kagra-o3": {'H1': 117.4, 'L1': 117.4, 'V1': 148.9, "K1": 89.0},
        "kagra-o3-8": {'H1': 117.4, 'L1': 117.4, 'V1': 148.9, "K1": 89.0},
        "kagra-o3-15": {'H1': 117.4, 'L1': 117.4, 'V1': 148.9, "K1": 89.0},
        "kagra-o3-25": {'H1': 117.4, 'L1': 117.4, 'V1': 148.9, "K1": 89.0},
        "nokagra-o3": {'H1': 117.4, 'L1': 117.4, 'V1': 148.9},

        "aligoplus": {'H1': 150., 'L1': 150., 'V1': 80., "K1": 80.0},
        "bala": {'H1': 117.4, 'H2': 117.4, 'L1': 117.4, 'V1': 148.9,
                 "I1": 117.4, "K1": 89.0},
        "sa": {'H1': 117.4, 'L1': 117.4, 'V1': 148.9, "I1": 117.4,
               "K1": 89.0, "S1": 117.4},
        "sa2": {'H1': 117.4, 'L1': 117.4, 'V1': 148.9, "I1": 117.4,
                "K1": 89.0, "S1": 117.4},
        "steve": {'H1': 100.0, 'L1': 100.0, 'V1': 100.0, "I1": 100.0},
        "s6vsr2": {'H1': 100., 'L1': 100., 'V1': 120.},
        "ET1": {'H1': 117.4, 'L1': 117.4, 'V1': 148.9, 'ETdet1': 117.4, 'ETdet2': 117.4},
        "ET2": {'H1': 117.4, 'L1': 117.4, 'V1': 148.9, 'ETdet1': 117.4, 'ETdet3': 117.4},
    }
    return bandwidth_dict_all[configuration]


def fmean(configuration):
    """
    Provide the mean frequency for a set of detectors based upon the given configuration

    Parameters
    ----------
    configuration: the name of the network configuration

    Returns
    -------
    fmean_dict: dictionary of mean frequencies for ifos in network
    """
    fmean_dict_all = {
        "steve": {'H1': 100.0, 'L1': 100.0, 'V1': 100.0, "I1": 100.0},
        "early": {'H1': 100., 'L1': 100.},
        "2016": {'H1': 118., 'L1': 118., 'V1': 119.},
        "design": {'H1': 100., 'L1': 100., 'V1': 130.},
        "o3": {'H1': 100., 'L1': 100., 'V1': 130.},
        "india": {'H1': 100., 'I1': 100., 'L1': 100., 'V1': 130.},
        "kagra": {'H1': 100., 'I1': 100., 'L1': 100., 'V1': 130., "K1": 100},
        "kagra-o3": {'H1': 100., 'L1': 100., 'V1': 130., "K1": 100},
        "kagra-o3-8": {'H1': 100., 'L1': 100., 'V1': 130., "K1": 100},
        "kagra-o3-15": {'H1': 100., 'L1': 100., 'V1': 130., "K1": 100},
        "kagra-o3-25": {'H1': 100., 'L1': 100., 'V1': 130., "K1": 100},
        "nokagra-o3": {'H1': 100., 'L1': 100., 'V1': 130.},
        "aligoplus": {'H1': 120., 'K1': 100., 'L1': 120., 'V1': 100.},
        "s6vsr2": {'H1': 180., 'L1': 180., 'V1': 150.},
        "ET1": {'H1': 100., 'L1': 100., 'V1': 130., 'ETdet1': 100., 'ETdet2': 100},
        "ET2": {'H1': 100., 'L1': 100., 'V1': 130., 'ETdet1': 100., 'ETdet3': 100},
    }
    return fmean_dict_all[configuration]


def sigma_t(configuration):
    """
    return the timing accuracy.  We use SNR of 10 in LIGO, but scale the expected
    SNR in other detectors based on the range.
    It's just 1/(20 pi sigma_f for LIGO.
    But 1/(20 pi sigma_f)(r_ligo/r_virgo) for others ;

    Parameters
    ----------
    configuration: the name of the network configuration

    Returns
    -------
    sigma_t_dict: dictionary of timing accuracies for ifos in network
    """
    b = bandwidth(configuration)
    r = range_8(configuration)
    s = {}
    for ifo in r.keys():
        s[ifo] = 1. / 20 / np.pi / b[ifo] * r["H1"] / r[ifo]
    return s

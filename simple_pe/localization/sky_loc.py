from simple_pe.detectors import detectors
from pycbc import detector
import numpy as np
from scipy import optimize, constants


def chisq_loc(t_r, t_i, d_i, f_band_i):
    """
    Calculate expression (A.3) from 2nd localization paper

    :param: t_theta_phi: the time of arrival and (theta, phi) for source location
    :param: t_i: array with times of arrival in each detector
    :param: d_i: matrix with locations of detectors
    :param: f_band_i: array with bandwidths in each detector
    """
    t = t_r[0]
    r = detectors.xyz(t_r[1], t_r[2])
    chisq = np.sum(((t_i - t) + np.inner(r, d_i) / constants.c) ** 2 / f_band_i ** 2)
    return chisq


def localization_from_timing(ifos, arrival_times, bandwidths):
    """
    Calculate RA and dec based upon time of arrival in a network of ifos

    :param: ifos: list of ifos
    :param: arrival_times: dictionary of arrival times in different ifos
    :param: bandwidths: dictionary of signal bandwidth in each ifo
    :return ra: the right ascension of the signal
    :return dec: the declination of the signal
    """
    times = np.array([arrival_times[ifo] for ifo in ifos])
    f_bands = np.array([bandwidths[ifo] for ifo in ifos])
    det_locations = np.array([detector.Detector(ifo).location for ifo in ifos])
    initial_theta = 1.
    initial_phi = 1.

    out = optimize.minimize(chisq_loc, np.array([0, initial_theta, initial_phi]),
                            args=(times - times.mean(), det_locations, f_bands), tol=1e-12)

    time = out.x[0] + times.mean()
    ra = (out.x[1] + detector.gmst_accurate(time)) % (2 * np.pi)
    dec = out.x[2]

    return ra, dec

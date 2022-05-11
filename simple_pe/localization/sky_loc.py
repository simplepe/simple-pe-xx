from simple_pe.localization import loc
from simple_pe.localization import event
from simple_pe.detectors import detectors


def chisq_loc(t_r, t_i, d_i, f_band_i):
    """
    Calculate expression (A.3) from 2nd localization paper

    :param: t_theta_phi: the time of arrival and unit vector for source location
    :param: t_i: array with times of arrival in each detector
    :param: d_i: matrix with locations of detectors
    :param: f_band_i: array with bandwidths in each detector
    """
    t = t_r[0]
    r = detectors.xyz(t_r[1], t_r[2])
    chisq = np.sum(((t_i - t) + np.inner(r, d_i) / scipy.constants.c) ** 2 / f_band_i ** 2)
    return chisq

def localization_from_timing()
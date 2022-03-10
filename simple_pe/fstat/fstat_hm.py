import numpy as np

# The following functions all convert between physical parameters and f-stat values
# In particular, they do not need anything about a detector or network other than the relative
# sensitivity to the modes, encoded in the sigmas

t = lambda iota: np.tan(iota / 2.)
sqri43 = lambda iota: np.sin(iota / 2.) * np.cos(iota / 2.) ** 5 * (1 - 2 * np.cos(iota))
sqri4m3 = lambda iota: np.sin(iota / 2.) ** 5 * np.cos(iota / 2.) * (2 * np.cos(iota) + 1)

amp = {
    # expressions taken from Mills and Fairhurst, PRD 103, 024042 (2021)
    '22+': lambda iota: (np.cos(iota) ** 2 + 1) / 2.,
    '22x': lambda iota: np.cos(iota),
    '21+': lambda iota: np.sin(iota),
    '21x': lambda iota: np.sin(iota) * np.cos(iota),
    '33+': lambda iota: 2 * np.sin(iota) * amp['22+'](iota),
    '33x': lambda iota: 2 * np.sin(iota) * amp['22x'](iota),
    '32+': lambda iota: 1 - 2 * np.cos(iota) ** 2,
    '32x': lambda iota: 0.5 * (np.cos(iota) - 3 * np.cos(iota) ** 3),
    '44+': lambda iota: 2 * np.sin(iota) **2 * amp['22+'](iota),
    '44x': lambda iota: 2 * np.sin(iota) **2 * amp['22x'](iota),
    '43+': lambda iota: 4 * (sqri43(iota) + sqri4m3(iota)),
    '43x': lambda iota: 4 * (sqri43(iota) - sqri4m3(iota)),
}


def params_to_mode_a(mode, d, cosi, psi, phi=0, sigma=1, d0=1):
    """
    Calculate the mode A params given the physical parameters and a choice of
    d0 to set the overall scaling

    Note, the reference iota for 22 will always be iota = 0, where + and x both equal to one.
    Therefore this factor never appears in the equations. For the other modes, however, there is no
    iota for which both polarizations are equal to 1, but we use their respective values at iota=pi/2
    as reference.
    :param mode: the mode for which to calculate As
    :param d: distance to source
    :param cosi: cos(inclination) of source
    :param psi: polarization of source
    :param phi: coalescence phase of source
    :param sigma: overall scaling of the mode relative to 22 -->
    calculated with at iota = pi/2. (22 at iota = 0).
    :param d0: overall scaling of A's
    """

    try:
        n = len(d)
    except:
        n = 1
    a = np.zeros((n, 5))

    if mode+'+' not in amp.keys():
        print('Invalid mode, choose one of')
        print(amp.keys())
        return a

    iota = np.arccos(cosi)
    a_plus = sigma * d0 / d * amp[mode+'+'](iota)
    a_cross = sigma * d0 / d * amp[mode+'x'](iota)

    mode_m = int(mode[1])

    a[:, 0] = sigma
    a[:, 1] = a_plus * np.cos(mode_m * phi) * np.cos(2 * psi) - a_cross * np.sin(mode_m * phi) * np.sin(2 * psi)
    a[:, 2] = a_plus * np.cos(mode_m * phi) * np.sin(2 * psi) + a_cross * np.sin(mode_m * phi) * np.cos(2 * psi)
    a[:, 3] = - a_plus * np.sin(mode_m * phi) * np.cos(2 * psi) - a_cross * np.cos(mode_m * phi) * np.sin(2 * psi)
    a[:, 4] = - a_plus * np.sin(mode_m * phi) * np.sin(2 * psi) + a_cross * np.cos(mode_m * phi) * np.cos(2 * psi)
    return a


def expected_snr_in_modes(a_dict, f_plus, f_cross):
    """
    Calculate the SNR for a given set of A parameters and network sensitivity.
    :param a_dict: a dictionary, labelled by the modes, of the amplitude parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    """
    f = np.array([np.zeros_like(f_plus), f_plus, f_cross, f_plus, f_cross])
    snr = {}
    total_snrsq = 0
    for mode, a in a_dict.items():
        snr[mode] = np.sqrt(sum(f**2 * a.T**2))
        total_snrsq += snr[mode]**2

    return total_snrsq ** 0.5, snr


def set_snr_in_modes(a_dict, f_plus, f_cross, snr):
    """
    FIXME: for now ignoring all the cross terms.
    rescale distance to give desired SNR, return rescaled as and distance
    :param a_dict: a dictionary, labelled by the modes, of the amplitude parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param snr: the desired SNR
    """
    s = expected_snr_in_modes(a_dict, f_plus, f_cross)
    scaling_factor = snr / s
    a_scale = {}
    for mode, a in a_dict.items():
        a_scale[mode] = a * scaling_factor
        a_scale[mode][:, 0] = a[:, 0]

    return a_scale, scaling_factor


def lost_snr_in_modes(a_hat, a, f_plus, f_cross):
    """
    Calculate the difference in SNRSQ between the true parameters a_hat
    and the templates a for the given modes (ignoring cross terms), and
    network sensitivity f_plus, f_cross
    :param a_hat: the observed F-stat A parameters for the modes
    :param a: the "template" F-stat A parameters for the modes
    :param f_plus: sensitivity to plus polarization
    :param f_cross: sensitivity to cross polarization
    """
    f = np.array([0, f_plus, f_cross, f_plus, f_cross])
    if a_hat.keys() != a.keys():
        print("Require same modes in A and A hat")
        return 0

    a_diff = np.zeros_like(a['22'])
    for mode in a.keys():
        a_diff += a_hat[mode] - a[mode]

    lost_snr, _ = expected_snr_in_modes(a_diff, f_plus, f_cross)

    return lost_snr

import numpy as np

# The following functions all convert between physical parameters and f-stat values
# In particular, they do not need anything about a detector or network.

t = lambda iota: np.tan(iota / 2.)
sqri33 = lambda iota: np.sin(iota / 2.) * np.cos(iota / 2.) ** 5
sqri3m3 = lambda iota: 1 / 2. * np.sin(iota / 2.) ** 4 * np.sin(iota)
sqri21 = lambda iota: 1 / 2. * np.sin(iota) * (np.cos(iota) + 1)
sqri2m1 = lambda iota: np.sin(iota / 2.) ** 2 * np.sin(iota)
sqri32 = lambda iota: np.cos(iota / 2.) ** 4 * (3 * np.cos(iota) - 2)
sqri3m2 = lambda iota: np.sin(iota / 2.) ** 4 * (3 * np.cos(iota) + 2)
sqri43 = lambda iota: np.sin(iota / 2.) * np.cos(iota / 2.) ** 5 * (1 - 2 * np.cos(iota))
sqri4m3 = lambda iota: np.sin(iota / 2.) ** 5 * np.cos(iota / 2.) * (2 * np.cos(iota) + 1)

amp = {
    '22+': lambda iota: (np.cos(iota) ** 2 + 1) / 2.,
    '22x': lambda iota: np.cos(iota),
    #         '22+' : lambda iota: (1+t(iota)**4) / (1+t(iota)**2)**2,
    # I have inserted a root(2) and 4/3. factor for the 44+ and 44x respectively,
    # corresponding to the reciprocal amplitude of each at the reference iota where
    # the sigma_44 should be calculated --> ref_iota=pi/4.
    # NOOOO do the reciprocal of the plus at the maximum so for the 3,3 this is the value of the plus at 0.95531661
    #     and for the 4,4 at pi/2 (=1).
    '44+': lambda iota: np.sin(iota) ** 2 * (np.cos(iota) ** 2 + 1),
    '44x': lambda iota: 2 * np.sin(iota) ** 2 * np.cos(iota),
    #         '44x' : lambda iota: 8*(t(iota)**2+t(iota)**6) / (1+t(iota)**2)**4,
    # I have inserted a root(2)*8/3. and 4 factor for the 33+ and 33x respectively,
    # corresponding to the reciprocal amplitude of each at the reference iota where
    # the sigma_33 should be calculated --> ref_iota=pi/4.

    #     the reciprocal of the plus at the maximum so for the 3,3 (iota= 0.95531661) is 3.6742346141747673
    '33+': lambda iota: 3.6742346141747673 * (sqri33(iota) + sqri3m3(iota)),
    '33x': lambda iota: 3.6742346141747673 * (sqri33(iota) - sqri3m3(iota)),

    '21+': lambda iota: sqri21(iota) + sqri2m1(iota),
    '21x': lambda iota: sqri21(iota) - sqri2m1(iota),
    '43+': lambda iota: 4 * (sqri43(iota) + sqri4m3(iota)),
    '43x': lambda iota: 4 * (sqri43(iota) - sqri4m3(iota)),
    '32+': lambda iota: sqri32(iota) + sqri3m2(iota),
    '32x': lambda iota: sqri32(iota) - sqri3m2(iota)
}


def params_to_a44(d, cosi, psi, phi=0, sigma_44=1., d0=1.):
    """
    Calculate the 44 A params given the physical parameters and a choice of
    d0 to set the overall scaling

    Note, the reference iota for 22 will always be iota = 0, where + and x both equal to one.
    Therefore this factor never appears in the equations. For the 44 mode, however, there is no
    iota for which both polarizations are equal to 1, but we use their respective values at iota=pi/4
    as reference.
    :param d: distance to source
    :param cosi: cos(inclination) of source
    :param psi: polarization of source
    :param phi: coalescence phase of source
    :param sigma_44: overall scaling of 44 mode relative to 22 -->
    calculated with 44 mode at iota = pi/4. (22 at iota = 0).
    :param d0: overall scaling of A's
    """
    iota = np.arccos(cosi)
    a_plus = sigma_44 * d0 / d * amp['44+'](iota)
    a_cross = sigma_44 * d0 / d * amp['44x'](iota)

    try:
        n = len(d)
    except:
        n = 1
    a44 = np.zeros((n, 5))
    a44[:, 0] = sigma_44
    a44[:, 1] = a_plus * np.cos(4 * phi) * np.cos(2 * psi) - a_cross * np.sin(4 * phi) * np.sin(2 * psi)
    a44[:, 2] = a_plus * np.cos(4 * phi) * np.sin(2 * psi) + a_cross * np.sin(4 * phi) * np.cos(2 * psi)
    a44[:, 3] = - a_plus * np.sin(4 * phi) * np.cos(2 * psi) - a_cross * np.cos(4 * phi) * np.sin(2 * psi)
    a44[:, 4] = - a_plus * np.sin(4 * phi) * np.sin(2 * psi) + a_cross * np.cos(4 * phi) * np.cos(2 * psi)
    return a44


def params_to_a33(d, cosi, psi, phi=0, sigma_33=1., d0=1.):
    """
    Calculate the 33 mode A params given the physical parameters and a choice of
    d0 to set the overall scaling

    Note, the reference iota for 22 will always be iota = 0, where + and x both equal to one.
    Therefore this factor never appears in the equations. For the 33 mode, however, there is no
    iota for which both polarizations are equal to 1, but we use their respective values at iota=pi/4
    as reference.
    :param d: distance to source
    :param cosi: cos(inclination) of source
    :param psi: polarization of source
    :param phi: coalescence phase of source
    :param sigma_33: overall scaling of 33 mode relative to 22 -->
    calculated with 33 mode at iota = pi/4. (22 at iota = 0).
    :param d0: overall scaling of A's
    """
    iota = np.arccos(cosi)
    a_plus = sigma_33 * d0 / d * amp['33+'](iota)
    a_cross = sigma_33 * d0 / d * amp['33x'](iota)
    try:
        n = len(d)
    except:
        n = 1
    a33 = np.zeros((n, 5))

    a33[:, 0] = sigma_33
    a33[:, 1] = a_plus * np.cos(3 * phi) * np.cos(2 * psi) - a_cross * np.sin(3 * phi) * np.sin(2 * psi)
    a33[:, 2] = a_plus * np.cos(3 * phi) * np.sin(2 * psi) + a_cross * np.sin(3 * phi) * np.cos(2 * psi)
    a33[:, 3] = - a_plus * np.sin(3 * phi) * np.cos(2 * psi) - a_cross * np.cos(3 * phi) * np.sin(2 * psi)
    a33[:, 4] = - a_plus * np.sin(3 * phi) * np.sin(2 * psi) + a_cross * np.cos(3 * phi) * np.cos(2 * psi)
    return a33


def params_to_a21(d, cosi, psi, phi=0, sigma_21=1., d0=1.):
    """
    Calculate the 21 mode A params given the physical parameters and a choice of
    d0 to set the overall scaling

    Note, the reference iota for 22 will always be iota = 0, where + and x both equal to one.
    Therefore this factor never appears in the equations. For the 33 mode, however, there is no
    iota for which both polarizations are equal to 1, but we use their respective values at iota=pi/4
    as reference.
    :param d: distance to source
    :param cosi: cos(inclination) of source
    :param psi: polarization of source
    :param phi: coalescence phase of source
    :param sigma_21: overall scaling of 33 mode relative to 22 -->
    calculated with 33 mode at iota = pi/4. (22 at iota = 0).
    :param d0: overall scaling of A's
    """
    iota = np.arccos(cosi)
    a_plus = sigma_21 * d0 / d * amp['21+'](iota)
    a_cross = sigma_21 * d0 / d * amp['21x'](iota)

    try:
        n = len(d)
    except:
        n = 1
    a21 = np.zeros((n, 5))

    a21[:, 0] = sigma_21
    a21[:, 1] = a_plus * np.cos(1 * phi) * np.cos(2 * psi) - a_cross * np.sin(1 * phi) * np.sin(2 * psi)
    a21[:, 2] = a_plus * np.cos(1 * phi) * np.sin(2 * psi) + a_cross * np.sin(1 * phi) * np.cos(2 * psi)
    a21[:, 3] = - a_plus * np.sin(1 * phi) * np.cos(2 * psi) - a_cross * np.cos(1 * phi) * np.sin(2 * psi)
    a21[:, 4] = - a_plus * np.sin(1 * phi) * np.sin(2 * psi) + a_cross * np.cos(1 * phi) * np.cos(2 * psi)
    return a21


# The following functions calculate SNRs, likelihoods, etc for a signal, given a network.
# They all work in the dominant polarization (i.e. assuming that the network is described
# by F+, Fx and they're orthogonal)

def expected_snr_2244(a22, a44, f_plus, f_cross):
    """
    Calculate the SNR for a given set of A parameters and network sensitivity.
    :param a22: the F-stat A parameters
    :param a44: the 44 mode A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    """
    f = np.array([np.zeros_like(f_plus), f_plus, f_cross, f_plus, f_cross])
    snrsq = sum(f ** 2 * (a22.T ** 2 + a44.T ** 2))
    return np.sqrt(snrsq)


def expected_snr_223344(a22, a33, a44, f_plus, f_cross):
    """
    Calculate the SNR for a given set of A parameters and network sensitivity.
    :param a22: the F-stat A parameters
    :param a44: the 44 mode A parameters
    :param a33: the 33 mode A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    """
    f = np.array([np.zeros_like(f_plus), f_plus, f_cross, f_plus, f_cross])
    snrsq = sum(f ** 2 * (a22.T ** 2 + a33.T ** 2 + a44.T ** 2))
    return np.sqrt(snrsq)


def expected_snr_all_modes(a22, a33, a44, a21, f_plus, f_cross):
    """
    FIXME: for now ignoring the cross terms.
    Calculate the SNR for a given set of A parameters and network sensitivity.
    :param a22: the F-stat A parameters
    :param a33: the 44 mode A parameters
    :param a44: the 44 mode A parameters
    :param a21: the 21 mode A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    """
    f = np.array([np.zeros_like(f_plus), f_plus, f_cross, f_plus, f_cross])
    snrsq = sum(f ** 2 * (a22.T ** 2 + a33.T ** 2 + a44.T ** 2 + a21.T ** 2))
    return np.sqrt(snrsq)


def expected_snr_in_each_mode(a22, a33, a44, a21, f_plus, f_cross):
    """
    Calculate the SNR for a given set of A parameters and network sensitivity.
    :param a22: the F-stat A parameters
    :param a33: the 33 mode A parameters
    :param a44: the 44 mode A parameters
    :param a21: the 21 mode A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    """
    f = np.array([np.zeros_like(f_plus), f_plus, f_cross, f_plus, f_cross])
    snrsq22 = sum(f ** 2 * (a22.T ** 2))
    snrsq33 = sum(f ** 2 * (a33.T ** 2))
    snrsq44 = sum(f ** 2 * (a44.T ** 2))
    snrsq21 = sum(f ** 2 * (a21.T ** 2))
    return np.sqrt(snrsq22), np.sqrt(snrsq33), np.sqrt(snrsq44), np.sqrt(snrsq21)


def set_snr_2244(a22, a44, f_plus, f_cross, snr):
    """
    rescale distance to give desired SNR, return rescaled as and distance
    :param a22: the F-stat A parameters
    :param a44: the 44 mode A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param snr: the desired SNR
    """
    s = expected_snr_2244(a22, a44, f_plus, f_cross)
    scaling_factor = snr / s
    a22_scale = a22 * scaling_factor
    a44_scale = a44 * scaling_factor
    a22_scale[:, 0] = a22[:, 0]
    a44_scale[:, 0] = a44[:, 0]
    return a22_scale, a44_scale, scaling_factor


def set_snr_223344(a22, a33, a44, f_plus, f_cross, snr):
    """
    rescale distance to give desired SNR, return rescaled as and distance
    :param a22: the F-stat A parameters
    :param a33: the 33 mode A parameters
    :param a44: the 44 mode A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param snr: the desired SNR
    """
    s = expected_snr_223344(a22, a33, a44, f_plus, f_cross)
    scaling_factor = snr / s
    a22_scale = a22 * scaling_factor
    a44_scale = a44 * scaling_factor
    a33_scale = a33 * scaling_factor
    a22_scale[:, 0] = a22[:, 0]
    a44_scale[:, 0] = a44[:, 0]
    a33_scale[:, 0] = a33[:, 0]
    return a22_scale, a33_scale, a44_scale, scaling_factor


def set_snr_all_modes(a22, a33, a44, a21, f_plus, f_cross, snr):
    """
    FIXME: for now ignoring all the cross terms.
    rescale distance to give desired SNR, return rescaled as and distance
    :param a22: the F-stat A parameters
    :param a33: the 33 mode A parameters
    :param a44: the 44 mode A parameters
    :param a21: the 21 mode A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param snr: the desired SNR
    """
    s = expected_snr_all_modes(a22, a33, a44, a21, f_plus, f_cross)
    scaling_factor = snr / s
    a22_scale = a22 * scaling_factor
    a33_scale = a33 * scaling_factor
    a44_scale = a44 * scaling_factor
    a21_scale = a21 * scaling_factor
    a22_scale[:, 0] = a22[:, 0]
    a33_scale[:, 0] = a33[:, 0]
    a44_scale[:, 0] = a44[:, 0]
    a21_scale[:, 0] = a21[:, 0]
    return a22_scale, a33_scale, a44_scale, a21_scale, scaling_factor


def lost_snrsq_all_modes(a22_hat, a22, f_plus, f_cross, a33_hat=0, a33=0, a44_hat=0, a44=0, a21_hat=0, a21=0):
    """
    Calculate the difference in SNRSQ between the true parameters a_hat, k_hat, a33_hat and g_hat
    and the templates a22k, a33 and g, for the 22, 21, 33 and 44 modes respectively (ignoring cross terms), and
    network sensitivity f_plus, f_cross
    :param a22_hat: the observed F-stat A parameters
    :param a22: the "template" F-stat A parameters
    :param f_plus: sensitivity to plus polarization
    :param f_cross: sensitivity to cross polarization
    :param a33_hat: the observed 33 mode A parameters
    :param a33: the "template" 33 mode A parameters
    :param a44_hat: the observed 44 mode A parameters
    :param a44: the "template" 44 mode A parameters
    :param a21_hat: the observed 21 mode A parameters
    :param a21: the "template" 21 mode A parameters
    """
    f = np.array([0, f_plus, f_cross, f_plus, f_cross])
    snrsq = sum(f ** 2 * (a22_hat + a33_hat + a44_hat + a21_hat - a22 - a33 - a44 - a21) ** 2)
    return snrsq

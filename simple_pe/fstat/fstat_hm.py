from numpy import *
from math import cos, sin, tan, sqrt


# The following functions all convert between physical parameters and f-stat values
# In particular, they do not need anything about a detector or network.


t = lambda iota: tan(iota/2.)
sqri33 = lambda iota: sin(iota/2.) * cos(iota/2.)**5
sqri3m3 = lambda iota: 1/2. * sin(iota/2.)**4 * sin(iota)
sqri21 = lambda iota: 1/2. * sin(iota) * (cos(iota)+1)
sqri2m1 = lambda iota: sin(iota/2.)**2*sin(iota)
sqri32 = lambda iota: cos(iota/2.)**4 * (3*cos(iota)-2)
sqri3m2 = lambda iota: sin(iota/2.)**4 * (3*cos(iota)+2)
sqri43 = lambda iota: sin(iota/2.) * cos(iota/2.)**5 * (1-2*cos(iota))
sqri4m3 = lambda iota: sin(iota/2.)**5 * cos(iota/2.) * (2*cos(iota)+1)

amp = {
        '22+' : lambda iota: (cos(iota)**2+1)/2.,
        '22x' : lambda iota: cos(iota),
#         '22+' : lambda iota: (1+t(iota)**4) / (1+t(iota)**2)**2,
    # I have inserted a root(2) and 4/3. factor for the 44+ and 44x respectively,
    # corresponding to the recipricol amplitude of each at the reference iota where
    # the sigma_44 should be calculated --> ref_iota=pi/4.
    # NOOOO do the reciprocal of the plus at the maximum so for the 3,3 this is the value of the plus at 0.95531661
#     and for the 4,4 at pi/2 (=1).
        '44+' : lambda iota: sin(iota)**2 * (cos(iota)**2+1),
        '44x' : lambda iota: 2*sin(iota)**2 * cos(iota),
#         '44x' : lambda iota: 8*(t(iota)**2+t(iota)**6) / (1+t(iota)**2)**4,
    # I have inserted a root(2)*8/3. and 4 factor for the 33+ and 33x respectively,
    # corresponding to the recipricol amplitude of each at the reference iota where
    # the sigma_33 should be calculated --> ref_iota=pi/4.

#     the reciprocal of the plus at the maximum so for the 3,3 (iota= 0.95531661) is 3.6742346141747673
        '33+' : lambda iota: 3.6742346141747673 * (sqri33(iota)+sqri3m3(iota)),
        '33x' : lambda iota: 3.6742346141747673 * (sqri33(iota)-sqri3m3(iota)),

        '21+' : lambda iota: sqri21(iota)+sqri2m1(iota),
        '21x' : lambda iota: sqri21(iota)-sqri2m1(iota),
        '43+' : lambda iota: 4* (sqri43(iota)+sqri4m3(iota)),
        '43x' : lambda iota: 4* (sqri43(iota)-sqri4m3(iota)),
        '32+' : lambda iota: sqri32(iota)+sqri3m2(iota),
        '32x' : lambda iota: sqri32(iota)-sqri3m2(iota)
        }

def params_to_a(d, cosi, psi, phi=0, d0=1.):
    """
    Calculate the F-stat A params given the physical parameters and a choice of
    d0 to set the overall scaling
    :param d: distance to source
    :param cosi: cos(inclination) of source
    :param psi: polarization of source
    :param phi: coalescence phase of source
    :param d0: overall scaling of A's
    """
    a_plus = d0 / d * (1. + cosi ** 2) / 2
    a_cross = d0 / d * cosi
    a = zeros(5)
    a[0] = d0
    a[1] = a_plus * cos(2 * phi) * cos(2 * psi) - a_cross * sin(2 * phi) * sin(2 * psi)
    a[2] = a_plus * cos(2 * phi) * sin(2 * psi) + a_cross * sin(2 * phi) * cos(2 * psi)
    a[3] = - a_plus * sin(2 * phi) * cos(2 * psi) - a_cross * cos(2 * phi) * sin(2 * psi)
    a[4] = - a_plus * sin(2 * phi) * sin(2 * psi) + a_cross * cos(2 * phi) * cos(2 * psi)
    return a

def params_to_g(d, cosi, psi, phi=0, sigma_44=1., d0=1.):
    """
    Calculate the 4422 G params given the physical parameters and a choice of
    d0 to set the overall scaling

    Note, the reference iota for 22 will always be iota = 0, where + and x both equal to one.
    Therefore this factor never appears in the equations. For the 44 mode, however, there is no
    iota for which both polarizations are equal to 1, but we use their respective values at iota=pi/4
    as reference.
    :param d: distance to source
    :param cosi: cos(inclination) of source
    :param psi: polarization of source
    :param phi: coalescence phase of source
    :param sigma_44: overall scaling of 44 mode relative to 22 --> calculated with 44 mode at iota = pi/4. (22 at iota = 0).
    :param d0: overall scaling of A's
    """
    iota = arccos(cosi)
    a_plus = sigma_44 * d0 / d * amp['44+'](iota)
    a_cross = sigma_44 * d0 / d * amp['44x'](iota)
    g = zeros(5)
    g[0] = sigma_44
    g[1] = a_plus * cos(4 * phi) * cos(2 * psi) - a_cross * sin(4 * phi) * sin(2 * psi)
    g[2] = a_plus * cos(4 * phi) * sin(2 * psi) + a_cross * sin(4 * phi) * cos(2 * psi)
    g[3] = - a_plus * sin(4 * phi) * cos(2 * psi) - a_cross * cos(4 * phi) * sin(2 * psi)
    g[4] = - a_plus * sin(4 * phi) * sin(2 * psi) + a_cross * cos(4 * phi) * cos(2 * psi)
    return g

def params_to_j(d, cosi, psi, phi=0, sigma_33=1., d0=1.):
    """
    Calculate the 33 mode J params given the physical parameters and a choice of
    d0 to set the overall scaling

    Note, the reference iota for 22 will always be iota = 0, where + and x both equal to one.
    Therefore this factor never appears in the equations. For the 33 mode, however, there is no
    iota for which both polarizations are equal to 1, but we use their respective values at iota=pi/4
    as reference.
    :param d: distance to source
    :param cosi: cos(inclination) of source
    :param psi: polarization of source
    :param phi: coalescence phase of source
    :param sigma_33: overall scaling of 33 mode relative to 22 --> calculated with 33 mode at iota = pi/4. (22 at iota = 0).
    :param d0: overall scaling of A's
    """
    iota = arccos(cosi)
    a_plus = sigma_33 * d0 / d * amp['33+'](iota)
    a_cross = sigma_33 * d0 / d * amp['33x'](iota)
    j = zeros(5)
    j[0] = sigma_33
    j[1] = a_plus * cos(3 * phi) * cos(2 * psi) - a_cross * sin(3 * phi) * sin(2 * psi)
    j[2] = a_plus * cos(3 * phi) * sin(2 * psi) + a_cross * sin(3 * phi) * cos(2 * psi)
    j[3] = - a_plus * sin(3 * phi) * cos(2 * psi) - a_cross * cos(3 * phi) * sin(2 * psi)
    j[4] = - a_plus * sin(3 * phi) * sin(2 * psi) + a_cross * cos(3 * phi) * cos(2 * psi)
    return j

def params_to_k(d, cosi, psi, phi=0, sigma_21=1., d0=1.):
    """
    Calculate the 21 mode K params given the physical parameters and a choice of
    d0 to set the overall scaling

    Note, the reference iota for 22 will always be iota = 0, where + and x both equal to one.
    Therefore this factor never appears in the equations. For the 33 mode, however, there is no
    iota for which both polarizations are equal to 1, but we use their respective values at iota=pi/4
    as reference.
    :param d: distance to source
    :param cosi: cos(inclination) of source
    :param psi: polarization of source
    :param phi: coalescence phase of source
    :param sigma_33: overall scaling of 33 mode relative to 22 --> calculated with 33 mode at iota = pi/4. (22 at iota = 0).
    :param d0: overall scaling of A's
    """
    iota = arccos(cosi)
    a_plus = sigma_21 * d0 / d * amp['21+'](iota)
    a_cross = sigma_21 * d0 / d * amp['21x'](iota)
    k = zeros(5)
    k[0] = sigma_21
    k[1] = a_plus * cos(1 * phi) * cos(2 * psi) - a_cross * sin(1 * phi) * sin(2 * psi)
    k[2] = a_plus * cos(1 * phi) * sin(2 * psi) + a_cross * sin(1 * phi) * cos(2 * psi)
    k[3] = - a_plus * sin(1 * phi) * cos(2 * psi) - a_cross * cos(1 * phi) * sin(2 * psi)
    k[4] = - a_plus * sin(1 * phi) * sin(2 * psi) + a_cross * cos(1 * phi) * cos(2 * psi)
    return k

def a_to_params(a):
    """
    Calculate the physical parameters based upon the F-stat A parameters.
    :param a: array of f-stat params, entry zero assumed to be d0
    """
    # these variables are what they say [ (a_plus +/- a_cross)^2 ]
    ap_plus_ac_2 = (a[1] + a[4]) ** 2 + (a[2] - a[3]) ** 2
    ap_minus_ac_2 = (a[1] - a[4]) ** 2 + (a[2] + a[3]) ** 2
    a_plus = 0.5 * (sqrt(ap_plus_ac_2) + sqrt(ap_minus_ac_2))
    a_cross = 0.5 * (sqrt(ap_plus_ac_2) - sqrt(ap_minus_ac_2))
    amp = a_plus + sqrt(a_plus ** 2 - a_cross ** 2)
    cosi = a_cross / amp
    d = a[0] / amp
    psi = 0.5 * arctan2(a_plus * a[2] + a_cross * a[3], a_plus * a[1] - a_cross * a[4])
    phi = 0.5 * arctan2(-a_plus * a[3] - a_cross * a[2], a_plus * a[1] - a_cross * a[4])
    return d, cosi, psi, phi


def a_to_circular(a):
    """
    Calculate the circular F-stat A parameters given in Whelan et al 2013
    :param a: array of f-stat params, entry zero assumed to be d0
    """
    a_circ = zeros(5)
    a_circ[0] = a[0]
    a_circ[1] = 0.5 * (a[1] + a[4])
    a_circ[2] = 0.5 * (a[2] - a[3])
    a_circ[3] = 0.5 * (a[1] - a[4])
    a_circ[4] = - 0.5 * (a[2] + a[3])
    return a_circ


def a_to_circ_amp(a):
    """
    Calculate the amplitudes of left/right circularly polarized waveforms
    from the F-stat A parameters
    :param a: array of f-stat params, entry zero assumed to be d0
    """
    a_circ = a_to_circular(a)
    ar_hat = sqrt(a_circ[1] ** 2 + a_circ[2] ** 2)
    al_hat = sqrt(a_circ[3] ** 2 + a_circ[4] ** 2)
    return ar_hat, al_hat


def phase_diff(a):
    """
    Calculate the phase difference (not sure what of)
    :param a: array of f-stat params, entry zero assumed to be d0
    """
    return arctan2(a[1] * a[4] - a[2] * a[3], a[1] * a[2] + a[3] * a[4])


def amp_ratio(a):
    """
    Calculate the amplitude ratio
    :param a: array of f-stat params, entry zero assumed to be d0
    """
    return sqrt((a[1] ** 2 + a[3] ** 2) / (a[2] ** 2 + a[4] ** 2))


# The following functions calculate SNRs, likelihoods, etc for a signal, given a network.
# They all work in the dominant polarization (i.e. assuming that the network is described
# by F+, Fx and they're orthogonal)

def expected_snr(a, f_plus, f_cross):
    """
    Calculate the SNR for a given set of A parameters and network sensitivity.
    :param a: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    """
    f = array([0, f_plus, f_cross, f_plus, f_cross])
    snrsq = sum(f ** 2 * a ** 2)
    return sqrt(snrsq)

def expected_snr_4422(a, g, f_plus, f_cross):
    """
    Calculate the SNR for a given set of A parameters and network sensitivity.
    :param a: the F-stat A parameters
    :param g: the 44 mode G parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    """
    f = array([0, f_plus, f_cross, f_plus, f_cross])
    snrsq = sum(f ** 2 * (a ** 2 + g ** 2))
    return sqrt(snrsq)

def expected_snr_442233(a, g, j, f_plus, f_cross):
    """
    Calculate the SNR for a given set of A parameters and network sensitivity.
    :param a: the F-stat A parameters
    :param g: the 44 mode G parameters
    :param j: the 44 mode J parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    """
    f = array([0, f_plus, f_cross, f_plus, f_cross])
    snrsq = sum(f ** 2 * (a ** 2 + g ** 2 + j ** 2))
    return sqrt(snrsq)

def expected_snr_442233(a, g, j, f_plus, f_cross):
    """
    Calculate the SNR for a given set of A parameters and network sensitivity.
    :param a: the F-stat A parameters
    :param g: the 44 mode G parameters
    :param j: the 44 mode J parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    """
    f = array([0, f_plus, f_cross, f_plus, f_cross])
    snrsq = sum(f ** 2 * (a ** 2 + g ** 2 + j ** 2))
    return sqrt(snrsq)

def expected_snr_all_modes(a, g, j, k, f_plus, f_cross):
    """
    FIXME: for now ignoring the cross terms.
    Calculate the SNR for a given set of A parameters and network sensitivity.
    :param a: the F-stat A parameters
    :param g: the 44 mode G parameters
    :param j: the 44 mode J parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    """
    f = array([0, f_plus, f_cross, f_plus, f_cross])
    snrsq = sum(f ** 2 * (a ** 2 + g ** 2 + j ** 2 + k **2))
    return sqrt(snrsq)

def expected_snr_in_each_mode(a, g, j, k, f_plus, f_cross):
    """
    Calculate the SNR for a given set of A parameters and network sensitivity.
    :param a: the F-stat A parameters
    :param g: the 44 mode G parameters
    :param j: the 44 mode J parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    """
    f = array([0, f_plus, f_cross, f_plus, f_cross])
    snrsq22 = sum(f ** 2 * (a ** 2))
    snrsq33 = sum(f ** 2 * (j ** 2))
    snrsq44 = sum(f ** 2 * (g ** 2))
    snrsq21 = sum(f ** 2 * (k ** 2))
    return sqrt(snrsq22), sqrt(snrsq33), sqrt(snrsq44), sqrt(snrsq21)

def set_snr(a, f_plus, f_cross, snr):
    """
    rescale distance to give desired SNR, return rescaled as and distance
    :param a: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param snr: the desired SNR
    """
    s = expected_snr(a, f_plus, f_cross)
    a_scale = a * snr / s
    a_scale[0] = a[0]
    d_scale = a_to_params(a_scale)[0]
    return a_scale, d_scale

def set_snr_4422(a, g, f_plus, f_cross, snr):
    """
    rescale distance to give desired SNR, return rescaled as and distance
    :param a: the F-stat A parameters
    :param g: the 44 mode G parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param snr: the desired SNR
    """
    s = expected_snr_4422(a, g, f_plus, f_cross)
    scaling_factor = snr / s
    a_scale = a * scaling_factor
    g_scale = g * scaling_factor
    a_scale[0] = a[0]
    g_scale[0] = g[0]
    return a_scale, g_scale, scaling_factor

def set_snr_442233(a, g, j, f_plus, f_cross, snr):
    """
    rescale distance to give desired SNR, return rescaled as and distance
    :param a: the F-stat A parameters
    :param g: the 44 mode G parameters
    :param j: the 44 mode J parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param snr: the desired SNR
    """
    s = expected_snr_442233(a, g, j, f_plus, f_cross)
    scaling_factor = snr / s
    a_scale = a * scaling_factor
    g_scale = g * scaling_factor
    j_scale = j * scaling_factor
    a_scale[0] = a[0]
    g_scale[0] = g[0]
    j_scale[0] = j[0]
    return a_scale, g_scale, j_scale, scaling_factor

def set_snr_all_modes(a, g, j, k, f_plus, f_cross, snr):
    """
    FIXME: for now ignoring all the cross terms.
    rescale distance to give desired SNR, return rescaled as and distance
    :param a: the F-stat A parameters
    :param g: the 44 mode G parameters
    :param j: the 44 mode J parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param snr: the desired SNR
    """
    s = expected_snr_all_modes(a, g, j, k, f_plus, f_cross)
    scaling_factor = snr / s
    a_scale = a * scaling_factor
    g_scale = g * scaling_factor
    j_scale = j * scaling_factor
    k_scale = k * scaling_factor
    a_scale[0] = a[0]
    g_scale[0] = g[0]
    j_scale[0] = j[0]
    k_scale[0] = k[0]
    return a_scale, g_scale, j_scale, k_scale, scaling_factor

def lost_snrsq(a_hat, a, f_plus, f_cross):
    """
    Calculate the difference in SNRSQ between the true parameters a_hat
    and the template a, network sensitivity f_plus, f_cross
    :param a_hat: the observed F-stat A parameters
    :param a: the "template" F-stat A parameters
    :param f_plus: sensitivity to plus polarization
    :param f_cross: sensitivity to cross polarization
    """
    f = array([0, f_plus, f_cross, f_plus, f_cross])
    snrsq = sum(f ** 2 * (a_hat - a) ** 2)
    return snrsq

# def lost_snrsq_4422_modes(a_hat, a, g_hat, g, f_plus, f_cross):
#     """
#     Calculate the difference in SNRSQ between the true parameters a_hat and g_hat
#     and the templates a and g, network sensitivity f_plus, f_cross
#     :param a_hat: the observed F-stat A parameters
#     :param a: the "template" F-stat A parameters
#     :param f_plus: sensitivity to plus polarization
#     :param f_cross: sensitivity to cross polarization
#     """
#     f = array([0, f_plus, f_cross, f_plus, f_cross])
#     sigma_44 = g_hat[0]
#     snrsq = sum(f ** 2 * (a_hat + g_hat - a - g) ** 2)
#     return snrsq

# def lost_snrsq_442233(a_hat, a, g_hat, g, j_hat, j, f_plus, f_cross):
#     """
#     Calculate the difference in SNRSQ between the true parameters a_hat and g_hat
#     and the templates a and g, network sensitivity f_plus, f_cross
#     :param a_hat: the observed F-stat A parameters
#     :param a: the "template" F-stat A parameters
#     :param f_plus: sensitivity to plus polarization
#     :param f_cross: sensitivity to cross polarization
#     """
#     f = array([0, f_plus, f_cross, f_plus, f_cross])
#     snrsq = sum(f ** 2 * (a_hat + g_hat + j_hat - a - g - j) ** 2)
#     return snrsq

def lost_snrsq_all_modes(a_hat, a, f_plus, f_cross, g_hat = 0, g = 0, j_hat = 0, j = 0, k_hat = 0, k = 0):
    """
    Calculate the difference in SNRSQ between the true parameters a_hat, k_hat, j_hat and g_hat
    and the templates a k, j and g, for the 22, 21, 33 and 44 modes respectively (ignoring cross terms), and 
    network sensitivity f_plus, f_cross
    :param a_hat: the observed F-stat A parameters
    :param a: the "template" F-stat A parameters
    :param f_plus: sensitivity to plus polarization
    :param f_cross: sensitivity to cross polarization
    :param g_hat: the observed 44 mode G parameters
    :param g: the "template" 44 mode G parameters
    :param j_hat: the observed 33 mode J parameters
    :param j: the "template" 33 mode J parameters
    :param k_hat: the observed 21 mode K parameters
    :param k: the "template" 21 mode K parameters
    """
    f = array([0, f_plus, f_cross, f_plus, f_cross])
    snrsq = sum(f ** 2 * (a_hat + g_hat + j_hat + k_hat - a - g - j - k) ** 2)
    return snrsq

def circ_project(a, f_plus, f_cross, hand):
    """
    Project the f-stat A parameters to those that would be observed by restricting
    to left or right circular polarization
    :param f_plus: sensitivity to plus polarization
    :param f_cross: sensitivity to cross polarization
    :param hand: one of "left", "right"
    """
    if hand == "right":
        x = 1
    elif hand == "left":
        x = -1
    else:
        raise ValueError("hand must be either left or right")

    f = array([f_plus, f_cross, f_plus, f_cross])

    fa = (a[1:] * f)
    # matrix that projects FA onto circular configuration
    p = array([[f_plus ** 2, 0, 0, x * f_plus * f_cross],
               [0, f_cross ** 2, -x * f_plus * f_cross, 0],
               [0, -x * f_plus * f_cross, f_plus ** 2, 0],
               [x * f_plus * f_cross, 0, 0, f_cross ** 2]])
    p /= (f_plus ** 2 + f_cross ** 2)
    fa_proj = inner(p, fa)
    a_proj = zeros_like(a)
    a_proj[0] = a[0]
    a_proj[1:] = fa_proj / f
    return a_proj

# These functions allow us to go from SNRs to F-stat params and back

def snr_f_to_a(z, f_sig):
    """
    Given the complex SNR and the detector sensitivities, calculate the f-stat A parameters
    :param z: array containing complex snrs for the detectors
    :param f_sig: sensitivity of detectors sigma * (F+, Fx)
    """
    m = zeros((2, 2))
    for f in f_sig:
        m += outer(f, f)
    s_h = inner(z, f_sig.transpose())
    a_max = inner(s_h, linalg.inv(m))
    a = array([1.0, a_max[0].real, a_max[1].real, a_max[0].imag, a_max[1].imag])
    return a

def a_f_to_snr(a, f_plus, f_cross):
    """
    Given the F-stat a parameters and f_plus, f_cross for a detector, calculate the SNR.
    From the Harry-Fairhurst paper, the waveform is $h = A^{\mu} h_{\mu}$ where
    $h_1 = F_{+} h_{0}$, $h_2 = F_{\times} h_{0}$, $h_{3} = F_{+} h_{\pi/2}$,
    $h_{4} = F_{\times} h_{\pi/2}$ and $z = (s | h_{0}) + i (s | h_{\pi/2})$.
    Given the $A^{\mu}$, the expected SNR is:
    $z = F_{+} A^{1} + F_{\times} A^{2} + i( F_{+} A^{3} + F_{\times} A^{4})$
    :param a: F-stat parameters
    :param f_plus: Sensitivity to plus polarization
    :param f_cross: Sensitivity to cross polarization
    """
    z = f_plus * (a[1] + 1j * a[3]) + f_cross * (a[2] + 1j * a[4])
    return z

# Calculate the dominant polarization F+, Fx and the angle between this and the original frame.

def dominant_polarization(f_sig):
    """
    Given the detector responses, compute the dominant polarization F+, Fx and the
    angle that we have to rotate through to get to D.P.
    :param f_sig: sensitivity of detectors: sigma * (F+, Fx)
    """
    m = zeros((2, 2))
    for f in f_sig:
        m += outer(f, f)
    f_cross, f_plus = sort(sqrt(linalg.eig(m)[0]))
    chi = 1./4 * arctan2(2 * m[0,1], m[0,0] - m[1,1])
    return f_plus, f_cross, chi

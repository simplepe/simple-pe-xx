from __future__ import division
from numpy import *
import numpy

# import math functions that will be called with floats, rather than arrays, with the math module
# as it is faster for floats.
from math import cos, sin, tan, sqrt
from scipy import special
from scipy.integrate import quad
import pylab
from simple_pe.fstat import fstat_hm as fstat

t = lambda iota: tan(iota / 2.0)
sqri33 = lambda iota: sin(iota / 2.0) * cos(iota / 2.0) ** 5
sqri3m3 = lambda iota: 1 / 2.0 * sin(iota / 2.0) ** 4 * sin(iota)
sqri21 = lambda iota: 1 / 2.0 * sin(iota) * (cos(iota) + 1)
sqri2m1 = lambda iota: sin(iota / 2.0) ** 2 * sin(iota)
sqri32 = lambda iota: cos(iota / 2.0) ** 4 * (3 * cos(iota) - 2)
sqri3m2 = lambda iota: sin(iota / 2.0) ** 4 * (3 * cos(iota) + 2)
sqri43 = lambda iota: sin(iota / 2.0) * cos(iota / 2.0) ** 5 * (1 - 2 * cos(iota))
sqri4m3 = lambda iota: sin(iota / 2.0) ** 5 * cos(iota / 2.0) * (2 * cos(iota) + 1)

amp = {
    "22+": lambda iota: (cos(iota) ** 2 + 1) / 2.0,
    "22x": lambda iota: cos(iota),
    #         '22+' : lambda iota: (1+t(iota)**4) / (1+t(iota)**2)**2,
    # I have inserted a root(2) and 4/3. factor for the 44+ and 44x respectively,
    # corresponding to the recipricol amplitude of each at the reference iota where
    # the sigma_44 should be calculated --> ref_iota=pi/4.
    # NOOOO do the reciprocal of the plus at the maximum so for the 3,3 this is the value of the plus at 0.95531661
    #     and for the 4,4 at pi/2 (=1).
    "44+": lambda iota: sin(iota) ** 2 * (cos(iota) ** 2 + 1),
    "44x": lambda iota: 2 * sin(iota) ** 2 * cos(iota),
    #         '44x' : lambda iota: 8*(t(iota)**2+t(iota)**6) / (1+t(iota)**2)**4,
    # I have inserted a root(2)*8/3. and 4 factor for the 33+ and 33x respectively,
    # corresponding to the recipricol amplitude of each at the reference iota where
    # the sigma_33 should be calculated --> ref_iota=pi/4.
    #     the reciprocal of the plus at the maximum so for the 3,3 (iota= 0.95531661) is 3.6742346141747673
    "33+": lambda iota: 3.6742346141747673 * (sqri33(iota) + sqri3m3(iota)),
    "33x": lambda iota: 3.6742346141747673 * (sqri33(iota) - sqri3m3(iota)),
    "21+": lambda iota: sqri21(iota) + sqri2m1(iota),
    "21x": lambda iota: sqri21(iota) - sqri2m1(iota),
    "43+": lambda iota: 4 * (sqri43(iota) + sqri4m3(iota)),
    "43x": lambda iota: 4 * (sqri43(iota) - sqri4m3(iota)),
    "32+": lambda iota: sqri32(iota) + sqri3m2(iota),
    "32x": lambda iota: sqri32(iota) - sqri3m2(iota),
}


def like_parts_d_cosi_psi(a_hat, f_plus, f_cross, x, psi):
    """
    calculate the two dimensional likelihood, marginalized over phi
    log-likelihood can be written as:
    1/2(ahat^2 - 2*d0/d * f(x, psi) * cos(2phi - phi0) + (d0/d)^2 g(x,psi))
    return: ahat2, f, g
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param x: cos(inclination)
    :param psi: polarization
    """
    c2p = cos(2 * psi)
    s2p = sin(2 * psi)
    x2 = (1 + x ** 2) / 2
    c2phi_fac = (
        a_hat[1] * f_plus ** 2 * x2 * c2p
        + a_hat[2] * f_cross ** 2 * x2 * s2p
        - a_hat[3] * f_plus ** 2 * x * s2p
        + a_hat[4] * f_cross ** 2 * x * c2p
    )
    s2phi_fac = (
        -a_hat[1] * f_plus ** 2 * x * s2p
        + a_hat[2] * f_cross ** 2 * x * c2p
        - a_hat[3] * f_plus ** 2 * x2 * c2p
        - a_hat[4] * f_cross ** 2 * x2 * s2p
    )
    f = sqrt(c2phi_fac ** 2 + s2phi_fac ** 2)
    g = (
        f_plus ** 2 * x2 ** 2 * c2p ** 2
        + f_cross ** 2 * x2 ** 2 * s2p ** 2
        + f_plus ** 2 * x ** 2 * s2p ** 2
        + f_cross ** 2 * x ** 2 * c2p ** 2
    )
    f_resp = array([0, f_plus, f_cross, f_plus, f_cross])
    ahat2 = sum(f_resp ** 2 * a_hat ** 2)
    return ahat2, f, g


def like_d_cosi_psi(a_hat, f_plus, f_cross, d, x, psi, marg=True):
    """
    Return the likelihood marginalized over phi, using flat (1/2pi) prior
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    :param psi: polarization
    :param marg: do or don't do the marginalization
    :returns: the marginalization factor.  I don't think it includes the exp(rho^2/2) term.
    """
    ahat2, f, g = like_parts_d_cosi_psi(a_hat, f_plus, f_cross, x, psi)
    d0 = a_hat[0]
    # Marginalizing over phi (from zero to 2 pi) gives:
    # 2 pi  i0e(a f) exp(-1/2(ahat^2 - 2 f a + g a^2))
    a = d0 / d
    like = exp(f * a - 0.5 * (ahat2 + g * a ** 2))
    if marg:
        like *= special.i0e(f * a)
    return like


def like_22_d_cosi_psi_phi(a_hat, f_plus, f_cross, d, x, psi, phi):
    """
    Return the un-marginalized likelihood for a 2,2 waveform model.
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    :param psi: polarization
    :param phi: coalescence phase
    """
    ###### for some reason the more compact commented-out code below is slower than
    ###### the longer older code below. TODO: find out why! Profile the code.
    #     a_temp = fstat.params_to_a(d, x, psi, phi)
    #     fresp2 = array([0, f_plus**2, f_cross**2, f_plus**2, f_cross**2])
    #     ahat2 = sum(fresp2 * a_hat ** 2)
    #     atemp2 = sum(fresp2 * a_temp ** 2)
    #     ah_at = sum(fresp2 * a_hat*a_temp)
    #     return exp(ah_at - 0.5 * (atemp2 + ahat2))

    c2p = cos(2 * psi)
    s2p = sin(2 * psi)
    x2 = (1 + x ** 2) / 2
    c2phi_fac = (
        a_hat[1] * f_plus ** 2 * x2 * c2p
        + a_hat[2] * f_cross ** 2 * x2 * s2p
        - a_hat[3] * f_plus ** 2 * x * s2p
        + a_hat[4] * f_cross ** 2 * x * c2p
    )
    s2phi_fac = (
        -a_hat[1] * f_plus ** 2 * x * s2p
        + a_hat[2] * f_cross ** 2 * x * c2p
        - a_hat[3] * f_plus ** 2 * x2 * c2p
        - a_hat[4] * f_cross ** 2 * x2 * s2p
    )
    f_22 = cos(2 * phi) * c2phi_fac + sin(2 * phi) * s2phi_fac
    g_22 = (
        f_plus ** 2 * x2 ** 2 * c2p ** 2
        + f_cross ** 2 * x2 ** 2 * s2p ** 2
        + f_plus ** 2 * x ** 2 * s2p ** 2
        + f_cross ** 2 * x ** 2 * c2p ** 2
    )
    f_resp = array([0, f_plus, f_cross, f_plus, f_cross])
    ahat2 = sum(f_resp ** 2 * a_hat ** 2)

    d0 = a_hat[0]
    a_22 = d0 / d
    like = exp(f_22 * a_22 - 0.5 * (ahat2 + g_22 * a_22 ** 2))
    return like


def like_d_cosi_alpha_33_psi_phi(
    a_hat, j_hat, f_plus, f_cross, d, x, psi, phi, alpha_33_prime
):
    """
    unmarginalized likelihood over five dimensions for signal model containing 22 and 33 modes.
    """
    L22 = like_22_d_cosi_psi_phi(a_hat, f_plus, f_cross, d, x, psi, phi)

    j_temp = fstat.params_to_j(d, x, psi, phi, sigma_33=1)

    f_resp = array([0, f_plus, f_cross, f_plus, f_cross])
    jhat2 = sum(f_resp ** 2 * j_hat ** 2)
    jtemp2 = sum(f_resp ** 2 * j_temp ** 2)

    a = 0.5 * jtemp2
    b = sum(f_resp ** 2 * j_hat * j_temp)
    return L22 * 1 * exp(-a * alpha_33_prime ** 2 + b * alpha_33_prime - 0.5 * jhat2)


def temp_signal_likelihood(a_temp, a_hat, f_resp_2, alpha_hm_prime=1):
    # f_resp = array([0, f_plus, f_cross, f_plus, f_cross])
    ahat2 = sum(f_resp_2 * a_hat ** 2)
    atemp2 = sum(f_resp_2 * a_temp ** 2)

    a = 0.5 * atemp2
    b = sum(f_resp_2 * a_hat * a_temp)
    return exp(-a * alpha_hm_prime ** 2 + b * alpha_hm_prime - 0.5 * ahat2)


def like_d_cosi_alpha_33_alpha_44_psi_phi(
    a_hat, j_hat, g_hat, f_plus, f_cross, d, x, psi, phi, alpha_33_prime, alpha_44_prime
):
    """
    unmarginalized likelihood over five dimensions for signal model
    containing 22, 33 and 44 modes.
    """
    L2233 = like_d_cosi_alpha_33_psi_phi(
        a_hat, j_hat, f_plus, f_cross, d, x, psi, phi, alpha_33_prime
    )

    g_temp = fstat.params_to_g(d, x, psi, phi, sigma_44=1)

    f_resp = array([0, f_plus, f_cross, f_plus, f_cross])
    ghat2 = sum(f_resp ** 2 * g_hat ** 2)
    gtemp2 = sum(f_resp ** 2 * g_temp ** 2)

    a = 0.5 * gtemp2
    b = sum(f_resp ** 2 * g_hat * g_temp)
    return L2233 * 1 * exp(-a * alpha_44_prime ** 2 + b * alpha_44_prime - 0.5 * ghat2)


def like_d_cosi_alpha_33_phi(a_hat, j_hat, f_plus, f_cross, d, x, phi, alpha_33_prime):
    """
    Return the likelihood marginalized over psi, using flat (1/2pi) prior
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    :param psi: polarization
    :param marg: do or don't do the marginalization
    :returns: the marginalization factor.  I don't think it includes the exp(rho^2/2) term.
    """
    integrand = lambda psi: like_d_cosi_alpha_33_psi_phi(
        a_hat, j_hat, f_plus, f_cross, d, x, psi, phi, alpha_33_prime
    )
    l = 2 / pi * quad(integrand, 0, pi / 2, epsrel=1.48, epsabs=1.48e-2)[0]
    return l


def like_d_cosi_alpha_33_alpha_44_phi(
    a_hat, j_hat, g_hat, f_plus, f_cross, d, x, phi, alpha_33_prime, alpha_44_prime
):
    """
    Return the likelihood marginalized over psi, using flat (1/2pi) prior
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    :param psi: polarization
    :param marg: do or don't do the marginalization
    :returns: the marginalization factor.  I don't think it includes the exp(rho^2/2) term.
    """
    integrand = lambda psi: like_d_cosi_alpha_33_alpha_44_psi_phi(
        a_hat,
        j_hat,
        g_hat,
        f_plus,
        f_cross,
        d,
        x,
        psi,
        phi,
        alpha_33_prime,
        alpha_44_prime,
    )
    l = 2 / pi * quad(integrand, 0, pi / 2, epsrel=1.48, epsabs=1.48e-2)[0]
    return l


def like_d_cosi_alpha_33(
    a_hat, j_hat, g_hat, f_plus, f_cross, d, x, alpha_33_prime, alpha_44_prime
):
    """
    Return the likelihood marginalized over phi and psi, with a uniform
    (1/2pi) prior on both
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    """
    l = (
        1
        / (2 * pi)
        * quad(
            lambda phi: like_d_cosi_alpha_33_phi(
                a_hat, j_hat, f_plus, f_cross, d, x, phi, alpha_33_prime
            ),
            0,
            2 * pi,
            epsrel=1.48,
            epsabs=1.48e-2,
        )[0]
    )
    return l


def like_d_cosi_alpha_33_alpha_44(
    a_hat, j_hat, g_hat, f_plus, f_cross, d, x, alpha_33_prime, alpha_44_prime
):
    """
    Return the likelihood marginalized over phi and psi, with a uniform
    (1/2pi) prior on both
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    """
    l = (
        1
        / (2 * pi)
        * quad(
            lambda phi: like_d_cosi_alpha_33_alpha_44_phi(
                a_hat,
                j_hat,
                g_hat,
                f_plus,
                f_cross,
                d,
                x,
                phi,
                alpha_33_prime,
                alpha_44_prime,
            ),
            0,
            2 * pi,
            epsrel=1.48,
            epsabs=1.48e-2,
        )[0]
    )
    return l


def like_d_alpha_33(a_hat, j_hat, f_plus, f_cross, d, alpha_33_prime):
    """
    Return the likelihood marginalized over phi, psi, with a uniform
    (1/2pi) prior on both, and cosi with uniform 1/2 prior.
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    """
    l = (
        1.0
        / 2
        * quad(
            lambda x: like_d_cosi_alpha_33(
                a_hat, j_hat, f_plus, f_cross, d, x, alpha_33_prime
            ),
            -1,
            1,
            epsabs=1e-2,
            epsrel=1e-4,
        )[0]
    )
    return l


def like_d_alpha_33_alpha_44(
    a_hat, j_hat, g_hat, f_plus, f_cross, d, alpha_33_prime, alpha_44_prime
):
    """
    Return the likelihood marginalized over phi, psi, with a uniform
    (1/2pi) prior on both, and cosi with uniform 1/2 prior.
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    """
    l = (
        1.0
        / 2
        * quad(
            lambda x: like_d_cosi_alpha_33_alpha_44(
                a_hat,
                j_hat,
                g_hat,
                f_plus,
                f_cross,
                d,
                x,
                alpha_33_prime,
                alpha_44_prime,
            ),
            -1,
            1,
            epsabs=1e-2,
            epsrel=1e-4,
        )[0]
    )
    return l


def like_d_cosi_alpha_33_psi_phi_with_noise(
    a_hat, j_hat, f_plus, f_cross, d, x, psi, phi, alpha_33_prime
):
    """
    unmarginalized likelihood over five dimensions for signal model containing 22 and 33 modes.
    Containing gaussian noise in the matched filter SNR term that appears in the likelihood.
    """
    f_resp = array([0, f_plus, f_cross, f_plus, f_cross])
    ###### 22 ############
    a_temp = fstat.params_to_a(d, x, psi, phi)
    ahat2 = sum(f_resp ** 2 * a_hat ** 2)
    atemp2 = sum(f_resp ** 2 * a_temp ** 2)

    ###### 33 ############
    j_temp = fstat.params_to_j(d, x, psi, phi, sigma_33=1)
    jhat2 = sum(f_resp ** 2 * j_hat ** 2)
    jtemp2 = sum(f_resp ** 2 * j_temp ** 2) * alpha_33_prime ** 2

    temp_sigma = sqrt(atemp2 + jtemp2)
    characteristic_snr = sqrt(ahat2 + jhat2)
    matched_filter_snr = random.normal(loc=characteristic_snr, scale=1.0)
    dh = matched_filter_snr * temp_sigma
    dd = ahat2 + jhat2
    hh = atemp2 + jtemp2
    return exp(dh - 0.5 * (dd + hh))


def like_d_cosi_alpha_33_phi_with_noise(
    a_hat, j_hat, f_plus, f_cross, d, x, phi, alpha_33_prime
):
    """
    Return the likelihood marginalized over psi, using flat (1/2pi) prior
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    :param psi: polarization
    :param marg: do or don't do the marginalization
    :returns: the marginalization factor.  I don't think it includes the exp(rho^2/2) term.
    """
    integrand = lambda psi: like_d_cosi_alpha_33_psi_phi_with_noise(
        a_hat, j_hat, f_plus, f_cross, d, x, psi, phi, alpha_33_prime
    )
    l = 2 / pi * quad(integrand, 0, pi / 2, epsrel=1.48, epsabs=1.48e-2)[0]
    return l


def like_d_cosi_alpha_33_with_noise(
    a_hat, j_hat, f_plus, f_cross, d, x, alpha_33_prime
):
    """
    Return the likelihood marginalized over phi and psi, with a uniform
    (1/2pi) prior on both
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    """
    l = (
        1
        / (2 * pi)
        * quad(
            lambda phi: like_d_cosi_alpha_33_phi_with_noise(
                a_hat, j_hat, f_plus, f_cross, d, x, phi, alpha_33_prime
            ),
            0,
            2 * pi,
            epsrel=1.48,
            epsabs=1.48e-2,
        )[0]
    )
    return l


def like_d_cosi_alpha_44_psi_phi(
    a_hat, g_hat, f_plus, f_cross, d, x, psi, phi, alpha_44_prime, k=256
):
    """
    unmarginalized likelihood over five dimensions for signal model containing 22 and 44 modes.
    """
    L22 = like_22_d_cosi_psi_phi(a_hat, f_plus, f_cross, d, x, psi, phi)

    g_temp = fstat.params_to_g(d, x, psi, phi, sigma_44=1)

    f_resp = array([0, f_plus, f_cross, f_plus, f_cross])
    ghat2 = sum(f_resp ** 2 * g_hat ** 2)
    gtemp2 = sum(f_resp ** 2 * g_temp ** 2)

    a = 0.5 * gtemp2

    # using a prior k*e^(-k*alpha_44). k is fit to the expected dist of alpha_44 for a salpeter population -->256
    if k:
        b = sum(f_resp ** 2 * g_hat * g_temp) - k
        return (
            L22 * k * exp(-a * alpha_44_prime ** 2 + b * alpha_44_prime - 0.5 * ghat2)
        )
    else:
        b = sum(f_resp ** 2 * g_hat * g_temp)
        return (
            L22 * 1 * exp(-a * alpha_44_prime ** 2 + b * alpha_44_prime - 0.5 * ghat2)
        )


def like_cosi_alpha_44_psi_phi(
    a_hat, g_hat, f_plus, f_cross, x, psi, phi, alpha_44_prime, d_max, k=256
):
    """
    Likelihood marginalized over distance using 3/d_max**3 d**2 dd prior.
    Currently no prior on alpha_44 (k exponential) is implemented.
    """
    a_temp = fstat.params_to_a(1, x, psi, phi)
    g_temp = fstat.params_to_g(1, x, psi, phi, alpha_44_prime)
    fresp2 = array([0, f_plus ** 2, f_cross ** 2, f_plus ** 2, f_cross ** 2])
    ahat2 = sum(fresp2 * a_hat ** 2)
    ghat2 = sum(fresp2 * g_hat ** 2)
    atemp2 = sum(fresp2 * a_temp ** 2)
    gtemp2 = sum(fresp2 * g_temp ** 2)
    ah_at = sum(fresp2 * a_hat * a_temp)
    gh_gt = sum(fresp2 * g_hat * g_temp)

    b = 0.5 * (atemp2 + gtemp2)
    a = ah_at + gh_gt

    like = (
        3
        / d_max ** 3
        * sqrt(pi)
        / 2
        * exp(a / d_max - b / d_max ** 2 - 0.5 * (ghat2 + ahat2))
        / sqrt(4 * b)
        * special.erfcx((2 * b - a * d_max) / (d_max * sqrt(4 * b)))
    )
    return like


def like_cosi_alpha_44_phi(
    a_hat, g_hat, f_plus, f_cross, x, phi, alpha_44_prime, d_max
):
    """
    Return the likelihood marginalized over d and psi, using flat (1/2pi) prior
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    :param psi: polarization
    :param marg: do or don't do the marginalization
    :returns: the marginalization factor.  I don't think it includes the exp(rho^2/2) term.
    """
    integrand = lambda psi: like_cosi_alpha_44_psi_phi(
        a_hat, g_hat, f_plus, f_cross, x, psi, phi, alpha_44_prime, d_max
    )
    l = 2 / pi * quad(integrand, 0, pi / 2, epsrel=1.48, epsabs=1.48e-2)[0]
    return l


def like_cosi_alpha_44(a_hat, g_hat, f_plus, f_cross, x, alpha_44_prime, d_max=1000):
    """
    Return the likelihood marginalized over phi and psi, with a uniform
    (1/2pi) prior on both
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    """
    l = (
        1
        / (2 * pi)
        * quad(
            lambda phi: like_cosi_alpha_44_phi(
                a_hat, g_hat, f_plus, f_cross, x, phi, alpha_44_prime, d_max
            ),
            0,
            2 * pi,
            epsrel=1.48,
            epsabs=1.48e-2,
        )[0]
    )
    return l


def like_marg_over_alpha_44_psi_phi(
    a_hat, g_hat, f_plus, f_cross, d, x, psi, phi, alpha_44_hat, marg
):
    """
    to be used by below function

    int_0^inf L22*k*exp(-a alpha_44^2 + b alpha_44) dalpha44
    = L22 * k * sqrt(pi) * exp(b**2/(4*a))
    * ( erf(b/(2*sqrt(a))) + 1 ) / (2*sqrt(a))

    ---the last line can be re-written using the complementary error
    function, cerf=1-erf, and we can use the exponentially scaled version
    to prevent rounding errors. AND must add the data ghat2 factor to normalize.
    """
    L22 = like_22_d_cosi_psi_phi(a_hat, f_plus, f_cross, d, x, psi, phi)

    g_temp = fstat.params_to_g(d, x, psi, phi, sigma_44=1)
    #     using a prior k*e^(-k*alpha_44). k is fit to the expected dist of alpha_44 for a salpeter population
    k = 256
    #     k=0

    f_resp = array([0, f_plus, f_cross, f_plus, f_cross])
    ghat2 = sum(f_resp ** 2 * g_hat ** 2)
    gtemp2 = sum(f_resp ** 2 * g_temp ** 2)

    a = 0.5 * gtemp2
    b = sum(f_resp ** 2 * g_hat * g_temp) - k
    #     b = sum(f_resp ** 2 * g_hat*g_temp)

    s4a = sqrt(4 * a)
    marg_like = L22 * k * exp(-0.5 * ghat2) * sqrt(pi) / s4a * special.erfcx(-b / s4a)
    #     marg_like = L22 * 1 * exp(-0.5*ghat2) * sqrt(pi)/s4a * special.erfcx(-b/s4a)
    return marg_like


# FIXME:  change the 1 below back to a k after debugging (and uncomment -k above)
#     if marg:
#         marg_like = L22 * 1 * exp(-0.5*ghat2) * sqrt(pi)/s4a * special.erfcx(-b/s4a)
#         return marg_like
#     else: return L22 * 1 * exp(-a*alpha_44_hat**2+b*alpha_44_hat - 0.5*ghat2)


def like_marg_over_alpha_44_phi(
    a_hat, g_hat, f_plus, f_cross, d, x, phi, alpha_44_hat, marg
):
    """
    Return the likelihood marginalized over psi, using flat (1/2pi) prior
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    :param psi: polarization
    :param marg: do or don't do the marginalization
    :returns: the marginalization factor.  I don't think it includes the exp(rho^2/2) term.
    """
    integrand = lambda psi: like_marg_over_alpha_44_psi_phi(
        a_hat, g_hat, f_plus, f_cross, d, x, psi, phi, alpha_44_hat, marg
    )
    l = 2 / pi * quad(integrand, 0, pi / 2, epsrel=1.48, epsabs=1.48e-2)[0]
    return l


def like_marg_over_alpha_44(
    a_hat, g_hat, f_plus, f_cross, d, x, alpha_44_hat, marg=True
):
    """
    Return the likelihood marginalized over phi and psi, with a uniform
    (1/2pi) prior on both
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    """
    l = (
        1
        / (2 * pi)
        * quad(
            lambda phi: like_marg_over_alpha_44_phi(
                a_hat, g_hat, f_plus, f_cross, d, x, phi, alpha_44_hat, marg
            ),
            0,
            2 * pi,
            epsrel=1.48,
            epsabs=1.48e-2,
        )[0]
    )
    return l


# def like_marg_over_alpha_44(a_hat, g_hat, f_plus, f_cross, d, x, psi, alpha_44_prime, alpha_44_max = 0.5):
#     '''
#     Return likelhood marginalized over alpha_44 assuming a uniform prior between 0 and alpha_max
#     FixME: an exponential prior would be nicer and should be doable, look into it.
#     '''
#     a, b, c = like_marg_over_alpha_44_parts(a_hat, g_hat, f_plus, f_cross, d, x, psi, alpha_44_prime)
#     twosqrta = 2*sqrt(a)
#     marg_like = (exp(b**2/(4*a)) * sqrt(pi) * (erf(b/(twosqrta)) \
#                 + erf( (-b + 2*a*c)/(twosqrta) ) ) ) / (twosqrta)
#     return marg_like * exp(a) # check what is left after marginalization - i.e.
#                               # what the exp(a) factor actually is.


def like_d_cosi(a_hat, f_plus, f_cross, d, x):
    """
    Return the likelihood marginalized over phi and psi, with a uniform
    (1/2pi) prior on both
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    """
    l = (
        2
        / pi
        * quad(lambda p: like_d_cosi_psi(a_hat, f_plus, f_cross, d, x, p), 0, pi / 2.0)[
            0
        ]
    )
    return l


def like_d(a_hat, f_plus, f_cross, d):
    """
    Return the likelihood marginalized over phi, psi, with a uniform
    (1/2pi) prior on both, and cosi with uniform 1/2 prior.
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    """
    l = (
        1.0
        / 2
        * quad(
            lambda x: like_d_cosi(a_hat, f_plus, f_cross, d, x),
            -1,
            1,
            epsabs=1e-2,
            epsrel=1e-4,
        )[0]
    )
    return l


def like_d_cosi_phi(a_hat, f_plus, f_cross, d, x, phi, marg=True):
    """
    Return the likelihood marginalized over psi, using flat (1/2pi) prior
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    :param psi: polarization
    :param marg: do or don't do the marginalization
    :returns: the marginalization factor.  I don't think it includes the exp(rho^2/2) term.
    """
    ahat2, f, g = like_parts_d_cosi_phi(a_hat, f_plus, f_cross, x, phi)
    d0 = a_hat[0]
    # Marginalizing over psi (from zero to 2 pi) gives:
    # 2 pi  i0e(a f) exp(-1/2(ahat^2 - 2 f a + g a^2))
    a = d0 / d
    like = exp(f * a - 0.5 * (ahat2 + g * a ** 2))
    if marg:
        like *= special.i0e(f * a)
    return like


def like_d_cosi_2(a_hat, f_plus, f_cross, d, x):
    """
    Return the likelihood marginalized over phi, using flat (1/2pi) prior
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    :param psi: polarization
    :param marg: do or don't do the marginalization
    :returns: the marginalization factor.  I don't think it includes the exp(rho^2/2) term.
    """
    integrand = lambda phi: like_d_cosi_phi(a_hat, f_plus, f_cross, d, x, phi)
    l = 1 / (pi) * quad(integrand, 0, pi)[0]
    return l


############################# likelihood of waveform containing 22 and 44 modes ######################


def l4422_parts_d_cosi_psi(a_hat, g_hat, f_plus, f_cross, d, x, psi, phi):
    """
    calculate the two dimensional likelihood, marginalized over phi
    log-likelihood can be written as:
    1/2(ahat^2 - 2*d0/d * f(x, psi) * cos(2phi - phi0) + (d0/d)^2 g(x,psi))
    return: ahat2, f, g
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param x: cos(inclination)
    :param psi: polarization
    """
    ########################### 22 mode #####################
    c2p = cos(2 * psi)
    s2p = sin(2 * psi)
    x2 = (1 + x ** 2) / 2
    c2phi_fac = (
        a_hat[1] * f_plus ** 2 * x2 * c2p
        + a_hat[2] * f_cross ** 2 * x2 * s2p
        - a_hat[3] * f_plus ** 2 * x * s2p
        + a_hat[4] * f_cross ** 2 * x * c2p
    )
    s2phi_fac = (
        -a_hat[1] * f_plus ** 2 * x * s2p
        + a_hat[2] * f_cross ** 2 * x * c2p
        - a_hat[3] * f_plus ** 2 * x2 * c2p
        - a_hat[4] * f_cross ** 2 * x2 * s2p
    )
    #     weight = sqrt(c2phi_fac ** 2 + s2phi_fac ** 2)
    f_22 = cos(2 * phi) * c2phi_fac + sin(2 * phi) * s2phi_fac
    g_22 = (
        f_plus ** 2 * x2 ** 2 * c2p ** 2
        + f_cross ** 2 * x2 ** 2 * s2p ** 2
        + f_plus ** 2 * x ** 2 * s2p ** 2
        + f_cross ** 2 * x ** 2 * c2p ** 2
    )
    f_resp = array([0, f_plus, f_cross, f_plus, f_cross])
    ahat2 = sum(f_resp ** 2 * a_hat ** 2)

    ########################### 44 mode #####################

    iota = arccos(x)
    sin2 = 1 - x ** 2
    gp = amp["44+"](iota)  # 2 * sin2 * x2 #
    gc = amp["44x"](iota)  # 2 * sin2 * x  #
    c4phi_fac = (
        g_hat[1] * f_plus ** 2 * gp * c2p
        + g_hat[2] * f_cross ** 2 * gp * s2p
        - g_hat[3] * f_plus ** 2 * gc * s2p
        + g_hat[4] * f_cross ** 2 * gc * c2p
    )
    s4phi_fac = (
        -g_hat[1] * f_plus ** 2 * gc * s2p
        + g_hat[2] * f_cross ** 2 * gc * c2p
        - g_hat[3] * f_plus ** 2 * gp * c2p
        - g_hat[4] * f_cross ** 2 * gp * s2p
    )
    #     weight = sqrt(c2phi_fac ** 2 + s2phi_fac ** 2)
    f_44 = cos(4 * phi) * c4phi_fac + sin(4 * phi) * s4phi_fac

    ######## THINK: should the g factor be changed to incorporate phi not equal to zero?? #######
    #     maybe this is the reason for descrepency in the integrals before?
    #     for now I am ignoring this, perhaps it's fine as the snr is independent of phi_0?

    g_44 = (
        f_plus ** 2 * gp ** 2 * c2p ** 2
        + f_cross ** 2 * gp ** 2 * s2p ** 2
        + f_plus ** 2 * gc ** 2 * s2p ** 2
        + f_cross ** 2 * gc ** 2 * c2p ** 2
    )
    #     f_resp = array([0, f_plus, f_cross, f_plus, f_cross]) # defined earlier

    ghat2 = sum(f_resp ** 2 * g_hat ** 2)

    d0 = a_hat[0]
    sigma_44 = g_hat[0]
    # idea: could have an assertion to check that d0 is the same for g_hat and a_hat...
    a_22 = d0 / d
    a_44 = sigma_44 * a_22
    like = exp(
        f_22 * a_22
        - 0.5 * (ahat2 + g_22 * a_22 ** 2)
        + f_44 * a_44
        - 0.5 * (ghat2 + g_44 * a_44 ** 2)
    )

    return like


def l4422_d_cosi_psi(a_hat, g_hat, f_plus, f_cross, d, x, psi, marg=True):
    """
    Return the likelihood marginalized over phi, using flat (1/2pi) prior
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    :param psi: polarization
    :param marg: do or don't do the marginalization
    :returns: the marginalization factor.  I don't think it includes the exp(rho^2/2) term.
    """
    integrand = lambda phi: l4422_parts_d_cosi_psi(
        a_hat, g_hat, f_plus, f_cross, d, x, psi, phi
    )
    l = 1 / (pi) * quad(integrand, 0, pi)[0]
    return l


def l4422_d_cosi(a_hat, g_hat, f_plus, f_cross, d, x):
    """
    Return the likelihood marginalized over phi and psi, with a uniform
    (1/2pi) prior on both
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    """
    l = (
        2
        / pi
        * quad(
            lambda p: l4422_d_cosi_psi(a_hat, g_hat, f_plus, f_cross, d, x, p),
            0,
            pi / 2,
        )[0]
    )
    return l


######################## 33 mode too ##################################################


def lhm_parts_d_cosi_psi(
    a_hat,
    f_plus,
    f_cross,
    d,
    x,
    psi,
    phi,
    g_hat=None,
    j_hat=None,
    k_hat=None,
    cross_term_alphas={},
):
    """
    calculate the two dimensional likelihood, marginalized over phi
    log-likelihood can be written as:
    1/2(ahat^2 - 2*d0/d * f(x, psi) * cos(2phi - phi0) + (d0/d)^2 g(x,psi))
    return: ahat2, f, g
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param x: cos(inclination)
    :param psi: polarization
    :param g_hat: the 44 mode G Params
    :param j_hat: the 33 mode J Params
    :param k_hat: the 21 mode K Params
    :param cross_term_alphas: dictionary containing a tuple of the additional pair of alpha values required for each mode (one each at 0 and pi/2. phase offset).
    """
    ########################### 22 mode #####################

    c2p = cos(2 * psi)
    s2p = sin(2 * psi)
    x2 = (1 + x ** 2) / 2
    c2phi_fac = (
        a_hat[1] * f_plus ** 2 * x2 * c2p
        + a_hat[2] * f_cross ** 2 * x2 * s2p
        - a_hat[3] * f_plus ** 2 * x * s2p
        + a_hat[4] * f_cross ** 2 * x * c2p
    )
    s2phi_fac = (
        -a_hat[1] * f_plus ** 2 * x * s2p
        + a_hat[2] * f_cross ** 2 * x * c2p
        - a_hat[3] * f_plus ** 2 * x2 * c2p
        - a_hat[4] * f_cross ** 2 * x2 * s2p
    )
    f_22 = cos(2 * phi) * c2phi_fac + sin(2 * phi) * s2phi_fac
    g_22 = (
        f_plus ** 2 * x2 ** 2 * c2p ** 2
        + f_cross ** 2 * x2 ** 2 * s2p ** 2
        + f_plus ** 2 * x ** 2 * s2p ** 2
        + f_cross ** 2 * x ** 2 * c2p ** 2
    )
    f_resp = array([0, f_plus, f_cross, f_plus, f_cross])
    ahat2 = sum(f_resp ** 2 * a_hat ** 2)

    d0 = a_hat[0]
    a_22 = d0 / d
    like = exp(f_22 * a_22 - 0.5 * (ahat2 + g_22 * a_22 ** 2))

    ########################### 44 mode #####################
    if type(g_hat) == numpy.ndarray:
        #     if True:
        #     iota = arccos(x)
        sin2 = 1 - x ** 2
        gp = 2 * sin2 * x2  # amp['44+'](iota)
        gc = 2 * sin2 * x  # amp['44x'](iota)
        c4phi_fac = (
            g_hat[1] * f_plus ** 2 * gp * c2p
            + g_hat[2] * f_cross ** 2 * gp * s2p
            - g_hat[3] * f_plus ** 2 * gc * s2p
            + g_hat[4] * f_cross ** 2 * gc * c2p
        )
        s4phi_fac = (
            -g_hat[1] * f_plus ** 2 * gc * s2p
            + g_hat[2] * f_cross ** 2 * gc * c2p
            - g_hat[3] * f_plus ** 2 * gp * c2p
            - g_hat[4] * f_cross ** 2 * gp * s2p
        )
        f_44 = cos(4 * phi) * c4phi_fac + sin(4 * phi) * s4phi_fac

        g_44 = (
            f_plus ** 2 * gp ** 2 * c2p ** 2
            + f_cross ** 2 * gp ** 2 * s2p ** 2
            + f_plus ** 2 * gc ** 2 * s2p ** 2
            + f_cross ** 2 * gc ** 2 * c2p ** 2
        )
        ghat2 = sum(f_resp ** 2 * g_hat ** 2)

        sigma_44 = g_hat[0]
        a_44 = sigma_44 * a_22
        like *= exp(f_44 * a_44 - 0.5 * (ghat2 + g_44 * a_44 ** 2))

    ########################### 33 mode #####################
    if type(j_hat) == numpy.ndarray:
        iota = arccos(x)
        jp = amp["33+"](iota)
        jc = amp["33x"](iota)
        c3phi_fac = (
            j_hat[1] * f_plus ** 2 * jp * c2p
            + j_hat[2] * f_cross ** 2 * jp * s2p
            - j_hat[3] * f_plus ** 2 * jc * s2p
            + j_hat[4] * f_cross ** 2 * jc * c2p
        )
        s3phi_fac = (
            -j_hat[1] * f_plus ** 2 * jc * s2p
            + j_hat[2] * f_cross ** 2 * jc * c2p
            - j_hat[3] * f_plus ** 2 * jp * c2p
            - j_hat[4] * f_cross ** 2 * jp * s2p
        )
        f_33 = cos(3 * phi) * c3phi_fac + sin(3 * phi) * s3phi_fac
        g_33 = (
            f_plus ** 2 * jp ** 2 * c2p ** 2
            + f_cross ** 2 * jp ** 2 * s2p ** 2
            + f_plus ** 2 * jc ** 2 * s2p ** 2
            + f_cross ** 2 * jc ** 2 * c2p ** 2
        )
        jhat2 = sum(f_resp ** 2 * j_hat ** 2)

        sigma_33 = j_hat[0]
        a_33 = sigma_33 * a_22
        like *= exp(f_33 * a_33 - 0.5 * (jhat2 + g_33 * a_33 ** 2))

    ########################### 21 mode #####################
    if type(k_hat) == numpy.ndarray:
        iota = arccos(x)
        kp = amp["21+"](iota)
        kc = amp["21x"](iota)
        c1phi_fac = (
            k_hat[1] * f_plus ** 2 * kp * c2p
            + k_hat[2] * f_cross ** 2 * kp * s2p
            - k_hat[3] * f_plus ** 2 * kc * s2p
            + k_hat[4] * f_cross ** 2 * kc * c2p
        )
        s1phi_fac = (
            -k_hat[1] * f_plus ** 2 * kc * s2p
            + k_hat[2] * f_cross ** 2 * kc * c2p
            - k_hat[3] * f_plus ** 2 * kp * c2p
            - k_hat[4] * f_cross ** 2 * kp * s2p
        )
        f_21 = cos(1 * phi) * c1phi_fac + sin(1 * phi) * s1phi_fac
        g_21 = (
            f_plus ** 2 * kp ** 2 * c2p ** 2
            + f_cross ** 2 * kp ** 2 * s2p ** 2
            + f_plus ** 2 * kc ** 2 * s2p ** 2
            + f_cross ** 2 * kc ** 2 * c2p ** 2
        )
        khat2 = sum(f_resp ** 2 * k_hat ** 2)

        sigma_21 = k_hat[0]
        a_21 = sigma_21 * a_22
        like *= exp(f_21 * a_21 - 0.5 * (khat2 + g_21 * a_21 ** 2))

    ########################### 22_21 cross terms ###################
    if "22_21" in cross_term_alphas.keys():

        alpha_2221, alpha_2221_i = cross_term_alphas["22_21"]
        fp2 = (f_plus * alpha_2221) ** 2
        fc2 = (f_cross * alpha_2221) ** 2
        fp2_i = (f_plus * alpha_2221_i) ** 2
        fc2_i = (f_cross * alpha_2221_i) ** 2

        a_temp, k_temp = (
            fstat.params_to_a(d, x, psi, phi),
            fstat.params_to_k(d, x, psi, phi, sigma_21=1),
        )
        # a_hat * k:
        f_22_21 = (
            fp2 * (a_hat[1] * k_temp[1] + a_hat[3] * k_temp[3])
            + fc2 * (a_hat[2] * k_temp[2] + a_hat[4] * k_temp[4])
            + fp2_i * (a_hat[1] * k_temp[3] - a_hat[3] * k_temp[1])
            + fc2_i * (a_hat[2] * k_temp[4] - a_hat[4] * k_temp[2])
            # k_hat * a:
            + (
                +fp2 * (k_hat[1] * a_temp[1] + k_hat[3] * a_temp[3])
                + fc2 * (k_hat[2] * a_temp[2] + k_hat[4] * a_temp[4])
                + fp2_i * (k_hat[1] * a_temp[3] - k_hat[3] * a_temp[1])
                + fc2_i * (k_hat[2] * a_temp[4] - k_hat[4] * a_temp[2])
            )
            / sigma_21  # (must divide by alpha_21 factor in k_hat)
        )

        g_22_21 = 2 * (
            fp2 * (a_temp[1] * k_temp[1] + a_temp[3] * k_temp[3])
            + fc2 * (a_temp[2] * k_temp[2] + a_temp[4] * k_temp[4])
            + fp2_i * (a_temp[1] * k_temp[3] - a_temp[3] * k_temp[1])
            + fc2_i * (a_temp[2] * k_temp[4] - a_temp[4] * k_temp[2])
        )

        ahat_khat = (
            2
            * (
                fp2 * (a_hat[1] * k_hat[1] + a_hat[3] * k_hat[3])
                + fc2 * (a_hat[2] * k_hat[2] + a_hat[4] * k_hat[4])
                + fp2_i * (a_hat[1] * k_hat[3] - a_hat[3] * k_hat[1])
                + fc2_i * (a_hat[2] * k_hat[4] - a_hat[4] * k_hat[2])
            )
            / sigma_21
        )  # (must divide by alpha_21 factor in k_hat)

        like *= exp(f_22_21 * a_22 - 0.5 * (ahat_khat + g_22_21 * a_22 ** 2))

    ########################### 22_33 cross terms ###################
    if "22_33" in cross_term_alphas.keys():

        alpha_2233, alpha_2233_i = cross_term_alphas["22_33"]
        fp2 = (f_plus * alpha_2233) ** 2
        fc2 = (f_cross * alpha_2233) ** 2
        fp2_i = (f_plus * alpha_2233_i) ** 2
        fc2_i = (f_cross * alpha_2233_i) ** 2

        a_temp, j_temp = (
            fstat.params_to_a(d, x, psi, phi),
            fstat.params_to_j(d, x, psi, phi, sigma_33=1),
        )
        # a_hat * j:
        f_22_33 = (
            fp2 * (a_hat[1] * j_temp[1] + a_hat[3] * j_temp[3])
            + fc2 * (a_hat[2] * j_temp[2] + a_hat[4] * j_temp[4])
            + fp2_i * (a_hat[1] * j_temp[3] - a_hat[3] * j_temp[1])
            + fc2_i * (a_hat[2] * j_temp[4] - a_hat[4] * j_temp[2])
            # j_hat * a:
            + (
                +fp2 * (j_hat[1] * a_temp[1] + j_hat[3] * a_temp[3])
                + fc2 * (j_hat[2] * a_temp[2] + j_hat[4] * a_temp[4])
                + fp2_i * (j_hat[1] * a_temp[3] - j_hat[3] * a_temp[1])
                + fc2_i * (j_hat[2] * a_temp[4] - j_hat[4] * a_temp[2])
            )
            / sigma_33  # (must divide by alpha_33 factor in j_hat)
        )

        g_22_33 = 2 * (
            fp2 * (a_temp[1] * j_temp[1] + a_temp[3] * j_temp[3])
            + fc2 * (a_temp[2] * j_temp[2] + a_temp[4] * j_temp[4])
            + fp2_i * (a_temp[1] * j_temp[3] - a_temp[3] * j_temp[1])
            + fc2_i * (a_temp[2] * j_temp[4] - a_temp[4] * j_temp[2])
        )

        ahat_jhat = (
            2
            * (
                fp2 * (a_hat[1] * j_hat[1] + a_hat[3] * j_hat[3])
                + fc2 * (a_hat[2] * j_hat[2] + a_hat[4] * j_hat[4])
                + fp2_i * (a_hat[1] * j_hat[3] - a_hat[3] * j_hat[1])
                + fc2_i * (a_hat[2] * j_hat[4] - a_hat[4] * j_hat[2])
            )
            / sigma_33
        )  # (must divide by alpha_33 factor in j_hat)

        like *= exp(f_22_33 * a_22 - 0.5 * (ahat_jhat + g_22_33 * a_22 ** 2))

    ########################### 22_44 cross terms ###################
    if "22_44" in cross_term_alphas.keys():

        alpha_2244, alpha_2244_i = cross_term_alphas["22_44"]
        fp2 = (f_plus * alpha_2244) ** 2
        fc2 = (f_cross * alpha_2244) ** 2
        fp2_i = (f_plus * alpha_2244_i) ** 2
        fc2_i = (f_cross * alpha_2244_i) ** 2

        a_temp, g_temp = (
            fstat.params_to_a(d, x, psi, phi),
            fstat.params_to_g(d, x, psi, phi, sigma_44=1),
        )
        # a_hat * g:
        f_22_44 = (
            fp2 * (a_hat[1] * g_temp[1] + a_hat[3] * g_temp[3])
            + fc2 * (a_hat[2] * g_temp[2] + a_hat[4] * g_temp[4])
            + fp2_i * (a_hat[1] * g_temp[3] - a_hat[3] * g_temp[1])
            + fc2_i * (a_hat[2] * g_temp[4] - a_hat[4] * g_temp[2])
            # g_hat * a:
            + (
                +fp2 * (g_hat[1] * a_temp[1] + g_hat[3] * a_temp[3])
                + fc2 * (g_hat[2] * a_temp[2] + g_hat[4] * a_temp[4])
                + fp2_i * (g_hat[1] * a_temp[3] - g_hat[3] * a_temp[1])
                + fc2_i * (g_hat[2] * a_temp[4] - g_hat[4] * a_temp[2])
            )
            / sigma_44  # (must divide by alpha_44 factor in g_hat)
        )

        g_22_44 = 2 * (
            fp2 * (a_temp[1] * g_temp[1] + a_temp[3] * g_temp[3])
            + fc2 * (a_temp[2] * g_temp[2] + a_temp[4] * g_temp[4])
            + fp2_i * (a_temp[1] * g_temp[3] - a_temp[3] * g_temp[1])
            + fc2_i * (a_temp[2] * g_temp[4] - a_temp[4] * g_temp[2])
        )

        ahat_ghat = (
            2
            * (
                fp2 * (a_hat[1] * g_hat[1] + a_hat[3] * g_hat[3])
                + fc2 * (a_hat[2] * g_hat[2] + a_hat[4] * g_hat[4])
                + fp2_i * (a_hat[1] * g_hat[3] - a_hat[3] * g_hat[1])
                + fc2_i * (a_hat[2] * g_hat[4] - a_hat[4] * g_hat[2])
            )
            / sigma_44
        )  # (must divide by alpha_44 factor in g_hat)

        like *= exp(f_22_44 * a_22 - 0.5 * (ahat_ghat + g_22_44 * a_22 ** 2))

    return like


def lhm_d_cosi_psi(
    a_hat,
    f_plus,
    f_cross,
    d,
    x,
    psi,
    g_hat=None,
    j_hat=None,
    k_hat=None,
    cross_term_alphas={},
):
    """
    Return the likelihood marginalized over phi, using flat (1/2pi) prior
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    :param psi: polarization
    :param marg: do or don't do the marginalization
    :returns: the marginalization factor.  I don't think it includes the exp(rho^2/2) term.
    """
    integrand = lambda phi: lhm_parts_d_cosi_psi(
        a_hat, f_plus, f_cross, d, x, psi, phi, g_hat, j_hat, k_hat, cross_term_alphas
    )
    l = 1 / (pi) * quad(integrand, 0, pi)[0]  # , epsrel=1.48,epsabs=1.48e-2)[0]
    return l


def lhm_d_cosi_phi(
    a_hat,
    f_plus,
    f_cross,
    d,
    x,
    phi,
    g_hat=None,
    j_hat=None,
    k_hat=None,
    cross_term_alphas={},
):
    """
    Return the likelihood marginalized over psi, using flat (1/2pi) prior
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    :param psi: polarization
    :param marg: do or don't do the marginalization
    :returns: the marginalization factor.  I don't think it includes the exp(rho^2/2) term.
    """
    integrand = lambda psi: lhm_parts_d_cosi_psi(
        a_hat, f_plus, f_cross, d, x, psi, phi, g_hat, j_hat, k_hat, cross_term_alphas
    )
    l = 2 / pi * quad(integrand, 0, pi / 2, epsrel=1.48, epsabs=1.48e-2)[0]
    return l


def lhm_d_cosi(
    a_hat,
    f_plus,
    f_cross,
    d,
    x,
    g_hat=None,
    j_hat=None,
    k_hat=None,
    cross_term_alphas={},
):
    """
    Return the likelihood marginalized over phi and psi, with a uniform
    (1/2pi) prior on both
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    """
    l = (
        1
        / (2 * pi)
        * quad(
            lambda phi: lhm_d_cosi_phi(
                a_hat,
                f_plus,
                f_cross,
                d,
                x,
                phi,
                g_hat,
                j_hat,
                k_hat,
                cross_term_alphas,
            ),
            0,
            2 * pi,
            epsrel=1.48,
            epsabs=1.48e-2,
        )[0]
    )
    return l


# def lhm_d_cosi(a_hat, f_plus, f_cross, d, x, g_hat=None, j_hat=None):
#     """
#     Return the likelihood marginalized over phi and psi, with a uniform
#     (1/2pi) prior on both
#     :param a_hat: the F-stat A parameters
#     :param f_plus: F_plus sensitivity
#     :param f_cross: F_cross sensitivity
#     :param d: distance
#     :param x: cos(inclination)
#     """
#     l = (
#         2
#         / pi
#         * quad(lambda psi: lhm_d_cosi_psi(a_hat, f_plus, f_cross, d, x, psi, g_hat, j_hat), 0, pi / 2)[0] #,epsrel=1.48,epsabs=1.48e-2)[0]
#     )
#     return l

######################### numerical integration functions ###############################


def like_parts_d_cosi_psi_phi(
    a_hat, f_plus, f_cross, d, x, psi, phi, numerical=False, phi_first=False
):
    """
    calculate the two dimensional likelihood, marginalized over phi
    log-likelihood can be written as:
    1/2(ahat^2 - 2*d0/d * f(x, psi) * cos(2phi - phi0) + (d0/d)^2 g(x,psi))
    return: ahat2, f, g
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param x: cos(inclination)
    :param psi: polarization
    """
    f_resp = array([0, f_plus, f_cross, f_plus, f_cross])
    ahat2 = sum(f_resp ** 2 * a_hat ** 2)

    d0 = a_hat[0]
    a = d0 / d

    if not phi_first:
        c2ph = cos(2 * phi)
        s2ph = sin(2 * phi)
        x2 = (1 + x ** 2) / 2
        c2psi_fac = (
            a_hat[1] * f_plus ** 2 * x2 * c2ph
            + a_hat[2] * f_cross ** 2 * x * s2ph
            - a_hat[3] * f_plus ** 2 * x2 * s2ph
            + a_hat[4] * f_cross ** 2 * x * c2ph
        )
        s2psi_fac = (
            -a_hat[1] * f_plus ** 2 * x * s2ph
            + a_hat[2] * f_cross ** 2 * x2 * c2ph
            - a_hat[3] * f_plus ** 2 * x * c2ph
            - a_hat[4] * f_cross ** 2 * x2 * s2ph
        )

        if numerical:
            g_c2psi_fac = (
                f_plus ** 2 * x2 ** 2 * c2ph ** 2
                + f_cross ** 2 * x ** 2 * s2ph ** 2
                + f_plus ** 2 * x2 ** 2 * s2ph ** 2
                + f_cross ** 2 * x ** 2 * c2ph ** 2
            )
            g_s2psi_fac = (
                f_plus ** 2 * x ** 2 * s2ph ** 2
                + f_cross ** 2 * x2 ** 2 * c2ph ** 2
                + f_plus ** 2 * x ** 2 * c2ph ** 2
                + f_cross ** 2 * x2 ** 2 * s2ph ** 2
            )
            g = cos(2 * psi) * g_c2psi_fac + sin(2 * psi) * g_s2psi_fac
            f = cos(2 * psi) * c2psi_fac + sin(2 * psi) * s2psi_fac
            like = exp(f * a - 0.5 * (ahat2 + g * a ** 2))
        else:
            raise "cannot do psi integral analytically"

    else:
        c2p = cos(2 * psi)
        s2p = sin(2 * psi)
        x2 = (1 + x ** 2) / 2
        c2phi_fac = (
            a_hat[1] * f_plus ** 2 * x2 * c2p
            + a_hat[2] * f_cross ** 2 * x2 * s2p
            - a_hat[3] * f_plus ** 2 * x * s2p
            + a_hat[4] * f_cross ** 2 * x * c2p
        )
        s2phi_fac = (
            -a_hat[1] * f_plus ** 2 * x * s2p
            + a_hat[2] * f_cross ** 2 * x * c2p
            - a_hat[3] * f_plus ** 2 * x2 * c2p
            - a_hat[4] * f_cross ** 2 * x2 * s2p
        )
        g = (
            f_plus ** 2 * x2 ** 2 * c2p ** 2
            + f_cross ** 2 * x2 ** 2 * s2p ** 2
            + f_plus ** 2 * x ** 2 * s2p ** 2
            + f_cross ** 2 * x ** 2 * c2p ** 2
        )

        if numerical:
            f = cos(2 * phi) * c2phi_fac + sin(2 * phi) * s2phi_fac
            like = exp(f * a - 0.5 * (ahat2 + g * a ** 2))
        else:
            f = sqrt(c2phi_fac ** 2 + s2phi_fac ** 2)
            like = special.i0e(f * a) * exp(f * a - 0.5 * (ahat2 + g * a ** 2))

    return like


def l_d_c_phi(a_hat, f_plus, f_cross, d, x, phi, numerical=False):
    """
    Return the likelihood marginalized over phi, using flat (1/2pi) prior
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    :param psi: polarization
    :param marg: do or don't do the marginalization
    :returns: the marginalization factor.  I don't think it includes the exp(rho^2/2) term.
    """
    if numerical:
        l = (
            2
            / pi
            * quad(
                lambda psi: like_parts_d_cosi_psi_phi(
                    a_hat, f_plus, f_cross, d, x, psi, phi, numerical=True
                ),
                0,
                pi / 2.0,
            )[0]
        )
    else:
        l = like_parts_d_cosi_psi_phi(a_hat, f_plus, f_cross, d, x, None, phi)
    return l


def l_d_c_psi(a_hat, f_plus, f_cross, d, x, psi, numerical=False):
    """
    Return the likelihood marginalized over phi, using flat (1/2pi) prior
    :param a_hat: the F-stat A parameters
    :param f_plus: F_plus sensitivity
    :param f_cross: F_cross sensitivity
    :param d: distance
    :param x: cos(inclination)
    :param psi: polarization
    :param marg: do or don't do the marginalization
    :returns: the marginalization factor.  I don't think it includes the exp(rho^2/2) term.
    """
    if numerical:
        l = (
            1
            / pi
            * quad(
                lambda phi: like_parts_d_cosi_psi_phi(
                    a_hat,
                    f_plus,
                    f_cross,
                    d,
                    x,
                    psi,
                    phi,
                    numerical=True,
                    phi_first=True,
                ),
                0,
                pi,
                epsrel=1.48,
                epsabs=1.48e-2,
            )[0]
        )
    else:
        l = like_parts_d_cosi_psi_phi(
            a_hat, f_plus, f_cross, d, x, psi, None, phi_first=True
        )
    return l


# def ldc(a_hat, f_plus, f_cross, d, x, numerical=False, phi_first=False):
#     """
#     Return the likelihood marginalized over phi, using flat (1/2pi) prior
#     :param a_hat: the F-stat A parameters
#     :param f_plus: F_plus sensitivity
#     :param f_cross: F_cross sensitivity
#     :param d: distance
#     :param x: cos(inclination)
#     :param psi: polarization
#     :param marg: do or don't do the marginalization
#     :returns: the marginalization factor.  I don't think it includes the exp(rho^2/2) term.
#     """
#     if not phi_first:
#         l = (
#             1
#             / pi
#             * quad(lambda phi: l_d_c_phi(a_hat, f_plus, f_cross, d, x,
#                                          phi, numerical), 0, pi)[0]
#         )
#     else:
#         l = (
#             2
#             / pi
#             * quad(lambda psi: l_d_c_psi(a_hat, f_plus, f_cross, d, x,
#                                          psi, numerical), 0, pi/2., epsrel=1.48,epsabs=1.48e-2)[0]
#         )
#     return l
#
#
# def l_d_c_phi_numerical(a_hat, f_plus, f_cross, d, x, phi, marg=True):
#     """
#     Return the likelihood marginalized over psi, using flat (1/2pi) prior
#     :param a_hat: the F-stat A parameters
#     :param f_plus: F_plus sensitivity
#     :param f_cross: F_cross sensitivity
#     :param d: distance
#     :param x: cos(inclination)
#     :param psi: polarization
#     :param marg: do or don't do the marginalization
#     :returns: the marginalization factor.  I don't think it includes the exp(rho^2/2) term.
#     """
#     l = (
#         1
#         / pi
#         * quad(lambda psi: like_parts_d_cosi_psi_phi_numerical(a_hat, f_plus, f_cross, d, x, psi, phi), 0, pi)[0]
#     )
#     return l
#
# def l_d_c_psi_first(a_hat, f_plus, f_cross, d, x):
#     """
#     Return the likelihood marginalized over phi, using flat (1/2pi) prior
#     :param a_hat: the F-stat A parameters
#     :param f_plus: F_plus sensitivity
#     :param f_cross: F_cross sensitivity
#     :param d: distance
#     :param x: cos(inclination)
#     :param psi: polarization
#     :param marg: do or don't do the marginalization
#     :returns: the marginalization factor.  I don't think it includes the exp(rho^2/2) term.
#     """
#     integrand = lambda phi: l_d_c_phi_numerical(a_hat, f_plus, f_cross, d, x, phi)
#     l = (1/(pi) * quad(integrand, 0, pi)[0])
#     return l
#
# def l_d_c_phi_first(a_hat, f_plus, f_cross, d, x):
#     """
#     Return the likelihood marginalized over phi, using flat (1/2pi) prior
#     :param a_hat: the F-stat A parameters
#     :param f_plus: F_plus sensitivity
#     :param f_cross: F_cross sensitivity
#     :param d: distance
#     :param x: cos(inclination)
#     :param psi: polarization
#     :param marg: do or don't do the marginalization
#     :returns: the marginalization factor.  I don't think it includes the exp(rho^2/2) term.
#     """
#     integrand = lambda psi: l_d_c_psi_numerical(a_hat, f_plus, f_cross, d, x, psi)
#     l = (1/(pi) * quad(integrand, 0, pi)[0])
#     return l

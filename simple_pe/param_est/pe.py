from pycbc import conversions
import numpy as np
from lal import MSUN_SI, C_SI, MRSUN_SI
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions


def opening(mchirp, eta, chi_eff, chi_p, f_low):
    mass_1 = conversions.mass1_from_mchirp_eta(mchirp, eta)
    mass_2 = conversions.mass2_from_mchirp_eta(mchirp, eta)
    beta, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = \
                SimInspiralTransformPrecessingNewInitialConditions(
                    0., 0., np.arctan2(chi_p,chi_eff), 0., 0.,
                    np.sqrt(chi_p**2 + chi_eff**2), 0, mass_1 * MSUN_SI, mass_2 * MSUN_SI,
                    float(f_low), 0.)
    return beta


def waveform_distances(tau, b, a_net, a_33, snrs, d_o, tau_o):
    """
    Calculate the infer distance as a function of angle
    b: the precession opening angle
    tau: tan(theta_jn/2)
    alpha: relative network sensitivity to 2nd polarization
    d_L: the distance
    """
    amp = snrs['22'] * d_o * (1 + tau_o ** 2) ** 2 / (1 + tau ** 2) ** 2
    amp_fac = {}
    amp_fac['22'] = 1.
    amp_fac['33'] = 2 * tau / (1 + tau ** 2) * 2 * a_33
    amp_fac['prec'] = 4 * b * tau
    amp_fac['left'] = 2 * tau ** 4 * 2 * a_net / (1 + a_net ** 2)

    dist = {}
    dt = {}
    modes = ['22', '33', 'prec', 'left']
    for mode in modes:
        m, v = ncx2.stats(2, snrs[mode] ** 2, moments='mv')
        snr = np.array([np.sqrt(max(m + i * np.sqrt(v), 1e-15)) for i in range(-2, 3)])
        mode_amp = amp * amp_fac[mode]
        a, s = np.meshgrid(mode_amp, snr)
        dist[mode] = a / s
        dt[mode] = dist[mode] * (1 + tau ** 2) ** 2
    return dist, dt


def a33_vs_theta(a_33, snrs):
    """
    Calculate the allowed thetas vs a_33
    tau: tan(theta_jn/2)
    """
    m, v = ncx2.stats(2, snrs['33'] ** 2, moments='mv')
    snr33 = np.array([np.sqrt(max(m + i * np.sqrt(v), 1e-15)) for i in range(-2, 3)])
    a, s = np.meshgrid(a_33, snr33)
    theta = np.arcsin(np.minimum(s / snrs['22'] / 2 / a, 1))

    return theta


def rho_33pl(samples, chi_p, theta):
    a33 = 5./4 * (0.3 - samples[:,1])
    beta =  0.3 - samples[:,1]
    tau = np.tan(theta/2)
    rho_33 = 40 * np.sin(theta) * a33
    rho_p = 150 * beta * chi_p * tau
    a_net = 0.2
    rho_l = 20 * 2 * tau**4 * 2 * a_net / (1 + a_net**2)
    return(rho_p, rho_33, rho_l)
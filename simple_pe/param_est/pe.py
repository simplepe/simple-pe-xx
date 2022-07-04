from pycbc import conversions
import numpy as np
from scipy import interpolate
from lal import MSUN_SI, C_SI, MRSUN_SI
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
from simple_pe.waveforms import waveform_modes
from simple_pe.detectors import noise_curves


def opening(mchirp, eta, chi_eff, chi_p, freq):
    mass_1 = conversions.mass1_from_mchirp_eta(mchirp, eta)
    mass_2 = conversions.mass2_from_mchirp_eta(mchirp, eta)
    beta, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = \
        SimInspiralTransformPrecessingNewInitialConditions(
            0., 0., np.arctan2(chi_p, chi_eff), 0., 0.,
            np.sqrt(chi_p ** 2 + chi_eff ** 2), 0, mass_1 * MSUN_SI, mass_2 * MSUN_SI,
            float(freq), 0.)
    return beta


def waveform_distances(tau, b, a_net, a_33, snrs, d_o, tau_o):
    """
    Calculate the inferred distance as a function of angle
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


def calculate_rho_33(mchirp, eta, chi_eff, theta, snr,
                     psd, f_low, interp_points=5, approximant="IMRPhenomXPHM"):
    """
    Calculate the 33 SNRs
    :param mchirp:  array of chirp mass values
    :param eta: array of eta values
    :param chi_eff: array of chi_effective values
    :param theta: array of theta_JN values
    :param snr: the network SNR
    :param psd: the PSD to use
    :param f_low: low frequency cutoff
    :param interp_points: number of points to interpolate alpha_33 and beta
    :param approximant: waveform approximant
    :return rho_33: array of 33-mode SNRs
    """
    mc_array = np.linspace(mchirp.min(), mchirp.max(), interp_points)
    eta_array = np.linspace(eta.min(), eta.max(), interp_points)
    chi_eff_array = np.linspace(chi_eff.min(), chi_eff.max(), interp_points)
    mcs, etas, chi_effs = np.meshgrid(mc_array, eta_array, chi_eff_array, indexing='ij')
    alpha_33_grid = np.zeros_like(mcs)

    for i, m in np.ndenumerate(mcs):
        a, o = waveform_modes.calculate_alpha_lm_and_overlaps(conversions.mass1_from_mchirp_eta(m, etas[i]),
                                                              conversions.mass2_from_mchirp_eta(m, etas[i]),
                                                              chi_effs[i], chi_effs[i],
                                                              psd, f_low,
                                                              approximant=approximant,
                                                              modes=['22', '33'], dominant_mode='22')
        alpha_33_grid[i] = a['33']

    alpha_33 = interpolate.interpn([mc_array, eta_array, chi_eff_array], alpha_33_grid,
                                   np.array([mchirp, eta, chi_eff]).T)

    rho_33 = snr * 2 * np.sin(theta) * alpha_33

    return rho_33


def calculate_rho_p(mchirp, eta, chi_eff, theta, chi_p, snr,
                    psd, f_low, approximant="IMRPhenomXPHM"):
    """
    Calculate the precessing SNR
    :param mchirp:  array of chirp mass values
    :param eta: array of eta values
    :param chi_eff: array of chi_effective values
    :param theta: array of theta_JN values
    :param chi_p: array of chi_p values
    :param snr: the network SNR
    :param psd: the PSD to use
    :param f_low: low frequency cutoff
    :param approximant: waveform approximant
    :return rho_p: array of precession SNRs
    """
    mass1 = conversions.mass1_from_mchirp_eta(mchirp.mean(), eta.mean())
    mass2 = conversions.mass2_from_mchirp_eta(mchirp.mean(), eta.mean())

    _, f_mean, _ = noise_curves.calc_reach_bandwidth(mass1, mass2, approximant, psd, f_low, thresh=8.)

    beta = np.zeros_like(mchirp)
    for i, mc in enumerate(mchirp):
        beta[i] = opening(mc, eta[i], chi_eff[i], chi_p[i], f_mean)

    rho_p = snr * 4 * np.tan(beta / 2) * np.tan(theta / 2)

    return rho_p


def calculate_rho_2nd_pol(theta, a_net, snr):
    """
    Calculate the SNR in the second polarization
    :param theta: array of theta_JN values
    :param a_net: network sensitivity to x polarization (in DP frame)
    :param snr: the network SNR
    :return rho_2pol: array of SNRs in 2nd polarization
    """

    rho_2pol = snr * 2 * np.tan(theta / 2) ** 4 * 2 * a_net / (1 + a_net ** 2)

    return rho_2pol

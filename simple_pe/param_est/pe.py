from pycbc import conversions
import numpy as np
from scipy import interpolate
from lal import MSUN_SI, C_SI, MRSUN_SI
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
from simple_pe.waveforms import waveform_modes
from simple_pe.detectors import noise_curves
from simple_pe.fstat import fstat_hm
from pesummary.gw.conversions import convert
from pesummary.utils.samples_dict import SamplesDict


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


def interpolate_alpha_lm(param_max, param_min, psd, f_low, grid_points, modes, approximant):
    """
    generate interpolating functions for the amplitudes of the lm multipoles
    :param param_max: A dictionary containing the maximum value of each parameter
    :param param_min: A dictionary containing the maximum value of each parameter
    :param psd: the PSD to use
    :param f_low: low frequency cutoff
    :param grid_points: number of points to interpolate alpha_33 and beta
    :param modes: waveform modes to calculate
    :param approximant: waveform approximant
    :return alpha: dictionary of alpha[lm] values interpolated across the grid
    :return pts: set of points used in each direction
    """
    dirs = param_max.keys()
    pts = [np.linspace(param_min[d][0], param_max[d][0], grid_points) for d in dirs]
    grid = np.array(np.meshgrid(*pts, indexing='ij'))

    alpha = {}
    for m in modes:
        alpha[m] = np.zeros_like(grid[0])

    for i, x in np.ndenumerate(grid[0]):
        sample = SamplesDict(dirs, [grid[a][i] for a in range(len(grid))])
        sample = convert(sample, disable_remnant=True)
        # use equal component spins if chi_eff is given
        if 'chi_eff' in sample.keys():
            sample['spin_1z'] = float(sample['chi_eff'])
            sample['spin_2z'] = float(sample['chi_eff'])

        a, _ = waveform_modes.calculate_alpha_lm_and_overlaps(sample['chirp_mass'],
                                                              sample['symmetric_mass_ratio'],
                                                              sample['spin_1z'],
                                                              sample['spin_2z'],
                                                              psd, f_low, approximant, modes,
                                                              dominant_mode='22')
        for m, al in alpha.items():
            al[i] = a[m]

    return alpha, pts


def calculate_rho_lm(samples, psd, f_low, net_snr, modes, interp_directions, interp_points=5,
                     approximant="IMRPhenomXPHM"):
    """
    Calculate the 33 SNRs
    :param samples: SamplesDict containing the samples
    :param psd: the PSD to use
    :param f_low: low frequency cutoff
    :param net_snr: the network SNR
    :param modes: modes for which to calculate SNR
    :param interp_directions: directions to interpolate
    :param interp_points: number of points to interpolate alpha_lm
    :param approximant: waveform approximant
    :return rho_lm: array of lm-mode SNRs
    """
    maxs = dict((k, samples.maximum[k]) for k in interp_directions)
    mins = dict((k, samples.minimum[k]) for k in interp_directions)

    alpha_grid, pts = interpolate_alpha_lm(maxs, mins, psd, f_low, interp_points, modes, approximant)

    alpha = {}
    rho_lm = {}
    for m in modes:
        alpha[m] = interpolate.interpn(pts, alpha_grid[m], np.array([samples[k] for k in interp_directions]).T)
        if "polarization" in samples.keys():
            rho_lm[m] = net_snr * alpha[m] * (np.cos(2 * samples['polarization']) * fstat_hm.amp[m+'+'](samples["theta_JN"]) +
                                              np.sin(2 * samples['polarization']) * fstat_hm.amp[m + 'x'](samples["theta_JN"]))
        else:
            rho_lm[m] = net_snr * alpha[m] * fstat_hm.amp[m+'+'](samples["theta_JN"])

    s = samples.

    return rho_lm


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

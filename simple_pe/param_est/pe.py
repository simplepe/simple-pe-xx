import numpy as np
from scipy import interpolate
from lal import MSUN_SI, C_SI, MRSUN_SI
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
from simple_pe.waveforms import waveform_modes
from simple_pe.detectors import noise_curves
from simple_pe.fstat import fstat_hm
from pesummary.gw.conversions import convert
from pesummary.utils.samples_dict import SamplesDict
from scipy.stats import ncx2


def calculate_opening(samples, freq):
    """
    generate the opening angle for each sample at the frequency given
    :param samples: a PESummary SamplesDict
    :param freq: the frequency to use
    Currently supports 'uniform', 'left_circ', 'right_circ'
    """
    opening = np.zeros(samples.number_of_samples)
    for i in range(samples.number_of_samples):
        s = convert(samples[i:i + 1], disable_remnant=True)

        beta, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = \
            SimInspiralTransformPrecessingNewInitialConditions(
                0., 0., np.arctan2(s["chi_p"][0], s["chi_eff"][0]), np.arccos(np.sign(s["chi_eff"][0])), 0.,
                np.sqrt(s["chi_p"][0] ** 2 + s["chi_eff"][0] ** 2), np.abs(s["chi_eff"][0]),
                s["mass_1"][0] * MSUN_SI, s["mass_2"][0] * MSUN_SI,
                float(freq), 0.)
        opening[i] = beta

    return opening


def interpolate_opening(param_max, param_min, fixed_pars, psd, f_low, grid_points, approximant):
    """
    generate interpolating functions for the amplitudes of the opening angle
    :param param_max: A dictionary containing the maximum value of each parameter
    :param param_min: A dictionary containing the maximum value of each parameter
    :param param_min: A dictionary containing the fixed parameters and their values
    :param f_low: the low frequency cutoff to use
    :param grid_points: number of points to interpolate alpha_33 and beta
    :return opening: array of opening angle values interpolated across the grid
    :return pts: set of points used in each direction
    """
    dirs = param_max.keys()
    pts = [np.linspace(param_min[d][0], param_max[d][0], grid_points) for d in dirs]
    grid = np.array(np.meshgrid(*pts, indexing='ij'))

    opening = np.zeros_like(grid[0])

    for i, x in np.ndenumerate(grid[0]):
        sample = SamplesDict(dirs, [grid[a][i] for a in range(len(grid))])
        sample.update(fixed_pars)
        sample = convert(sample, disable_remnant=True)
        _, f_mean, _ = noise_curves.calc_reach_bandwidth(sample["mass_1"], sample["mass_2"],
                                                         approximant, psd, f_low, thresh=8.)
        opening[i] = calculate_opening(sample, f_mean)

    return opening, pts


def generate_theta_jn(samples, theta_dist='uniform'):
    """
    generate theta JN points with the desired distribution and include in the existing samples dict
    :param samples: a PESummary SamplesDict
    :param theta_dist: the distribution to use for theta.
    Currently supports 'uniform', 'left_circ', 'right_circ'
    """
    npts = samples.number_of_samples
    if theta_dist == 'uniform':
        cos_theta = np.random.uniform(-1, 1, npts)
    elif theta_dist == 'left_circ':
        cos_theta = 2 * np.random.power(1 + 6, npts) - 1
    elif theta_dist == 'right_circ':
        cos_theta = 1 - 2 * np.random.power(1 + 6, npts)
    else:
        print("only implemented for 'uniform', 'left_circ', 'right_circ")
        return -1

    theta = np.arccos(cos_theta)

    new_samples = SamplesDict(samples.keys() + ['theta_jn', 'cos_theta_jn'],
                              np.append(samples.samples, np.array([theta, cos_theta]), 0))

    return new_samples


def generate_chi_p(samples, chi_p_dist='uniform'):
    """
    generate chi_p points with the desired distribution and include in the existing samples dict
    :param samples: a PESummary SamplesDict
    :param chi_p_dist: the distribution to use for chi_p.
    Currently supports 'uniform'
    """
    if chi_p_dist == 'uniform':
        chi_p_samples = np.random.uniform(0, np.sqrt(0.99 - samples.maximum['chi_eff'] ** 2), samples.number_of_samples)
    else:
        print("only implemented for 'uniform'")
        return -1

    new_samples = SamplesDict(samples.keys() + ['chi_p'],
                              np.append(samples.samples, np.array([chi_p_samples]), 0))

    return new_samples


def interpolate_alpha_lm(param_max, param_min, fixed_pars, psd, f_low, grid_points, modes, approximant):
    """
    generate interpolating functions for the amplitudes of the lm multipoles
    :param param_max: A dictionary containing the maximum value of each parameter
    :param param_min: A dictionary containing the maximum value of each parameter
    :param fixed_pars: A dictionary containing values of fixed parameters
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
        sample.update(fixed_pars)
        sample = convert(sample, disable_remnant=True)
        # use equal component spins if chi_eff is given
        if 'chi_eff' in sample.keys():
            sample['spin_1z'] = float(sample['chi_eff'])
            sample['spin_2z'] = float(sample['chi_eff'])

        a, _ = waveform_modes.calculate_alpha_lm_and_overlaps(sample['mass_1'],
                                                              sample['mass_2'],
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
    Calculate the higher mode SNRs
    :param samples: SamplesDict containing the samples
    :param psd: the PSD to use
    :param f_low: low frequency cutoff
    :param net_snr: the network SNR
    :param modes: modes for which to calculate SNR
    :param interp_directions: directions to interpolate
    :param interp_points: number of points to interpolate alpha_lm
    :param approximant: waveform approximant
    :return new_samples: SamplesDict including rho_lm for the specified modes
    """
    maxs = dict((k, samples.maximum[k]) for k in interp_directions)
    mins = dict((k, samples.minimum[k]) for k in interp_directions)
    fixed_pars = {k: v[0] for k, v in samples.mean.items() if k not in interp_directions}

    alpha_grid, pts = interpolate_alpha_lm(maxs, mins, fixed_pars, psd, f_low, interp_points, modes, approximant)

    alpha = {}
    rho_lm = {}
    for m in modes:
        alpha[m] = interpolate.interpn(pts, alpha_grid[m], np.array([samples[k] for k in interp_directions]).T)
        if "polarization" in samples.keys():
            rho_lm[m] = net_snr * alpha[m] * (
                    np.cos(2 * samples['polarization']) *
                    fstat_hm.amp[m + '+'](samples["theta_jn"]) / fstat_hm.amp['22+'](samples["theta_jn"]) +
                    np.sin(2 * samples['polarization']) *
                    fstat_hm.amp[m + 'x'](samples["theta_jn"]) / fstat_hm.amp['22x'](samples["theta_jn"]))
        else:
            rho_lm[m] = net_snr * alpha[m] * fstat_hm.amp[m + '+'](samples["theta_jn"]) / fstat_hm.amp['22+'](
                samples["theta_jn"])

    new_samples = SamplesDict(samples.keys() + ['rho_' + k for k in rho_lm.keys()],
                              np.append(samples.samples, np.array([rho_lm[k] for k in rho_lm.keys()]), 0))

    return new_samples


def calculate_rho_2nd_pol(samples, a_net, net_snr):
    """
    Calculate the SNR in the second polarization
    :param samples: SamplesDict of samples
    :param a_net: network sensitivity to x polarization (in DP frame)
    :param net_snr: the network SNR
    :return new_samples: SamplesDict with SNR for 2nd polarization
    """

    rho_2pol = net_snr * 2 * np.tan(samples['theta_jn'] / 2) ** 4 * 2 * a_net / (1 + a_net ** 2)

    new_samples = SamplesDict(samples.keys() + ['rho_2pol'], np.append(samples.samples, [rho_2pol], 0))

    return new_samples


def calculate_rho_p(samples, psd, f_low, net_snr, interp_directions, interp_points=5,
                    approximant="IMRPhenomXPHM"):
    """
    Calculate the precession SNR
    :param samples: SamplesDict containing the samples
    :param psd: the PSD to use
    :param f_low: low frequency cutoff
    :param net_snr: the network SNR
    :param interp_directions: directions to interpolate
    :param interp_points: number of points to interpolate alpha_lm
    :param approximant: waveform approximant
    :return new_samples: SamplesDict containing rho_p
    """
    maxs = dict((k, samples.maximum[k]) for k in interp_directions)
    mins = dict((k, samples.minimum[k]) for k in interp_directions)

    fixed_pars = {k: v[0] for k, v in samples.mean.items() if k not in interp_directions}

    beta_grid, pts = interpolate_opening(maxs, mins, fixed_pars, psd, f_low, interp_points, approximant)

    beta = interpolate.interpn(pts, beta_grid, np.array([samples[k] for k in interp_directions]).T)

    rho_p = net_snr * 4 * np.tan(beta / 2) * np.tan(samples["theta_jn"] / 2)
    new_samples = SamplesDict(samples.keys() + ['beta', 'rho_p'], np.append(samples.samples, [beta, rho_p], 0))

    return new_samples


def calculate_hm_prec_probs(samples, hm_snr=None, prec_snr=None, snr_2pol=None):
    """
    Calculate the precession SNR
    :param samples: SamplesDict containing the samples
    :param hm_snr: dictionary of measured SNRs in higher modes
    :param prec_snr: measured precession SNR
    :param snr_2pol: the SNR in the second polarization
    :return new_samples: with probabilities for each SNR and an overall weight
    """
    prob = {}
    if hm_snr is not None:
        for lm, snr in hm_snr.items():
            rv = ncx2(2, snr ** 2)
            prob['p_' + lm] = rv.pdf(samples['rho_' + lm] ** 2)

    if prec_snr is not None:
        rv = ncx2(2, prec_snr ** 2)
        prob['p_p'] = rv.pdf(samples['rho_p'] ** 2)

    if snr_2pol is not None:
        rv = ncx2(2, snr_2pol ** 2)
        prob['p_2pol'] = rv.pdf(samples['rho_2pol'] ** 2)

    # maximum prob = 1
    weights = np.ones(samples.number_of_samples)
    for p in prob.values():
        p /= p.max()
        weights *= p

    weights /= weights.max()

    new_samples = SamplesDict(samples.keys() + [k for k in prob.keys()] + ['weight'],
                              np.append(samples.samples, np.array([prob[k] for k in prob.keys()] + [weights]), 0))

    return new_samples


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

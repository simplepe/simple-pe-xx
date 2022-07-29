import numpy as np
from scipy import interpolate
from simple_pe.waveforms import waveform_modes
from simple_pe.detectors import noise_curves
from simple_pe.fstat import fstat_hm
from pesummary.utils.samples_dict import SamplesDict
from scipy.stats import ncx2


class SimplePESamples(SamplesDict):
    """
    Class for holding Simple PE Samples, and generating PE distributions
    """
    def __init__(self, *args, logger_warn="warn", autoscale=True):
        """
        Initialize as a SamplesDict
        """
        SamplesDict.__init__(self, *args, logger_warn, autoscale)

    def add_fixed(self, name, value):
        """
        generate an additional parameter called 'name' with constant 'value'

        :param name: the name of the parameter
        :param value: its value
        """
        npts = self.number_of_samples
        self[name] = np.ones(npts) * value

    def generate_theta_jn(self, theta_dist='uniform', overwrite=False):
        """
        generate theta JN points with the desired distribution and include in the SimplePESamples

        :param theta_dist: the distribution to use for theta.  Currently supports 'uniform', 'left_circ', 'right_circ'
        :param overwrite: if True, then overwrite existing values, otherwise don't
        """
        npts = self.number_of_samples
        if theta_dist == 'uniform':
            cos_theta = np.random.uniform(-1, 1, npts)
        elif theta_dist == 'left_circ':
            cos_theta = 2 * np.random.power(1 + 6, npts) - 1
        elif theta_dist == 'right_circ':
            cos_theta = 1 - 2 * np.random.power(1 + 6, npts)
        else:
            print("only implemented for 'uniform', 'left_circ', 'right_circ")
            return

        if 'theta_jn' in self.keys() and overwrite:
            print('Overwriting theta_jn values')
            self.pop('theta_jn')
        if 'cos_theta_jn' in self.keys() and overwrite:
            print('Overwriting cos_theta_jn values')
            self.pop('cos_theta_jn')

        if ('theta_jn' not in self.keys()) and ('cos_theta_jn' not in self.keys()):
            theta = np.arccos(cos_theta)
            self['theta_jn'] = theta
            self['cos_theta_jn'] = cos_theta
        else:
            print('Did not overwrite theta_jn and cos_theta_jn samples')

    def generate_chi_p(self, chi_p_dist='uniform', overwrite=False):
        """
        generate chi_p points with the desired distribution and include in the existing samples dict

        :param chi_p_dist: the distribution to use for chi_p. Currently supports 'uniform'
        :param overwrite: if True, then overwrite existing values, otherwise don't
        """
        if chi_p_dist == 'uniform':
            chi_p_samples = np.random.uniform(0, np.sqrt(0.99 - self.maximum['chi_eff'] ** 2), self.number_of_samples)
        else:
            print("only implemented for 'uniform'")
            return

        if 'chi_p' in self.keys() and overwrite:
            print('Overwriting chi_p values')
            self.pop('chi_p')

        if 'chi_p' not in self.keys():
            self['chi_p'] = chi_p_samples
        else:
            print('Did not overwrite chi_p samples')

    def generate_spin_z(self, overwrite=False):
        """
        Generate z-component spins from chi_eff

        :param overwrite: if True, then overwrite existing values, otherwise don't
        """
        if 'chi_eff' not in self.keys():
            print("Need to have 'chi_eff' in samples")
            return

        if 'spin_1z' in self.keys() and overwrite:
            print('Overwriting spin_1z values')
            self.pop('spin_1z')
        if 'spin_2z' in self.keys() and overwrite:
            print('Overwriting spin_2z values')
            self.pop('spin_2z')

        if ('spin_1z' not in self.keys()) and ('spin_2z' not in self.keys()):
            # put chi_eff on both BHs, no x,y components
            self['spin_1z'] = self['chi_eff']
            self['spin_2z'] = self['chi_eff']
        else:
            print('Did not overwrite spin_1z and spin_2z samples')

    def generate_prec_spin(self, overwrite=False):
        """
        Generate component spins from chi_eff and chi_p

        :param overwrite: if True, then overwrite existing values, otherwise don't
        """
        if ('chi_eff' not in self.keys()) or ('chi_p' not in self.keys()):
            print("Need to specify 'chi_eff' and 'chi_p'")
            return

        for k in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl']:
            if k in self.keys():
                if overwrite:
                    print('Overwriting %s values' % k)
                    self.pop(k)
                else:
                    print('%s already in samples, not overwriting' % k)
                    return

        self['a_1'] = np.sqrt(self["chi_p"] ** 2 + self["chi_eff"] ** 2)
        self['a_2'] = np.abs(self["chi_eff"])
        self['tilt_1'] = np.arctan2(self["chi_p"], self["chi_eff"])
        self['tilt_2'] = np.arccos(np.sign(self["chi_eff"]))
        self.add_fixed('phi_12', 0.)
        self.add_fixed('phi_jl', 0.)

    def calculate_rho_lm(self, psd, f_low, net_snr, modes, interp_directions, interp_points=5,
                         approximant="IMRPhenomXPHM"):
        """
        Calculate the higher mode SNRs

        :param psd: the PSD to use
        :param f_low: low frequency cutoff
        :param net_snr: the network SNR
        :param modes: modes for which to calculate SNR
        :param interp_directions: directions to interpolate
        :param interp_points: number of points to interpolate alpha_lm
        :param approximant: waveform approximant
        """
        maxs = dict((k, self.maximum[k]) for k in interp_directions)
        mins = dict((k, self.minimum[k]) for k in interp_directions)
        fixed_pars = {k: v[0] for k, v in self.mean.items() if k not in interp_directions}

        alpha_grid, pts = interpolate_alpha_lm(maxs, mins, fixed_pars, psd, f_low, interp_points, modes, approximant)

        for m in modes:
            alpha = interpolate.interpn(pts, alpha_grid[m], np.array([self[k] for k in interp_directions]).T)
            if "polarization" in self.keys():
                self['rho_' + m] = net_snr * alpha * (
                        np.cos(2 * self['polarization']) *
                        fstat_hm.amp[m + '+'](self["theta_jn"]) / fstat_hm.amp['22+'](self["theta_jn"]) +
                        np.sin(2 * self['polarization']) *
                        fstat_hm.amp[m + 'x'](self["theta_jn"]) / fstat_hm.amp['22x'](self["theta_jn"]))
            else:
                self['rho_' + m] = net_snr * alpha * fstat_hm.amp[m + '+'](self["theta_jn"]) / fstat_hm.amp['22+'](
                    self["theta_jn"])

    def calculate_rho_2nd_pol(self, a_net, net_snr):
        """
        Calculate the SNR in the second polarization

        :param a_net: network sensitivity to x polarization (in DP frame)
        :param net_snr: the network SNR
        """
        self['rho_2pol'] = net_snr * 2 * np.tan(self['theta_jn'] / 2) ** 4 * 2 * a_net / (1 + a_net ** 2)

    def calculate_rho_p(self, psd, f_low, net_snr, interp_directions, interp_points=5,
                        approximant="IMRPhenomXP"):
        """
        Calculate the precession SNR

        :param psd: the PSD to use
        :param f_low: low frequency cutoff
        :param net_snr: the network SNR
        :param interp_directions: directions to interpolate
        :param interp_points: number of points to interpolate alpha_lm
        :param approximant: waveform approximant
        """
        maxs = dict((k, self.maximum[k]) for k in interp_directions)
        mins = dict((k, self.minimum[k]) for k in interp_directions)

        fixed_pars = {k: v[0] for k, v in self.mean.items() if k not in interp_directions}

        beta_grid, pts = interpolate_opening(maxs, mins, fixed_pars, psd, f_low, interp_points, approximant)

        self['beta'] = interpolate.interpn(pts, beta_grid, np.array([self[k] for k in interp_directions]).T)
        self['rho_p'] = net_snr * 4 * np.tan(self['beta'] / 2) * np.tan(self["theta_jn"] / 2)

    def calculate_hm_prec_probs(self, hm_snr=None, prec_snr=None, snr_2pol=None):
        """
        Calculate the precession SNR

        :param hm_snr: dictionary of measured SNRs in higher modes
        :param prec_snr: measured precession SNR
        :param snr_2pol: the SNR in the second polarization
        """
        weights = np.ones(self.number_of_samples)

        if hm_snr is not None:
            for lm, snr in hm_snr.items():
                rv = ncx2(2, snr ** 2)
                p = rv.pdf(self['rho_' + lm] ** 2)
                self['p_' + lm] = p/p.max()
                weights *  self['p_' + lm]

        if prec_snr is not None:
            rv = ncx2(2, prec_snr ** 2)
            p = rv.pdf(self['rho_p'] ** 2)
            self['p_p'] = p / p.max()
            weights *= self['p_p']

        if snr_2pol is not None:
            rv = ncx2(2, snr_2pol ** 2)
            p = rv.pdf(self['rho_2pol'] ** 2)
            self['p_2pol'] = p/p.max()
            weights *= self['p_2pol']

        self['weight'] = weights


def interpolate_opening(param_max, param_min, fixed_pars, psd, f_low, grid_points, approximant):
    """
    generate interpolating functions for the amplitudes of the opening angle

    :param param_max: A dictionary containing the maximum value of each parameter
    :param param_min: A dictionary containing the maximum value of each parameter
    :param fixed_pars: the fixed parameters needed to generate the waveform
    :param psd: the psd to use in calculating mean frequency, used for opening angle]
    :param f_low: the low frequency cutoff to use
    :param grid_points: number of points to interpolate opening angle
    :param approximant: the waveform approximant to use
    :return opening: array of opening angle values interpolated across the grid
    :return pts: set of points used in each direction
    """
    dirs = param_max.keys()
    pts = [np.linspace(param_min[d][0], param_max[d][0], grid_points) for d in dirs]
    grid_dict = dict(zip(dirs, np.array(np.meshgrid(*pts, indexing='ij'))))

    grid_samples = SimplePESamples({k: i.flatten() for k, i in grid_dict.items()})
    for k, i in fixed_pars.items():
        grid_samples.add_fixed(k, i)
    grid_samples.add_fixed('f_ref', 0)
    grid_samples.add_fixed('phase', 0)
    grid_samples.generate_prec_spin()
    grid_samples.generate_all_posterior_samples(disable_remnant=True)

    for i in range(grid_samples.number_of_samples):
        sample = grid_samples[i:i+1]
        _, f_mean, _ = noise_curves.calc_reach_bandwidth(sample["mass_1"], sample["mass_2"],
                                                         approximant, psd, f_low, thresh=8.)
        sample['f_ref'] = f_mean

    grid_samples.generate_all_posterior_samples(disable_remnant=True)

    return grid_samples['beta'].reshape(list(grid_dict.values())[0].shape), pts


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
    grid_dict = dict(zip(dirs, np.array(np.meshgrid(*pts, indexing='ij'))))

    grid_samples = SimplePESamples({k: i.flatten() for k, i in grid_dict.items()})
    for k, i in fixed_pars.items():
        grid_samples.add_fixed(k, i)
    grid_samples.generate_spin_z()
    grid_samples.generate_all_posterior_samples(disable_remnant=True)

    alpha = {}
    for m in modes:
        alpha[m] = np.zeros(grid_samples.number_of_samples)

    for i in range(grid_samples.number_of_samples):
        sample = grid_samples[i:i+1]
        a, _ = waveform_modes.calculate_alpha_lm_and_overlaps(sample['mass_1'],
                                                              sample['mass_2'],
                                                              sample['spin_1z'],
                                                              sample['spin_2z'],
                                                              psd, f_low, approximant, modes,
                                                              dominant_mode='22')
        for m, al in alpha.items():
            al[i] = a[m]

    for k, i in alpha.items():
        alpha[k] = i.reshape(list(grid_dict.values())[0].shape)

    return alpha, pts


# def waveform_distances(tau, b, a_net, a_33, snrs, d_o, tau_o):
#     """
#     Calculate the inferred distance as a function of angle
#
#     :param b: the precession opening angle
#     :param tau: tan(theta_jn/2)
#     :param alpha: relative network sensitivity to 2nd polarization
#     :param d_L: the distance
#     """
#     amp = snrs['22'] * d_o * (1 + tau_o ** 2) ** 2 / (1 + tau ** 2) ** 2
#     amp_fac = {'22': 1.,
#                '33': 2 * tau / (1 + tau ** 2) * 2 * a_33,
#                'prec': 4 * b * tau,
#                'left': 2 * tau ** 4 * 2 * a_net / (1 + a_net ** 2)
#                }
#
#     dist = {}
#     dt = {}
#     modes = ['22', '33', 'prec', 'left']
#     for mode in modes:
#         m, v = ncx2.stats(2, snrs[mode] ** 2, moments='mv')
#         snr = np.array([np.sqrt(max(m + i * np.sqrt(v), 1e-15)) for i in range(-2, 3)])
#         mode_amp = amp * amp_fac[mode]
#         a, s = np.meshgrid(mode_amp, snr)
#         dist[mode] = a / s
#         dt[mode] = dist[mode] * (1 + tau ** 2) ** 2
#     return dist, dt


def calculate_interpolated_snrs(
    samples, psd, f_low, dominant_snr, modes, alpha_net, hm_interp_dirs,
    prec_interp_dirs, interp_points, approximant, **kwargs
):
    """Wrapper function to calculate the SNR in the (l,m) multipoles,
    the SNR in the second polarisation and the SNR in precession.

    Parameters
    ----------
    samples: simple_pe.param_est.pe.SimplePESamples
        table of posterior distributions
    psd: pycbc.types.frequencyseries
        frequency series containing the PSD
    f_low: float
        low frequency cut-off to use for SNR calculations
    dominant_snr: float
        SNR in the dominant 22 multipole
    modes: list
        list of higher order multipoles that you wish to calculate
        the SNR for
    alpha_net: float
        network sensitivity to x polarization (in DP frame) used to
        calculate the SNR in the second
    hm_interp_dirs: list
        directions to interpole the higher multipole SNR calculation
    prec_interp_dirs: list
        directions to interpole the precession SNR calculation
    interp_points: int
        number of points to interpolate the SNRs
    approximant: str
        approximant to use when calculating the SNRs
    """
    if not isinstance(samples, SimplePESamples):
        samples = SimplePESamples(samples)
    # generate required parameters if necessary
    if "theta_jn" not in samples.keys():
        samples.generate_theta_jn('left_circ')
    if "chi_p" not in samples.keys():
        samples.generate_chi_p('uniform')
    samples.calculate_rho_lm(
        psd, f_low, dominant_snr, modes, hm_interp_dirs, interp_points, approximant
    )
    samples.calculate_rho_2nd_pol(alpha_net, dominant_snr)
    samples.calculate_rho_p(
        psd, f_low, dominant_snr, prec_interp_dirs, interp_points, approximant
    )
    return samples


def reweight_based_on_observed_snrs(samples, **kwargs):
    """Resample a table of posterior distributions based on the observed
    SNR in the higher multipoles, precession and second polarisation.

    Parameters
    ----------
    samples: simple_pe.param_est.pe.SimplePESamples
        table containing posterior distributions
    **kwargs: dict, optional
        all kwargs passed to the samples.calculate_hm_prec_probs function
    """
    from pesummary.core.reweight import rejection_sampling
    if not isinstance(samples, SimplePESamples):
        samples = SimplePESamples(samples)
    samples.calculate_hm_prec_probs(**kwargs)
    return rejection_sampling(samples, samples['weight'])

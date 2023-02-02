import numpy as np
from scipy import interpolate
from simple_pe.waveforms import waveform_modes, waveform
from simple_pe.detectors import noise_curves
from simple_pe.fstat import fstat_hm
from pesummary.utils.array import Array
from pesummary.utils.samples_dict import SamplesDict
from scipy.stats import ncx2, norm
from pycbc.filter import sigma
import tqdm

spin_max = 0.98
ecc_max = 0.5
prec_min = 1e-3

param_mins = {'chirp_mass': 1.,
              'total_mass': 2.,
              'mass_1': 1.,
              'mass_2': 1.,
              'symmetric_mass_ratio': 0.04,
              'chi_eff': -1 * spin_max,
              'chi_align': -1 * spin_max,
              'chi_p': prec_min,
              'chi_p2': prec_min,
              'prec': prec_min,
              'chi': prec_min,
              'spin_1z': -1 * spin_max,
              'spin_2z': -1 * spin_max,
              'a_1': 0.,
              'a_2': 0.,
              'tilt_1': 0.,
              'tilt_2': 0.,
              'tilt': prec_min,
              'eccentricity': 0.,
              'ecc2': 0.,
              }

param_maxs = {'chirp_mass': 1e4,
              'total_mass': 1e4,
              'mass_1': 1e4,
              'mass_2': 1e4,
              'symmetric_mass_ratio': 0.25,
              'chi_eff': spin_max,
              'chi_align': spin_max,
              'chi_p': spin_max,
              'chi_p2': spin_max**2,
              'chi': spin_max,
              'spin_1z': spin_max,
              'spin_2z': spin_max,
              'a_1': spin_max,
              'a_2': spin_max,
              'tilt_1': np.pi,
              'tilt_2': np.pi,
              'tilt': np.pi - prec_min,
              'eccentricity': ecc_max,
              'ecc2': ecc_max**2,
              }


def _add_chi_align(data):
    """Add samples for chi_align. If spin_1z, spin_2z, mass_ratio are not
    in data, samples for chi_align are not added to data

    Parameters
    ----------
    data: dict
        dictionary of samples
    """
    try:
        data["chi_align"] = (
            data["spin_1z"] + data["mass_ratio"]**(4. / 3) * data["spin_2z"]
        ) / (1 + data["mass_ratio"]**(4. / 3))
    except KeyError:
        pass
    return data


def convert(*args, **kwargs):
    from pesummary.gw.conversions import convert as _convert
    data = _convert(*args, **kwargs)
    return _add_chi_align(data)


class SimplePESamples(SamplesDict):
    """
    Class for holding Simple PE Samples, and generating PE distributions
    """
    def __init__(self, *args, logger_warn="warn", autoscale=True):
        """
        Initialize as a SamplesDict
        """
        if isinstance(args[0], dict):
            _args = {}

            for key, item in args[0].items():
                if isinstance(item, (float, int, np.number)):
                    _args[key] = [item]
                else:
                    _args[key] = item
            args = (_args,)
        SamplesDict.__init__(self, *args, logger_warn, autoscale)

    def __setitem__(self, key, value):
        _value = value
        if isinstance(_value, (float, int, np.number)):
            _value = [_value]
        if not isinstance(_value, Array):
            _value = Array(_value)
        if not _value.ndim:
            _value = Array([_value])
        super(SamplesDict, self).__setitem__(key, _value)
        try:
            if key not in self.parameters:
                self.parameters.append(key)
                try:
                    cond = (
                        np.array(self.samples).ndim == 1 and isinstance(
                            self.samples[0], (float, int, np.number)
                        )
                    )
                except Exception:
                    cond = False
                if cond and isinstance(self.samples, np.ndarray):
                    self.samples = np.append(self.samples, _value)
                elif cond and isinstance(self.samples, list):
                    self.samples.append(_value)
                else:
                    self.samples = np.vstack([self.samples, _value])
                self._update_latex_labels()
        except (AttributeError, TypeError):
            pass

    def update(self, dictionary):
        for key, value in dictionary.items():
            self.__setitem__(key, value)

    def _update_latex_labels(self):
        super(SimplePESamples, self)._update_latex_labels()
        self._latex_labels.update({"chi_align": r"$\chi_{A}$", "distance": r"$d_{L}$"})

    def generate_all_posterior_samples(self, function=None, **kwargs):
        """Convert samples stored in the SamplesDict according to a conversion
        function

        Parameters
        ----------
        function: func, optional
            function to use when converting posterior samples. Must take a
            dictionary as input and return a dictionary of converted posterior
            samples. Default `simple_pe.param_est.convert`
        **kwargs: dict, optional
            All additional kwargs passed to function
        """
        # if "chi_p" not in self.keys() and "chi_p2" in self.keys():
        #     self["chi_p"] = np.sqrt(self["chi_p2"])
        if function is None:
            function = convert
        return super(SimplePESamples, self).generate_all_posterior_samples(
            function=function, **kwargs
        )

    def add_fixed(self, name, value):
        """
        generate an additional parameter called 'name' with constant 'value'

        :param name: the name of the parameter
        :param value: its value
        """
        try:
            npts = self.number_of_samples
            self[name] = np.ones(npts) * value
        except TypeError:
            self[name] = value 

    def generate_theta_jn(self, theta_dist='uniform', snr_left=0., snr_right=0., overwrite=False):
        """
        generate theta JN points with the desired distribution and include in the SimplePESamples

        :param theta_dist: the distribution to use for theta.  Currently, supports 'uniform', 'left_circ', 'right_circ',
        'left_right'
        :param snr_left: left snr
        :param snr_right: right snr
        :param overwrite: if True, then overwrite existing values, otherwise don't
        """
        if 'theta_jn' in self.keys() and overwrite:
            print('Overwriting theta_jn values')
            self.pop('theta_jn')

        if 'cos_theta_jn' in self.keys() and overwrite:
            print('Overwriting cos_theta_jn values')
            self.pop('cos_theta_jn')

        if ('theta_jn' in self.keys()) or ('cos_theta_jn' in self.keys()):
            print('Did not overwrite theta_jn and cos_theta_jn samples')
            return

        npts = self.number_of_samples
        if theta_dist == 'uniform':
            cos_theta = np.random.uniform(-1, 1, npts)
        else:
            if theta_dist == 'left_circ':
                n_left = npts
                n_right = 0
            elif theta_dist == 'right_circ':
                n_left = 0
                n_right = npts
            elif theta_dist == 'left_right':
                n_left = int(
                    npts * np.exp(0.5 * snr_left ** 2) /
                    (np.exp(0.5 * snr_left ** 2) + np.exp(0.5 * snr_right ** 2)))
                n_right = npts - n_left
            else:
                print("only implemented for 'uniform', 'left_circ', 'right_circ', 'left_right'")
                return

            cos_theta_r = 2 * np.random.power(1 + 6, n_right) - 1
            cos_theta_l = 1 - 2 * np.random.power(1 + 6, n_left)
            cos_theta = np.concatenate((cos_theta_l, cos_theta_r))

        theta = np.arccos(cos_theta)
        self['theta_jn'] = theta
        self['cos_theta_jn'] = cos_theta

    def generate_distance(self, fiducial_distance, fiducial_sigma,
                          psd, f_low, interp_directions, interp_points=5,
                          approximant="IMRPhenomXPHM", overwrite=False):
        """
        generate distance points using the existing theta_JN samples and fiducial distance.
        interpolate sensitivity over the parameter space
        :param fiducial_distance: distance for a fiducial set of parameters
        :param fiducial_sigma: the range for a fiducial set of parameters
        :param psd: the PSD to use
        :param f_low: low frequency cutoff
        :param interp_directions: directions to interpolate
        :param interp_points: number of points to interpolate alpha_lm
        :param approximant: waveform approximant
        :param overwrite: if True, then overwrite existing values, otherwise don't
        """
        if 'theta_jn' not in self.keys():
            print('Require theta_jn values to calculate distances')
            return

        if 'distance' in self.keys() and overwrite:
            print('Overwriting distance values')
            self.pop('distance')

        if 'distance' not in self.keys():
            tau = np.tan(np.minimum(self['theta_jn'], np.pi - self['theta_jn']) / 2)

            maxs = dict((k, self.maximum[k]) for k in interp_directions)
            mins = dict((k, self.minimum[k]) for k in interp_directions)
            fixed_pars = {k: v[0] for k, v in self.mean.items() if k not in interp_directions}
            fixed_pars['distance'] = 1.0

            sigma_grid, pts = interpolate_sigma(maxs, mins, fixed_pars, psd, f_low, interp_points,
                                                approximant)

            sigma_int = interpolate.interpn(pts, sigma_grid, np.array([self[k] for k in interp_directions]).T)

            self['distance'] = fiducial_distance * sigma_int/fiducial_sigma / (1 + tau**2) ** 2
        else:
            print('Did not overwrite distance samples')

    def jitter_distance(self, net_snr, response_sigma=0.):
        """
        jitter distance values based upon existing distances
        jitter due to SNR and variation of network response
        :param net_snr: the network SNR
        :param response_sigma: standard deviation of network response (over sky)
        """
        if 'distance' not in self.keys():
            print('Need existing distance values before jittering them')
            return

        std = np.sqrt(net_snr ** -2 + response_sigma ** 2)
        # generate distance scaling factors
        d_scale = norm.rvs(1, std, 10 * self.number_of_samples)
        d_scale = d_scale[d_scale > 0]
        # weight using uniform volume distribution
        d_weight = d_scale ** 3
        d_weight /= d_weight.max()
        keep = (d_weight > np.random.uniform(0, 1, len(d_weight)))
        try:
            d_keep = d_scale[keep][:self.number_of_samples]
            self['distance'] *= d_keep
        except:
            print('Failed to generate enough samples')
            print('Not performing distance jitter')
            return

    def generate_chi_p(self, chi_p_dist='uniform', overwrite=False):
        """
        generate chi_p points with the desired distribution and include in the existing samples dict

        :param chi_p_dist: the distribution to use for chi_p. Currently supports 'uniform'
        :param overwrite: if True, then overwrite existing values, otherwise don't
        """
        param = "chi_eff" if "chi_eff" in self.keys() else "chi_align"
        self.trim_unphysical()
        if chi_p_dist == 'uniform':
            chi_p_samples = np.random.uniform(0, np.sqrt(0.99 - self.maximum[param] ** 2), self.number_of_samples)
        elif chi_p_dist == "isotropic_on_sky":
            from pesummary.gw.conversions import chi_p as _chi_p, q_from_eta, m1_from_mchirp_q, m2_from_mchirp_q
            a_1 = np.random.uniform(0, np.sqrt(0.99 - self.maximum[param] ** 2), self.number_of_samples)
            a_2 = np.random.uniform(0, np.sqrt(0.99 - self.maximum[param] ** 2), self.number_of_samples)
            tilt_1 = np.arccos(np.random.uniform(0, 1, self.number_of_samples))
            tilt_2 = np.arccos(np.random.uniform(0, 1, self.number_of_samples))
            spin_1x = a_1 * np.cos(tilt_1)
            spin_1y = np.zeros_like(spin_1x)
            spin_1z = a_1 * np.sin(tilt_1)
            spin_2x = a_2 * np.cos(tilt_2)
            spin_2y = np.zeros_like(spin_2x)
            spin_2z = a_2 * np.sin(tilt_2)
            chirp_mass = np.random.uniform(
                self.minimum["chirp_mass"], self.maximum["chirp_mass"], self.number_of_samples
            )
            mass_ratio = np.random.uniform(
                q_from_eta(self.minimum["symmetric_mass_ratio"]), q_from_eta(self.maximum["symmetric_mass_ratio"]),
                self.number_of_samples
            )
            mass_1 = m1_from_mchirp_q(chirp_mass, mass_ratio)
            mass_2 = m2_from_mchirp_q(chirp_mass, mass_ratio)
            chi_p_samples = _chi_p(mass_1, mass_2, spin_1x, spin_1y, spin_2x, spin_2y)
        else:
            print("only implemented for 'uniform' and 'isotropic_on_sky'")
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
        if not any(_ in self.keys() for _ in ['chi_eff', 'chi_align']):
            print("Need to have 'chi_align' in samples")
            return

        if 'spin_1z' in self.keys() and overwrite:
            print('Overwriting spin_1z values')
            self.pop('spin_1z')
        if 'spin_2z' in self.keys() and overwrite:
            print('Overwriting spin_2z values')
            self.pop('spin_2z')

        param = "chi_eff" if "chi_eff" in self.keys() else "chi_align"
        if ('spin_1z' not in self.keys()) and ('spin_2z' not in self.keys()):
            # put chi_eff on both BHs, no x,y components
            self['spin_1z'] = self[param]
            self['spin_2z'] = self[param]
        else:
            print('Did not overwrite spin_1z and spin_2z samples')

    def generate_prec_spin(self, overwrite=False):
        """
        Generate component spins from chi_eff and chi_p

        :param overwrite: if True, then overwrite existing values, otherwise don't
        """
        if 'chi_p' not in self.keys() and "chi_p2" not in self.keys():
            print("Need to specify precessing spin component, please give either 'chi_p' or 'chi_p2")
            return

        param = "chi_eff" if "chi_eff" in self.keys() else "chi_align"
        if param not in self.keys():
            print("Need to specify aligned spin component, please give either 'chi_eff' or 'chi_align'")
            return

        if "chi_p2" in self.keys():
            if "chi_p" in self.keys():
                print('Both chi_p and chi_p2 in samples, using chi_p')
            else:
                if sum(self['chi_p2'] < 0):
                    print('negative values of chi_p2, smallest = %.2g' % min(self['chi_p2']) )
                    print('setting equal 0')
                    self['chi_p2'][self['chi_p2'] < 0] = 0
                self['chi_p'] = np.sqrt(self['chi_p2'])

        for k in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl']:
            if k in self.keys():
                if overwrite:
                    print('Overwriting %s values' % k)
                    self.pop(k)
                else:
                    print('%s already in samples, not overwriting' % k)
                    return

        self['a_1'] = np.sqrt(self["chi_p"] ** 2 + self[param] ** 2)
        # # limit a_1 < 1
        # self['a_1'][self['a_1'] > 1.] = 1
        self['a_2'] = np.abs(self[param])
        # # limit a_2 < 1
        # self['a_2'][self['a_2'] > 1.] = 1.
        self['tilt_1'] = np.arctan2(self["chi_p"], self[param])
        self['tilt_2'] = np.arccos(np.sign(self[param]))
        self.add_fixed('phi_12', 0.)
        self.add_fixed('phi_jl', 0.)

    def trim_unphysical(self, maxs=None, mins=None, set_to_bounds=False):
        """
        Trim unphysical points from SimplePESamples

        :param maxs: the maximum permitted values of the physical parameters
        :param mins: the minimum physical values of the physical parameters
        :param set_to_bounds: move points that lie outside physical space to the boundary of allowed space
        :return physical_samples: SamplesDict with points outside the param max and min given
        """
        if mins is None:
            mins = param_mins

        if maxs is None:
            maxs = param_maxs

        if not set_to_bounds:
            keep = np.ones(self.number_of_samples, bool)
            for d, v in self.items():
                if d in maxs:
                    keep *= (v <= maxs[d])
                if d in mins:
                    keep *= (v >= mins[d])
            self.__init__(self[keep])
        else:
            for d, v in self.items():
                if d in maxs:
                    self[d][v >= maxs[d]] = maxs[d]
                if d in mins:
                    self[d][v <= mins[d]] = mins[d]

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
        self['rho_not_right'] = net_snr * np.tan(self['theta_jn'] / 2) ** 4 * 2 * a_net / (1 + a_net ** 2)
        self['rho_not_left'] = net_snr * np.tan((np.pi - self['theta_jn']) / 2) ** 4 * 2 * a_net / (1 + a_net ** 2)
        # doesn't make sense to have this larger than net_snr:
        self['rho_not_right'][self['rho_not_right'] > net_snr] = net_snr
        self['rho_not_left'][self['rho_not_left'] > net_snr] = net_snr

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
        maxs = dict((k, [self[k].max()]) for k in interp_directions)
        mins = dict((k, [self[k].min()]) for k in interp_directions)

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
                weights *= self['p_' + lm]

        if prec_snr is not None:
            rv = ncx2(2, prec_snr ** 2)
            p = rv.pdf(self['rho_p'] ** 2)
            self['p_p'] = p / p.max()
            weights *= self['p_p']

        if snr_2pol is not None:
            self.add_fixed('p_2pol', 0)
            for pol, snr in snr_2pol.items():
                rv = ncx2(2, snr ** 2)
                self['p_' + pol] = rv.pdf(self['rho_' + pol] ** 2)
                self['p_2pol'] = np.maximum(self['p_2pol'], self['p_' + pol])

            self['p_2pol'] /= self['p_2pol'].max()
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
    # if "chi_p2" in grid_samples.keys() and "chi_p" in grid_samples.keys():
    #     grid_samples.pop("chi_p")
    grid_samples.generate_prec_spin()
    # need to use set_to_bounds=True so we do not modify the grid
    grid_samples.trim_unphysical(set_to_bounds=True)
    grid_samples.generate_all_posterior_samples(disable_remnant=True)

    for i in tqdm.tqdm(range(grid_samples.number_of_samples), desc="calculating opening angle on grid"):
        sample = grid_samples[i:i+1]
        param = "chi_eff" if "chi_eff" in sample.keys() else "chi_align"
        _, f_mean, _ = noise_curves.calc_reach_bandwidth(sample["mass_1"], sample["mass_2"],
                                                         sample[param],
                                                         approximant, psd, f_low, thresh=8.)
        sample['f_ref'] = f_mean

    grid_samples.generate_all_posterior_samples(disable_remnant=True)
    return grid_samples['beta'].reshape(list(grid_dict.values())[0].shape), pts


def interpolate_sigma(param_max, param_min, fixed_pars, psd, f_low, grid_points, approximant):
    """
    generate interpolating function for sigma

    :param param_max: A dictionary containing the maximum value of each parameter
    :param param_min: A dictionary containing the maximum value of each parameter
    :param fixed_pars: A dictionary containing values of fixed parameters
    :param psd: the PSD to use
    :param f_low: low frequency cutoff
    :param grid_points: number of points to interpolate alpha_33 and beta
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

    sig = np.zeros(grid_samples.number_of_samples)

    for i in tqdm.tqdm(range(grid_samples.number_of_samples), desc="calculating sigma on grid"):
        sample = grid_samples[i:i+1]
        h = waveform.make_waveform(sample, psd.delta_f, f_low, len(psd), approximant)
        sig[i] = sigma(h, psd, low_frequency_cutoff=f_low,
                       high_frequency_cutoff=psd.sample_frequencies[-1])

    return sig.reshape(list(grid_dict.values())[0].shape), pts


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

    for i in tqdm.tqdm(range(grid_samples.number_of_samples), desc="calculating alpha_lm on grid"):
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


def calculate_interpolated_snrs(
        samples, psd, f_low, dominant_snr, left_snr, right_snr, modes, alpha_net, response_sigma,
        fiducial_distance, fiducial_sigma, dist_interp_dirs,
        hm_interp_dirs, prec_interp_dirs, interp_points, approximant, **kwargs
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
    left_snr: float
        SNR consistent with left circular polarization
    right_snr: float
        SNR consistent with right circular polarization
    modes: list
        list of higher order multipoles that you wish to calculate
        the SNR for
    alpha_net: float
        network sensitivity to x polarization (in DP frame) used to
        calculate the SNR in the second
    response_sigma: float
        standard deviation of network response over sky region
    fiducial_distance: float
        distance at which a face on signal would give the observed dominant SNR
    fiducial_sigma: float
        distance at which a face on signal would give SNR=8 at (using params for fiducial_distance)
    dist_interp_dirs: list
        directions to interpolate the distance
    hm_interp_dirs: list
        directions to interpolate the higher multipole SNR calculation
    prec_interp_dirs: list
        directions to interpolate the precession SNR calculation
    interp_points: int
        number of points to interpolate the SNRs
    approximant: str
        approximant to use when calculating the SNRs
    """
    if not isinstance(samples, SimplePESamples):
        samples = SimplePESamples(samples)
    # generate required parameters if necessary
    if "theta_jn" not in samples.keys():
        samples.generate_theta_jn('left_right', snr_left=left_snr, snr_right=right_snr)
    if "distance" not in samples.keys():
        samples.generate_distance(fiducial_distance, fiducial_sigma, psd, f_low,
                                  dist_interp_dirs, interp_points, approximant)
        samples.jitter_distance(dominant_snr, response_sigma)
    if "chi_p" not in samples.keys() and "chi_p2" not in samples.keys():
        samples.generate_chi_p('isotropic_on_sky')
    samples.calculate_rho_lm(
        psd, f_low, dominant_snr, modes, hm_interp_dirs, interp_points, approximant
    )
    samples.calculate_rho_2nd_pol(alpha_net, dominant_snr)
    if ("chi_p" in prec_interp_dirs) and ("chi_p" not in samples.keys()):
        samples['chi_p'] = samples['chi_p2']**0.5
    samples.calculate_rho_p(
        psd, f_low, dominant_snr, prec_interp_dirs, interp_points, approximant
    )
    if ("chi_p2" in samples.keys()) and ("chi_p" not in samples.keys()):
        samples['chi_p'] = samples['chi_p2']**0.5
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

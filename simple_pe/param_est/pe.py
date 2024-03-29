import numpy as np
import copy
import math
from scipy import interpolate
from simple_pe.waveforms import parameter_bounds, waveform_modes
from simple_pe.detectors import noise_curves
from simple_pe.fstat import fstat_hm
from pesummary.utils.array import Array
from pesummary.utils.samples_dict import SamplesDict
from pesummary.gw.conversions.spins import opening_angle
from scipy.stats import ncx2, norm
from pycbc.filter import sigma
import lalsimulation as ls
import tqdm


def _add_chi_align(data, **kwargs):
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


def _component_spins_from_chi_align_chi_p(data, chip_to_spin1x=False, **kwargs):
    """Add samples for the component spins. If chi_align and mass ratio
    are not in data, z component spins are not added. If chi_p and component
    masses are not in data, x, y component spins are not added. Component spins
    are drawn from a uniform distribution, and conditioned on the
    provided chi_align, chi_p samples (except when `chip_to_spin1x=True`). This
    is an expensive operation and can likely be optimised.

    Parameters
    ----------
    data: dict
        dictionary of samples
    chip_to_spin1x: Bool, optional
        if True, set spin_1x=chi_p and all other in-plane spin components=0
    """
    from pesummary.utils.utils import draw_conditioned_prior_samples
    _data = SimplePESamples(data.copy())
    _data.generate_all_posterior_samples()
    if all(_ in _data.keys() for _ in ["chi_align", "mass_ratio"]):
        s1z = np.random.uniform(-1, 1, len(_data["chi_align"]))
        s2z = np.random.uniform(-1, 1, len(_data["chi_align"]))
        conditioned = _add_chi_align(
            {"spin_1z": s1z, "spin_2z": s2z, "mass_ratio": _data["mass_ratio"]}
        )
        conditioned = draw_conditioned_prior_samples(
            _data, conditioned, ["chi_align"], {"chi_align": -1}, {"chi_align": 1},
            nsamples=len(_data["chi_align"])
        )
        _data["spin_1z"] = conditioned["spin_1z"]
        _data["spin_2z"] = conditioned["spin_2z"]
        _data["_chi_align"] = _data["chi_align"]
        _data["chi_align"] = conditioned["chi_align"]
    if "chi_p" in _data.keys() and chip_to_spin1x:
        _total = len(_data["chi_p"])
        _data["spin_1x"] = _data["chi_p"]
        _data["spin_1y"] = np.zeros(_total)
        _data["spin_2x"] = np.zeros(_total)
        _data["spin_2y"] = np.zeros(_total)
    elif all(_ in _data.keys() for _ in ["chi_p", "mass_1", "mass_2"]):
        from pesummary.gw.conversions import chi_p
        s1x = np.random.uniform(-1, 1, int(1e5))
        s1y = np.random.uniform(-1, 1, int(1e5))
        s2x = np.random.uniform(-1, 1, int(1e5))
        s2y = np.random.uniform(-1, 1, int(1e5))
        m1 = np.random.choice(_data["mass_1"], replace=True, size=int(1e5))
        m2 = np.random.choice(_data["mass_2"], replace=True, size=int(1e5))
        conditioned = chi_p(m1, m2, s1x, s1y, s2x, s2y)
        conditioned = draw_conditioned_prior_samples(
            _data, {"chi_p": conditioned}, ["chi_p"], {"chi_p": 0.}, {"chi_p": 1},
            nsamples=len(_data["chi_p"])
        )
        _data["spin_1x"] = conditioned["spin_1x"]
        _data["spin_1y"] = conditioned["spin_1y"]
        _data["spin_2x"] = conditioned["spin_2x"]
        _data["spin_2y"] = conditioned["spin_2y"]
        _data["_chi_p"] = _data["chi_p"]
        _data["chi_p"] = conditioned["chi_p"]
    for num in range(1, 3, 1):
        if all(f"spin_{num}{comp}" in _data.keys() for comp in ["x", "y", "z"]):
            _data[f"a_{num}"] = np.sqrt(
                _data[f"spin_{num}x"]**2 + _data[f"spin_{num}y"]**2 +
                _data[f"spin_{num}z"]**2
            )
            _data[f"a_{num}"][_data[f"a_{num}"] > 1.] = 1.
    return _data


def convert(*args, **kwargs):
    from pesummary.gw.conversions import convert as _convert
    if isinstance(args[0], dict):
        _dict = args[0]
        for key, value in _dict.items():
            if not isinstance(value, (np.ndarray, list)):
                _dict[key] = np.array([value])
            elif isinstance(value[0], (np.ndarray, list)):
                _dict[key] = Array(np.array(value).flatten())
        args = (_dict,)
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
                elif isinstance(item[0], (np.ndarray, list)):
                    _args[key] = Array(np.array(item).flatten())
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

    @property
    def neff(self):
        if "weight" in self.keys():
            return (np.sum(self["weight"]))**2 / np.sum(self["weight"]**2)
        return self.number_of_samples

    def update(self, dictionary):
        for key, value in dictionary.items():
            self.__setitem__(key, value)

    def _update_latex_labels(self):
        super(SimplePESamples, self)._update_latex_labels()
        self._latex_labels.update({"chi_align": r"$\chi_{A}$", 
                                   "distance": r"$d_{L}$"})

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
        if "a_1" in self.keys():
            # limit a_1 < 1
            self['a_1'][self['a_1'] > 1.] = 1
        if "a_2" in self.keys():
            # limit a_2 < 1
            self['a_2'][self['a_2'] > 1.] = 1
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

    def generate_snrs(
        self, psd, f_low, approximant, template_parameters, snrs,
        localization_method
    ):
        from simple_pe.detectors import calc_reach_bandwidth, Network
        from simple_pe.localization.event import Event
        net = Network(threshold=10.)
        for ifo, p in psd.items():
            hor, f_mean, f_band = calc_reach_bandwidth(
                [
                    template_parameters['chirp_mass'],
                    template_parameters['symmetric_mass_ratio']
                ], template_parameters["chi_align"], approximant, p, f_low,
                mass_configuration="chirp"
            )
            net.add_ifo(ifo, hor, f_mean, f_band, bns_range=False,
                        loc_thresh=4.)

        if all(len(np.unique(self[_])) == 1 for _ in ["ra", "dec"]):
            _ra = self["ra"][:1]
            _dec = self["dec"][:1]
        elif localization_method == "fullsky":
            _ra = self["ra"]
            _dec = self["dec"]
        elif localization_method == "average":
            # downsample to 1000 sky points to average over
            inds = np.random.choice(np.arange(len(self["ra"])), size=int(1e3))
            _ra = self["ra"][inds]
            _dec = self["dec"][inds]
        total = len(_ra)
        f_sig = np.zeros(total)
        alpha_net = np.zeros_like(f_sig)
        snr_left = np.zeros_like(f_sig)
        snr_right = np.zeros_like(f_sig)
        for i, (_r, _d) in tqdm.tqdm(enumerate(zip(_ra, _dec)), total=total):
            if total != self.number_of_samples:
                mc = template_parameters["chirp_mass"]
            else:
                mc = self["chirp_mass"][i]
            ee = Event.from_snrs(
                net, snrs["ifo_snr"], snrs["ifo_time"], mc, _r, _d
            )
            ee.calculate_sensitivity()
            f_sig[i] = ee.sensitivity
            alpha_net[i] = ee.alpha_net()
            snr_left[i] = np.linalg.norm(ee.projected_snr('left'))
            snr_right[i] = np.linalg.norm(ee.projected_snr('right'))

        if total == 1:
            self["f_sig"] = np.ones_like(self["chirp_mass"]) * f_sig[0]
            self["alpha_net"] =  np.ones_like(self["chirp_mass"]) * alpha_net[0]
            self["left_snr"] = np.ones_like(self["chirp_mass"]) * snr_left[0]
            self["right_snr"] = np.ones_like(self["chirp_mass"]) * snr_right[0]
            self["not_left"] = np.ones_like(self["chirp_mass"]) * np.sqrt(
                snrs["network"]**2 - snr_left[0]**2
            )
            self["not_right"] = np.ones_like(self["chirp_mass"]) * np.sqrt(
                snrs["network"]**2 - snr_right[0]**2
            )
        elif localization_method == "fullsky":
            self["left_snr"] = snr_left
            self["right_snr"] = snr_right
            self["not_left"] = np.sqrt(snrs["network"]**2 - snr_left**2)
            self["not_right"] = np.sqrt(snrs["network"]**2 - snr_right**2)
            self["f_sig"] = f_sig
            self["alpha_net"] = alpha_net
        else:
            self["left_snr"] = np.ones_like(self["chirp_mass"]) * np.mean(snr_left)
            self["right_snr"] = np.ones_like(self["chirp_mass"]) * np.mean(snr_right)
            self["not_left"] = np.ones_like(self["chirp_mass"]) * np.sqrt(
                np.min(snrs['network']**2 - snr_left**2)
            )
            self["not_right"] = np.ones_like(self["chirp_mass"]) * np.sqrt(
                np.min(snrs['network']**2 - snr_right**2)
            )
            self["f_sig"] = np.ones_like(self["chirp_mass"]) * np.mean(f_sig)
            self["alpha_net"] = np.ones_like(self["chirp_mass"]) * np.mean(alpha_net)

    def generate_theta_jn(self, theta_dist='uniform', snr_left=0., 
                          snr_right=0., overwrite=False):
        """
        generate theta JN points with the desired distribution and 
        include in the SimplePESamples

        :param theta_dist: the distribution to use for theta.  
            Currently, supports 'uniform', 'left_circ', 
            'right_circ', 'left_right'
        :param snr_left: left snr
        :param snr_right: right snr
        :param overwrite: if True, then overwrite existing values, 
        otherwise don't
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
                print("only implemented for 'uniform', "
                      "'left_circ', 'right_circ', 'left_right'")
                return

            cos_theta_r = 2 * np.random.power(1 + 6, n_right) - 1
            cos_theta_l = 1 - 2 * np.random.power(1 + 6, n_left)
            cos_theta = np.concatenate((cos_theta_l, cos_theta_r))

        theta = np.arccos(cos_theta)
        self['theta_jn'] = theta
        self['cos_theta_jn'] = cos_theta

    def generate_distance(self, fiducial_distance, fiducial_sigma,
                          psd, f_low, interp_directions, interp_points=5,
                          approximant="IMRPhenomXPHM", overwrite=False,
                          sigma_22_grid=None):
        """
        generate distance points using the existing theta_JN samples and
        fiducial distance.  Interpolate sensitivity over the parameter space

        :param fiducial_distance: distance for a fiducial set of parameters
        :param fiducial_sigma: the range for a fiducial set of parameters
        :param psd: the PSD to use
        :param f_low: low frequency cutoff
        :param interp_directions: directions to interpolate
        :param interp_points: number of points to interpolate alpha_lm
        :param approximant: waveform approximant
        :param overwrite: if True, then overwrite existing values, 
        otherwise don't
        """
        if 'theta_jn' not in self.keys():
            print('Require theta_jn values to calculate distances')
            return

        if 'distance' in self.keys() and overwrite:
            print('Overwriting distance values')
            self.pop('distance')

        if 'distance' not in self.keys():
            tau = np.tan(np.minimum(self['theta_jn'], 
                                    np.pi - self['theta_jn']) / 2)

            maxs = dict((k, self.maximum[k]) for k in interp_directions)
            mins = dict((k, self.minimum[k]) for k in interp_directions)
            fixed_pars = {k: v[0] for k, v in self.mean.items() 
                          if k not in interp_directions}
            fixed_pars['distance'] = 1.0
            # ensure we wind up with unphysical spins
            if 'chi_p' in fixed_pars.keys():
                fixed_pars['chi_p'] = 0.
            if 'chi_p2' in fixed_pars.keys():
                fixed_pars['chi_p2'] = 0.

            if sigma_22_grid is None:
                sigma_grid, pts = interpolate_sigma(maxs, mins, fixed_pars, psd,
                                                    f_low, interp_points,
                                                    approximant)
            else:
                sigma_grid, pts = sigma_22_grid

            sigma_int = interpolate.interpn(pts, sigma_grid, 
                                            np.array([self[k] for k in 
                                                      interp_directions]).T)

            self['distance'] = fiducial_distance * sigma_int/fiducial_sigma / \
                               (1 + tau**2) ** 2
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
        generate chi_p points with the desired distribution and include in the 
        existing samples dict

        :param chi_p_dist: the distribution to use for chi_p. 
            Currently supports 'uniform' and 'isotropic_on_sky'
        :param overwrite: if True, then overwrite existing values, 
            otherwise don't
        """
        param = "chi_eff" if "chi_eff" in self.keys() else "chi_align"
        self.trim_unphysical()
        if chi_p_dist == 'uniform':
            chi_p_samples = np.random.uniform(0, np.sqrt(0.99 - self.maximum[param] ** 2), 
                                              self.number_of_samples)
        elif chi_p_dist == "isotropic_on_sky":
            from pesummary.gw.conversions import chi_p as _chi_p, q_from_eta, \
                m1_from_mchirp_q, m2_from_mchirp_q
            a_1 = np.random.uniform(0, np.sqrt(0.99 - self.maximum[param] ** 2),
                                    self.number_of_samples)
            a_2 = np.random.uniform(0, np.sqrt(0.99 - self.maximum[param] ** 2),
                                    self.number_of_samples)
            tilt_1 = np.arccos(np.random.uniform(0, 1, self.number_of_samples))
            tilt_2 = np.arccos(np.random.uniform(0, 1, self.number_of_samples))
            spin_1x = a_1 * np.cos(tilt_1)
            spin_1y = np.zeros_like(spin_1x)
            spin_1z = a_1 * np.sin(tilt_1)
            spin_2x = a_2 * np.cos(tilt_2)
            spin_2y = np.zeros_like(spin_2x)
            spin_2z = a_2 * np.sin(tilt_2)
            chirp_mass = np.random.uniform(
                self.minimum["chirp_mass"], self.maximum["chirp_mass"], 
                self.number_of_samples
            )
            mass_ratio = np.random.uniform(
                q_from_eta(self.minimum["symmetric_mass_ratio"]), 
                q_from_eta(self.maximum["symmetric_mass_ratio"]),
                self.number_of_samples
            )
            mass_1 = m1_from_mchirp_q(chirp_mass, mass_ratio)
            mass_2 = m2_from_mchirp_q(chirp_mass, mass_ratio)
            chi_p_samples = _chi_p(mass_1, mass_2, spin_1x, spin_1y, 
                                   spin_2x, spin_2y)
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
        Generate z-component spins from chi_eff or chi_align

        :param overwrite: if True, then overwrite existing values, 
            otherwise don't
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
        Generate component spins from chi_eff/chi_align/spin_z and chi_p

        :param overwrite: if True, then overwrite existing values, 
            otherwise don't
        """
        param = "chi_eff" if "chi_eff" in self.keys() else "chi_align"
        if 'chi_p' not in self.keys() and "chi_p2" not in self.keys():
            self["spin_1z"] = self[param]
            self["spin_2z"] = self[param]
            self["a_1"] = np.abs(self["spin_1z"])
            self["a_2"] = np.abs(self["spin_2z"])
            self["tilt_1"] = np.arccos(np.sign(self["spin_1z"]))
            self["tilt_2"] = np.arccos(np.sign(self["spin_2z"]))
            for param in [
                "phi_12", "phi_jl", "beta", "spin_1x", "spin_1y",
                "spin_2x", "spin_2y"
            ]:
                self[param] = np.zeros_like(self["spin_1z"])
            return

        if ('spin_1z' in self.keys()) and ('spin_2z' in self.keys()):
            s1z = self['spin_1z']
            s2z = self['spin_2z']
        elif ('spin_1z' in self.keys()) or ('spin_2z' in self.keys()):
            print("Need to specify both 'spin_1z' and 'spin_2z'"
                  "(not just one) or else chi_align/chi_eff")
            return
        elif param not in self.keys():
            print("Need to specify aligned spin component, " 
                  "please give either 'chi_eff', 'chi_align' or components")
            return
        else:
            s1z = self[param]
            s2z = self[param]

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

        self['a_1'] = np.sqrt(self["chi_p"] ** 2 + s1z ** 2)
        # limit a_1 < 1
        #self['a_1'][self['a_1'] > 1.] = 1
        self['tilt_1'] = np.arctan2(self["chi_p"], s1z)
        self['a_2'] = np.abs(s2z)
        # limit a_2 < 1
        #self['a_2'][self['a_2'] > 1.] = 1.
        self['tilt_2'] = np.arccos(np.sign(s2z))
        self.add_fixed('phi_12', 0.)
        self.add_fixed('phi_jl', 0.)

    def trim_unphysical(self, maxs=None, mins=None, set_to_bounds=False):
        """
        Trim unphysical points from SimplePESamples

        :param maxs: the maximum permitted values of the physical parameters
        :param mins: the minimum physical values of the physical parameters
        :param set_to_bounds: move points that lie outside physical space to 
            the boundary of allowed space
        :return physical_samples: SamplesDict with points outside the param max 
            and min given
        """
        if mins is None:
            mins = parameter_bounds.param_mins

        if maxs is None:
            maxs = parameter_bounds.param_maxs

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
                ind = self.parameters.index(d)
                self.samples[ind] = self[d]

    def calculate_rho_lm(self, psd, f_low, net_snr, modes, interp_directions, 
                         interp_points=5, approximant="IMRPhenomXPHM",
                         alpha_lm_grid=None):
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
        fixed_pars = {k: v[0] for k, v in self.mean.items() 
                      if k not in interp_directions}

        if alpha_lm_grid is None:
            alpha_grid, pts = interpolate_alpha_lm(maxs, mins, fixed_pars, psd,
                                                   f_low, interp_points,
                                                   modes, approximant)
        else:
            alpha_grid, pts = alpha_lm_grid

        for m in modes:
            alpha = interpolate.interpn(pts, alpha_grid[m], 
                                        np.array([self[k] for k in 
                                                  interp_directions]).T)
            if "polarization" in self.keys():
                self['rho_' + m] = net_snr * alpha * (
                        np.cos(2 * self['polarization']) *
                        fstat_hm.amp[m + '+'](self["theta_jn"]) / 
                        fstat_hm.amp['22+'](self["theta_jn"]) +
                        np.sin(2 * self['polarization']) *
                        fstat_hm.amp[m + 'x'](self["theta_jn"]) / 
                        fstat_hm.amp['22x'](self["theta_jn"]))
            else:
                self['rho_' + m] = net_snr * alpha * \
                                   fstat_hm.amp[m + '+'](self["theta_jn"]) / \
                                   fstat_hm.amp['22+'](self["theta_jn"])

    def calculate_rho_2nd_pol(self, a_net, net_snr):
        """
        Calculate the SNR in the second polarization

        :param a_net: network sensitivity to x polarization (in DP frame)
        :param net_snr: the network SNR
        """
        self['rho_not_right'] = net_snr * np.tan(self['theta_jn'] / 2) ** 4 * \
                                2 * a_net / (1 + a_net ** 2)
        self['rho_not_left'] = net_snr * \
                               np.tan((np.pi - self['theta_jn']) / 2) ** 4 * \
                               2 * a_net / (1 + a_net ** 2)
        # doesn't make sense to have this larger than net_snr:
        if np.shape(net_snr) == np.shape(self['theta_jn']):
            self['rho_not_right'][self['rho_not_right'] > net_snr] = \
                net_snr[self['rho_not_right'] > net_snr]
            self['rho_not_left'][self['rho_not_left'] > net_snr] = \
                net_snr[self['rho_not_left'] > net_snr]
        else:
            self['rho_not_right'][self['rho_not_right'] > net_snr] = net_snr
            self['rho_not_left'][self['rho_not_left'] > net_snr] = net_snr

    def calculate_rho_p(self, psd, f_low, net_snr, interp_directions, 
                        interp_points=5, approximant="IMRPhenomXP",
                        beta_22_grid=None):
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

        fixed_pars = {k: v[0] for k, v in self.mean.items() 
                      if k not in interp_directions}

        if beta_22_grid is None:
            beta_grid, pts = interpolate_opening(maxs, mins, fixed_pars, psd, 
                                                 f_low, interp_points, approximant)
        else:
            beta_grid, pts = beta_22_grid

        self['beta'] = interpolate.interpn(pts, beta_grid, 
                                           np.array([self[k] for k 
                                                     in interp_directions]).T)
        t_over_2 = np.minimum(self['theta_jn'], np.pi - self['theta_jn'])/2
        self['rho_p'] = net_snr * 4 * np.tan(self['beta'] / 2) * \
                        np.tan(t_over_2)

    def calculate_hm_prec_probs(self, hm_snr=None, prec_snr=None, 
                                snr_2pol=None):
        """
        Calculate the precession SNR

        :param hm_snr: dictionary of measured SNRs in higher modes
        :param prec_snr: measured precession SNR
        :param snr_2pol: the SNR in the second polarization
        """
        weights = np.ones(self.number_of_samples)

        if hm_snr is not None:
            hm_snr = np.nan_to_num(hm_snr, 0.)
            for lm, snr in hm_snr.items():
                rv = ncx2(2, snr ** 2)
                p = rv.pdf(self['rho_' + lm] ** 2)
                self['p_' + lm] = p/p.max()
                weights *= self['p_' + lm]

        if prec_snr is not None:
            prec_snr = np.nan_to_num(prec_snr, 0.)
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


def interpolate_opening(param_max, param_min, fixed_pars, psd, f_low, 
                        grid_points, approximant):
    """
    generate interpolating functions for the amplitudes of the opening angle

    :param param_max: A dictionary containing the maximum value of
        each parameter
    :param param_min: A dictionary containing the maximum value of
        each parameter
    :param fixed_pars: the fixed parameters needed to generate the waveform
    :param psd: the psd to use in calculating mean frequency,
        used for opening angle
    :param f_low: the low frequency cutoff to use
    :param grid_points: number of points to interpolate opening angle
    :param approximant: the waveform approximant to use
    :return opening: array of opening angle values interpolated across the grid
    :return pts: set of points used in each direction
    """
    dirs = param_max.keys()
    pts = [np.linspace(param_min[d][0], param_max[d][0], grid_points)
           for d in dirs]
    grid_dict = dict(zip(dirs, np.array(np.meshgrid(*pts, indexing='ij'))))
    grid_samples = SimplePESamples({k: i.flatten()
                                    for k, i in grid_dict.items()})
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

    beta = np.zeros(grid_samples.number_of_samples)
    for i in tqdm.tqdm(range(grid_samples.number_of_samples), 
                       desc="calculating opening angle on grid"):
        sample = grid_samples[i:i+1]
        param = "chi_eff" if "chi_eff" in sample.keys() else "chi_align"
        _, f_mean, _ = noise_curves.calc_reach_bandwidth(
            [sample["mass_1"], sample["mass_2"]], sample[param], approximant,
            psd, f_low, thresh=8., mass_configuration="component"
        )
        beta[i] = opening_angle(
            sample["mass_1"], sample["mass_2"], sample["phi_jl"], 
            sample["tilt_1"], sample["tilt_2"], sample["phi_12"], 
            sample["a_1"], sample["a_2"], 
            f_mean * np.ones_like(sample["mass_1"]),
            sample["phase"])

    return beta.reshape(list(grid_dict.values())[0].shape), pts


def interpolate_sigma(param_max, param_min, fixed_pars, psd, f_low, grid_points, 
                      approximant):
    """
    generate interpolating function for sigma

    :param param_max: A dictionary containing the maximum value of
        each parameter
    :param param_min: A dictionary containing the maximum value of
        each parameter
    :param fixed_pars: A dictionary containing values of fixed parameters
    :param psd: the PSD to use
    :param f_low: low frequency cutoff
    :param grid_points: number of points to interpolate alpha_33 and beta
    :param approximant: waveform approximant
    :return alpha: dictionary of alpha[lm] values interpolated across the grid
    :return pts: set of points used in each direction
    """
    from simple_pe.waveforms import waveform
    dirs = param_max.keys()
    pts = [np.linspace(param_min[d][0], param_max[d][0], grid_points)
           for d in dirs]
    grid_dict = dict(zip(dirs, np.array(np.meshgrid(*pts, indexing='ij'))))

    grid_samples = SimplePESamples({k: i.flatten() for k, i in grid_dict.items()})
    for k, i in fixed_pars.items():
        grid_samples.add_fixed(k, i)

    sig = np.zeros(grid_samples.number_of_samples)

    for i in tqdm.tqdm(range(grid_samples.number_of_samples),
                       desc="calculating sigma on grid"):
        sample = grid_samples[i:i+1]
        h = waveform.make_waveform(sample, psd.delta_f, f_low, len(psd),
                                   approximant)
        sig[i] = sigma(h, psd, low_frequency_cutoff=f_low,
                       high_frequency_cutoff=psd.sample_frequencies[-1])

    return sig.reshape(list(grid_dict.values())[0].shape), pts


def interpolate_alpha_lm(param_max, param_min, fixed_pars, psd, f_low,
                         grid_points, modes, approximant):
    """
    generate interpolating functions for the amplitudes of the lm multipoles

    :param param_max: A dictionary containing the maximum value of
        each parameter
    :param param_min: A dictionary containing the maximum value of
        each parameter
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
    pts = [np.linspace(param_min[d][0], param_max[d][0], grid_points)
           for d in dirs]
    grid_dict = dict(zip(dirs, np.array(np.meshgrid(*pts, indexing='ij'))))
    grid_samples = SimplePESamples({k: i.flatten()
                                    for k, i in grid_dict.items()})
    for k, i in fixed_pars.items():
        grid_samples.add_fixed(k, i)
    grid_samples.generate_spin_z()
    grid_samples.generate_all_posterior_samples(disable_remnant=True)

    alpha = {}
    for m in modes:
        alpha[m] = np.zeros(grid_samples.number_of_samples)

    for i in tqdm.tqdm(range(grid_samples.number_of_samples),
                       desc="calculating alpha_lm on grid"):
        sample = grid_samples[i:i+1]
        a, _ = waveform_modes.calculate_alpha_lm_and_overlaps(sample['mass_1'],
                                                              sample['mass_2'],
                                                              sample['spin_1z'],
                                                              sample['spin_2z'],
                                                              psd, f_low,
                                                              approximant, modes,
                                                              dominant_mode='22'
                                                              )
        for m, al in alpha.items():
            al[i] = a[m]

    for k, i in alpha.items():
        alpha[k] = i.reshape(list(grid_dict.values())[0].shape)

    return alpha, pts


def calculate_interpolated_snrs(
        samples, psd, f_low, dominant_snr, modes, response_sigma,
        fiducial_sigma, dist_interp_dirs,
        hm_interp_dirs, prec_interp_dirs, interp_points, approximant,
        localization_method, **kwargs
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
    response_sigma: float
        standard deviation of network response over sky region
    fiducial_distance: float
        distance at which a face on signal would give the observed dominant SNR
    fiducial_sigma: float
        distance at which a face on signal would give SNR=8 at
        (using params for fiducial_distance)
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
    localization_method: str
        method to use when localizing the event. Must either be 'average'
        or 'fullsky'
    """
    from simple_pe import io
    if not isinstance(samples, SimplePESamples):
        samples = SimplePESamples(samples)
    hm_psd = io.calculate_harmonic_mean_psd(psd)
    # generate required parameters if necessary
    template_parameters = kwargs.get("template_parameters", None)
    snrs = kwargs.get("snrs", None)
    if any(kwargs.get(f"{_}_snr", None) is None for _ in ["left", "right"]):
        samples.generate_snrs(
            psd=psd, f_low=f_low, approximant=approximant,
            template_parameters=template_parameters, snrs=snrs,
            localization_method=localization_method
        )
    
    if "theta_jn" not in samples.keys() and \
            kwargs.get("left_snr", None) is not None:
        samples.generate_theta_jn(
            'left_right', snr_left=kwargs.pop("left_snr"),
            snr_right=kwargs.pop("right_snr")
        )
    elif "theta_jn" not in samples.keys():
        samples.generate_theta_jn('uniform')
    samples["distance_face_on"] = samples["f_sig"] / snrs["network"]
    if "distance" not in samples.keys():
        samples.generate_distance(samples["distance_face_on"], fiducial_sigma,
                                  hm_psd, f_low, dist_interp_dirs,
                                  interp_points, approximant,
                                  sigma_22_grid=kwargs.get("sigma_22_grid", None))
        samples.jitter_distance(dominant_snr, response_sigma)
    if "chi_p" not in samples.keys() and "chi_p2" not in samples.keys():
        if ls.SimInspiralGetSpinSupportFromApproximant(getattr(ls, approximant)) > 2:
            samples.generate_chi_p('isotropic_on_sky')
        else:
            samples["chi_p"] = np.zeros_like(samples["theta_jn"])
    samples.calculate_rho_lm(
        hm_psd, f_low, dominant_snr, modes, hm_interp_dirs, interp_points,
        approximant, alpha_lm_grid=kwargs.get("alpha_lm_grid", None)
    )
    samples.calculate_rho_2nd_pol(samples["alpha_net"], dominant_snr)
    if ("chi_p" in prec_interp_dirs) and ("chi_p" not in samples.keys()):
        samples['chi_p'] = samples['chi_p2']**0.5
    if ls.SimInspiralGetSpinSupportFromApproximant(getattr(ls, approximant)) > 2:
        samples.calculate_rho_p(
            hm_psd, f_low, dominant_snr, prec_interp_dirs, interp_points,
            approximant, beta_22_grid=kwargs.get("beta_22_grid", None)
        )
    else:
        samples["rho_p"] = np.zeros_like(samples["theta_jn"])
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


def isotropic_spin_prior_weight(samples, dx_directions):
    """
    Re-weight points to prior proportional to chi_p (1 - chi_p), rather
    than uniform in chi_p or chi_p2.
    This prior approximately matches the one that arises from using
    uniform priors on spin magnitudes and orientations.

    Parameters
    ----------
    samples: SimplePESamples
        set of samples to reweight
    dx_directions: list
        directions used in generating samples

    Returns
    -------
    reweighted_samples: SimplePESamples
    """
    from pesummary.core.reweight import rejection_sampling
    if 'chi_p' in dx_directions:
        weights = samples['chi_p'] * (1 - samples['chi_p'])
    elif 'chi_p2' in dx_directions:
        weights = 1 - samples['chi_p2']**0.5
    else:
        weights = np.ones_like(samples['chirp_mass'])
    return rejection_sampling(samples, weights)


def component_mass_prior_weight(samples, dx_directions):
    """
    Re-weight points to uniform in mass ratio rather than
    symmetric mass ratio.  Since the transformation is singular at equal mass
    we truncate at close to equal mass (eta = 0.2499)

    Parameters
    ----------
    samples: SimplePESamples
        set of samples to re-weight
    dx_directions: list
        directions used in generating samples

    Returns
    -------
    reweighted_samples: SimplePESamples
    """
    from pesummary.core.reweight import rejection_sampling
    if 'symmetric_mass_ratio' in dx_directions:
        mass_weights = samples['chirp_mass'] * np.minimum(50, 1 / np.sqrt(
            1 - 4 * samples['symmetric_mass_ratio']))
    else:
        mass_weights = np.ones_like(samples['chirp_mass'])

    return rejection_sampling(samples, mass_weights)
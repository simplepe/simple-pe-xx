import numpy as np
import lal
from lalsimulation import (
    SimInspiralFD, SimInspiralCreateModeArray, SimInspiralModeArrayActivateMode,
    SimInspiralWaveformParamsInsertModeArray, GetApproximantFromString
)
from pycbc.types import FrequencySeries
from simple_pe.waveforms import parameter_bounds, waveform_modes
from simple_pe.param_est.pe import SimplePESamples
from pesummary.gw import conversions


def make_waveform(params, df, f_low, flen, approximant="IMRPhenomD", return_hc=False, modes=None,
                  harm2=False):
    """
    This function makes a waveform for the given parameters and
    returns h_plus generated at value x.

    :param params: SimplePESamples with parameter values for waveform generation
    :param df: frequency spacing of points
    :param f_low: low frequency cutoff
    :param flen: length of the frequency domain array to generate
    :param approximant: the approximant generator to use
    :param return_hc: flag to choose to return cross polarization (only non-precessing)
    :param modes: the modes to generate (only for non-precessing)
    :param harm2: generate the 2-harmonics
    :return h_plus: waveform at parameter space point x
    """
    for k, i in params.items():
        if not hasattr(i, '__len__'):
            params[k] = [i]

    x = SimplePESamples(params)
    if 'phase' not in x.keys():
        x['phase'] = np.zeros_like(list(x.values())[0])
    if 'f_ref' not in x.keys():
        x['f_ref'] = f_low * np.ones_like(list(x.values())[0])

    if modes is None:
        modes = waveform_modes.mode_array('22', approximant)

    if 'chi' in x.keys() and 'tilt' in x.keys():
        x['chi_p'] = x['chi'] * np.sin(x['tilt'])
        x['chi_align'] = x['chi'] * np.cos(x['tilt'])
        x.pop('tilt')
        x.pop('chi')

    prec = "chi_p" if "chi_p" in x.keys() else "chi_p2"

    if (prec in x.keys()) and x[prec]:
        # generate the leading harmonic of the precessing waveform
        x.generate_prec_spin()

    if 'tilt_1' in x.keys() and x['tilt_1']:
        x.generate_all_posterior_samples(f_low=f_low, f_ref=x["f_ref"][0], delta_f=df, disable_remnant=True)
        if harm2:
            harmonics = [0, 1]
        else:
            harmonics = [0]
        # only works for FD approximants
        try:
            h_plus = conversions.snr._calculate_precessing_harmonics(
                x["mass_1"][0], x["mass_2"][0], x["a_1"][0], x["a_2"][0],
                x["tilt_1"][0], x["tilt_2"][0], x["phi_12"][0],
                x["beta"][0], x["distance"][0], harmonics=harmonics,
                approx=approximant, mode_array=modes, df=df, f_low=f_low,
                f_ref=x["f_ref"][0]
            )
        except Exception:
            h_plus = conversions.snr._calculate_precessing_harmonics(
                x["mass_1"][0], x["mass_2"][0], x["a_1"][0], x["a_2"][0],
                x["tilt_1"][0], x["tilt_2"][0], x["phi_12"][0],
                x["beta"][0], x["distance"][0], harmonics=harmonics,
                approx="IMRPhenomPv2", mode_array=modes, df=df, f_low=f_low,
                f_ref=x["f_ref"][0]
            )
        if return_hc:
            print('return_hc not available for precessing system')
        for k, h in h_plus.items():
            h.resize(flen)
        if not harm2:
            return h_plus[0]
        return h_plus

    else:
        if harm2:
            raise ValueError(
                "Currently unable to calculate 2 harmonic decomposition when "
                "lalsimulation.SimInspiralFD is called"
            )
        if ('spin_1z' not in x.keys()) or ('spin_2z' not in x.keys()):
            x.generate_spin_z()
        if "inc" not in x.keys():
            x["inc"] = 0.

        x.generate_all_posterior_samples(f_low=f_low, f_ref=x["f_ref"][0], delta_f=df, disable_remnant=True)
        waveform_dictionary = lal.CreateDict()
        mode_array_lal = SimInspiralCreateModeArray()
        for mode in modes:
            SimInspiralModeArrayActivateMode(mode_array_lal, mode[0], mode[1])
        SimInspiralWaveformParamsInsertModeArray(waveform_dictionary, mode_array_lal)
        if isinstance(x["mass_1"], list):
            m1 = x["mass_1"][0]
        else:
            m1 = x["mass_1"]
        if isinstance(x["mass_2"], list):
            m2 = x["mass_2"][0]
        else:
            m2 = x["mass_2"]
        if "eccentricity" in x.keys():
            if isinstance(x["eccentricity"], list):
                ecc = x["eccentricity"][0]
            else:
                ecc = x["eccentricity"]
        elif "ecc2" in x.keys():
            if isinstance(x["ecc2"], list):
                ecc = x["ecc2"][0]**0.5
            else:
                ecc = x["ecc2"]**0.5  
        else:
            ecc = 0.
            
        args = [
            m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0., 0.,
            x["spin_1z"], 0., 0., x["spin_2z"], x['distance'] * 1e6 * lal.PC_SI,
            x["inc"], 0., 0., ecc, 0., df, f_low, 2048., x['f_ref']
        ]
        args = [float(arg) for arg in args]
        hp, hc = SimInspiralFD(
            *args, waveform_dictionary, GetApproximantFromString(approximant)
        )
        dt = 1 / hp.deltaF + (hp.epoch.gpsSeconds + hp.epoch.gpsNanoSeconds * 1e-9)
        time_shift = np.exp(
            -1j * 2 * np.pi * dt * np.array(range(len(hp.data.data[:]))) * hp.deltaF
        )
        hp.data.data[:] *= time_shift
        hc.data.data[:] *= time_shift
        h_plus = FrequencySeries(hp.data.data[:], delta_f=hp.deltaF, epoch=hp.epoch)
        h_cross = FrequencySeries(hc.data.data[:], delta_f=hc.deltaF, epoch=hc.epoch)

    h_plus.resize(flen)
    h_cross.resize(flen)
    if return_hc:
        return h_plus, h_cross
    return h_plus


def offset_params(x, dx, scaling):
    """
    Update the parameters x by moving to a value (x + scaling * dx)

    :param x: dictionary with parameter values for initial point
    :param dx: dictionary with parameter variations (can be a subset of the parameters in x)
    :param scaling: the scaling to apply to dx
    :return x_prime: parameter space point x + scaling * dx
    """
    x_prime = copy.deepcopy(x)

    for k, dx_val in dx.items():
        if k not in x_prime:
            print("Value for %s not given at initial point" % k)
            return -1

        x_prime[k] += float(scaling * dx_val)

    return x_prime


def make_offset_waveform(x, dx, scaling, df, f_low, flen, approximant="IMRPhenomD", harm2=False):
    """
    This function makes a waveform for the given parameters and
    returns h_plus generated at value (x + scaling * dx).

    :param x: dictionary with parameter values for initial point
    :param dx: dictionary with parameter variations (can be a subset of the parameters in x)
    :param scaling: the scaling to apply to dx
    :param df: frequency spacing of points
    :param f_low: low frequency cutoff
    :param flen: length of the frequency domain array to generate
    :param approximant: the approximant generator to use
    :param harm2: generate the 2-harmonics
    :return h_plus: waveform at parameter space point x + scaling * dx
    """
    h_plus = make_waveform(offset_params(x, dx, scaling), df, f_low, flen, approximant, harm2=harm2)

    return h_plus


def check_physical(x, dx, scaling, maxs=None, mins=None, verbose=False):
    """
    A function to check whether the point described by the positions x + dx is
    physically permitted.  If not, rescale and return the scaling factor

    :param x: dictionary with parameter values for initial point
    :param dx: dictionary with parameter variations
    :param scaling: the scaling to apply to dx
    :param maxs: a dictionary with the maximum permitted values of the physical parameters
    :param mins: a dictionary with the minimum physical values of the physical parameters
    :param verbose: print logging messages
    :return alpha: the scaling factor required to make x + scaling * dx physically permissible
    """
    if mins is None:
        mins = parameter_bounds.param_mins

    if maxs is None:
        maxs = parameter_bounds.param_maxs

    x0 = offset_params(x, dx, 0.)
    if verbose:
        print('initial point')
        print(x0)
    x_prime = offset_params(x, dx, scaling)
    if verbose:
        print('proposed point')
        print(x_prime)

    alpha = 1.

    if ('chi_p' in x_prime.keys()) or ('chi_p2' in x_prime.keys()):
        if ('chi_p2' in x_prime.keys()) and (x_prime['chi_p2'] < mins['chi_p2']):
            alpha = min(alpha, (x0['chi_p2'] - mins['chi_p2']) / (x0['chi_p2'] - x_prime['chi_p2']))
            x_prime['chi_p2'][0] = mins['chi_p2']
            if verbose:
                print("scaling to %.2f in direction %s" % (alpha, 'chi_p2'))
        x_prime.generate_spin_z()
        x_prime.generate_prec_spin()
        x0.generate_spin_z()
        x0.generate_prec_spin()

    for k, dx_val in x0.items():
        if k in mins.keys() and x_prime[k] < mins[k]:
            alpha = min(alpha, (x0[k] - mins[k]) / (x0[k] - x_prime[k]))
            if verbose:
                print("scaling to %.2f in direction %s" % (alpha, k))
        if k in maxs.keys() and x_prime[k] > maxs[k]:
            alpha = min(alpha, (maxs[k] - x0[k]) / (x_prime[k] - x0[k]))
            if verbose:
                print("scaling to %.2f in direction %s" % (alpha, k))

    # if varying 'chi_p2' need to double-check we don't go over limits
    if 'chi_p2' in dx.keys() and (scaling * dx['chi_p2']):
        chia = "chi_eff" if "chi_eff" in x0.keys() else "chi_align"
        # need find alpha s.t. (chi + alpha dchi)^2 + chi_p2 + alpha dchi_p2 = max_spin^2
        c = x0[chia] ** 2 + x0['chi_p2'] - maxs['a_1']**2
        dcp2 = scaling * dx['chi_p2']
        if chia in dx.keys() and dx[chia]:
            dchi = scaling * dx[chia]
            a = dchi ** 2
            b = 2 * x0[chia] * dchi + dcp2
            alpha_prec = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        else:
            # not changing aligned spin, so easier
            # chi^2 + chi_p2 + alpha dchi_p2 = max_spin^2
            # if dchi_p2 < 0 then bound is positivity of chi_p2, else alpha = -c/dcp2
            if dcp2 < 0:
                # chi_p2 + alpha dchi_p2 = mins['chi_p2']
                alpha_prec = (mins['chi_p2'] - x0['chi_p2']) /dcp2
            else:
                alpha_prec = -c / dcp2
        if verbose:
            print("scaling to %.2f for precession" % alpha_prec)

        alpha = min(alpha, alpha_prec)

        if verbose:
            x_prime = offset_params(x, dx, alpha * scaling)
            print('new point')
            print(x_prime)

    return alpha

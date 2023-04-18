import numpy as np
import lal
from lalsimulation import (
    SimInspiralFD, SimInspiralCreateModeArray, SimInspiralModeArrayActivateMode,
    SimInspiralWaveformParamsInsertModeArray, GetApproximantFromString
)
from pycbc.types import FrequencySeries
from simple_pe.waveforms import waveform_modes
from simple_pe.param_est.pe import SimplePESamples
from pesummary.gw import conversions


def make_waveform(params, df, f_low, flen, approximant="IMRPhenomD", return_hc=False, modes=None, harm2=False):
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

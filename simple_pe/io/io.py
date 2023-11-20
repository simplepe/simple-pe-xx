import json
import copy
import numpy as np
from simple_pe.param_est import pe
import lalsimulation as ls
from gwpy.timeseries import TimeSeries
import pycbc.psd.read
from pycbc.psd.analytical import aLIGOMidHighSensitivityP1200087


def load_trigger_parameters_from_file(path, approximant):
    """Load the trigger parameters from file

    Parameters
    ----------
    path: str
        path to file containing trigger parameters
    approximant: str
        approximant you wish to use

    Returns
    -------
    data: SimplePESamples
        a SimplePESamples object containing trigger parameters
    """
    with open(path, "r") as f:
        data = json.load(f)
    data = pe.SimplePESamples(data)
    required_params = [
        "mass_1", "mass_2", "spin_1z", "spin_2z", "time",
    ]
    if "distance" in data.keys():
        data["distance"] /= data["distance"]
    else:
        data.add_fixed("distance", 1.0)
    data.generate_all_posterior_samples()
    if not all(param in data.keys() for param in required_params):
        raise ValueError(
            f"{path} does not include all required parameters: "
            f"{','.join(required_params)}"
        )
    # always assume the precessing case unless chi_p = 0
    _precessing = 1.
    if "chi_p" in data.keys() and data["chi_p"] == 0:
        _precessing = 0.
    elif "chi_p2" in data.keys() and data["chi_p2"] == 0:
        _precessing = 0.
    # check to see if approximant allows for precession
    if ls.SimInspiralGetSpinSupportFromApproximant(getattr(ls, approximant)) <= 2:
        _precessing = 0.
        if "chi_p" in data.keys():
            data["chi_p"] = 0.
        if "chi_p2" in data.keys():
            data["chi_p2"] = 0.
    data["_precessing"] = _precessing
    return data


def load_strain_data_from_file(
    trigger_parameters, strain_data, channels, f_low, f_high, fudge_length=1.1,
    fudge_min=0.02, minimum_data_length=16
):
    """

    Parameters
    ----------
    trigger_parameters: dict
        dictionary containing trigger parameters
    strain_data: str/dict
    f_low: float
        low frequency cut off to use for the analysis
    f_high: float
        high frequency cut off to use for the analysis
    fudge_length: float, optional
        factor to multiply the rough time estimate of the waveform by to ensure
        that it is a conservative value. Default 1.1
    fudge_min: float, optional
        the minimum that the rough time estimate of the waveform can be.
        Default 0.02
    minimum_data_length: int, optional
        the minimum that the data length can be in seconds. Default 16
    """
    data_len = _estimate_data_length_from_template_parameters(
        trigger_parameters, f_low, fudge_length=fudge_length,
        fudge_min=fudge_min, minimum_data_length=minimum_data_length
    )
    data_start = trigger_parameters["time"] - 3 * data_len / 4
    data_end = trigger_parameters["time"] + data_len / 4

    strain = {}
    strain_f = {}
    if not isinstance(strain_data, dict):
        with open(strain_data[0], "r") as f:
            strain_data = json.load(f)
        channels = {
            ifo: value["channel"] for ifo, value in strain_data.items()
        }
        strain_data = {
            ifo: value["strain"] for ifo, value in strain_data.items()
        }
    else:
        channels = {ifo: f"{ifo}:{value}" for ifo, value in channels.items()}
    for key in strain_data.keys():
        data = TimeSeries.read(strain_data[key], channels[key]).to_pycbc()
        strain[key] = data.time_slice(data_start, data_end)
        strain_f[key] = strain[key].to_frequencyseries()
        strain_f[key].resize(int(data_len * f_high + 1))
    return strain, strain_f


def load_psd_from_file(
    psd_data, asd_data, delta_f, f_low, f_high,
):
    """Load a dictionary of PSDs or ASDs

    Parameters
    ----------
    psd_data: dict
        dictionary containing paths to PSDs. The keys should give the IFO
        and the item should give the path to the txt file containing the PSD
    asd_data: dict
        dictionary containing paths to ASDs. The keys should give the IFO
        and the item should give the path to the txt file containing the ASD
    delta_f: float
        the difference in frequency samples
    f_low: float
        low frequency cut-off to use for PSD generation
    f_high: float
        high frequency cut-off to use for PSD generation
    """
    if not len(psd_data) and not len(asd_data):
        raise ValueError("Please provide a PSD or ASD")
    elif len(psd_data) and len(asd_data):
        raise ValueError("Please provide either an ASD or PSD")
    elif len(asd_data):
        psd_data = asd_data
        _psd_kwargs = {"is_asd_file": True}
    elif len(psd_data):
        _psd_kwargs = {"is_asd_file": False}
    length = int(f_high / delta_f + 1)

    psd = {}
    psa = None

    for ifo, path in psd_data.items():
        p = pycbc.psd.read.from_txt(
            path, length, delta_f, f_low, **_psd_kwargs
        )
        if p.sample_frequencies[-1] < f_high:
            # need to extend the PSD to the desired high frequency,
            # use arbitrary PSD for only needed for finer time sampling
            if not psa:
                psa = aLIGOMidHighSensitivityP1200087(length, delta_f, f_low)
            psd[ifo] = copy.deepcopy(psa)
            psd[ifo][0:len(p)] = p[:]
        else:
            psd[ifo] = p

    return psd


def _estimate_data_length_from_template_parameters(
    template_parameters, f_low, fudge_length=1.1, fudge_min=0.02,
    minimum_data_length=16
):
    """Estimate the required data length for a set of parameters. This is
    based on a rough time estimate of the waveform that is produced by
    the provided set of parameters

    Parameters
    ----------
    template_parameters: dict
        dictionary of template parameters. Dictionary must contain an
        entry for 'mass_1'
    f_low: float
        low frequency cut off to use for estimate the rough time estimate of
        the waveform
    fudge_length: float, optional
        factor to multiple the rough time estimate of the waveform by to ensure
        that it is a conservative value. Default 1.1
    fudge_min: float, optional
        the minimum that the rough time estimate of the waveform can be.
        Default 0.02
    minimum_data_length: int, optional
        the minimum that the data length can be in seconds. Default 16

    Returns
    -------
    data_len: int
        the required data length to use
    """
    from pycbc.waveform.compress import rough_time_estimate
    wf_len = rough_time_estimate(
        template_parameters["mass_1"], template_parameters["mass_1"],
        f_low, fudge_length=fudge_length, fudge_min=fudge_min
    )
    return max(int(2**(np.ceil(np.log2(wf_len)))), minimum_data_length)


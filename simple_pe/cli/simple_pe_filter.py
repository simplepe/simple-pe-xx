#! /usr/bin/env python

import os
from argparse import ArgumentParser
import json
import copy
import numpy as np
from gwpy.timeseries import TimeSeries
from pesummary.core.command_line import CheckFilesExistAction, DictionaryAction
from pesummary.gw.conversions.snr import _calculate_precessing_harmonics 
from pycbc.psd.analytical import flat_unity
import pycbc.psd.read
from pycbc.filter.matchedfilter import sigmasq
from pycbc.detector import Detector
from simple_pe.param_est import metric, filter, pe
from simple_pe.waveforms import waveform_modes
from simple_pe.fstat import fstat

# silence PESummary logger
import logging
_logger = logging.getLogger('PESummary')
_logger.setLevel(logging.CRITICAL + 10)

__author__ = [
    "Charlie Hoy <charlie.hoy@ligo.org>",
    "Stephen Fairhurst <stephen.fairhurst@ligo.org>"
]


def command_line():
    """Define the command line arguments for `simple_pe_filter`
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--trigger_parameters",
        help="json file containing the trigger parameters",
        action=CheckFilesExistAction,
        required=True,
    )
    parser.add_argument(
        "--strain",
        help=(
            "Time domain strain data to analyse. Must be provided as a space "
            "separated dictionary with keys giving the channel name, e.g. "
            "H1:HWINJ_INJECTED:/path/to/file L1:HWINJ_INJECTED/path/to/file "
        ),
        nargs="+",
        default={},
        action=DictionaryAction,
        required=True,
    )
    parser.add_argument(
        "--asd",
        help=(
            "ASD files to use for the analysis. Must be provided as a space "
            "separated dictionary, e.g. H1:path/to/file L1:path/to/file"
        ),
        nargs="+",
        default={},
        action=DictionaryAction,
    )
    parser.add_argument(
        "--psd",
        help=(
            "PSD files to use for the analysis. Must be provided as a space "
            "separated dictionary, e.g. H1:path/to/file L1:path/to/file"
        ),
        nargs="+",
        default={},
        action=DictionaryAction,
    )
    parser.add_argument(
        "--approximant",
        help="Approximant to use for the analysis",
        default="IMRPhenomXPHM",
    )
    parser.add_argument(
        "--f_low",
        help="Low frequency cutoff to use for the analysis",
        default=15.,
        type=float,
    )
    parser.add_argument(
        "--f_high",
        help=(
            "High frequency cutoff to use for the analysis. This also sets "
            "the sample rate (defined as f_high * 2)"
        ),
        default=8192.,
        type=float,
    )
    parser.add_argument(
        "--minimum_data_length",
        help="Minimum data length to use for the analysis",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--outdir",
        help="Directory to store the output",
        default="./",
    )
    return parser


def _load_trigger_parameters_from_file(path):
    """Load the trigger parameters from file

    Parameters
    ----------
    path: str
        path to file containing trigger parameters

    Returns
    -------
    data: SimplePESamples
        a SimplePESamples object containing trigger parameters
    """
    with open(path, "r") as f:
        data = json.load(f)
    data = pe.SimplePESamples(data)
    required_params = [
        "mass_1", "mass_2", "time", "ra", "dec", "psi"
    ]
    data = pe.convert(data, disable_remnant=True)
    if not all(param in data.keys() for param in required_params):
        raise ValueError(
            f"{path} does not include all required parameters: "
            f"{','.join(required_params)}"
        )
    return data


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
        the minimum that the rough time estimate of the waveform can be. Default
        0.02
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


def _load_strain_data_from_file(
    trigger_parameters, strain_data, f_low, f_high, fudge_length=1.1,
    fudge_min=0.02, minimum_data_length=16
):
    """

    Parameters
    ----------
    trigger_parameters: dict
        dictionary containing trigger parameters
    strain_data: dict
        dictionary containing paths to strain data. The keys should be of the
        form '{ifo}:{channel}' and the items should be the of the form
        '{path_to_file}'
    f_low: float
        low frequency cut off to use for the analysis
    f_high: float
        high frequency cut off to use for the analysis
    fudge_length: float, optional
        factor to multiple the rough time estimate of the waveform by to ensure
        that it is a conservative value. Default 1.1
    fudge_min: float, optional
        the minimum that the rough time estimate of the waveform can be. Default
        0.02
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
    for key, fname in strain_data.items():
        ifo, channel = key.split(":")
        data = TimeSeries.read(fname, f"{ifo}:{channel}").to_pycbc()
        strain[ifo] = data.time_slice(data_start, data_end) 
        strain_f[ifo] =  strain[ifo].to_frequencyseries()
        strain_f[ifo].resize(int(data_len * f_high + 1))
    return strain, strain_f


def _load_psd_from_file(
    psd_data, asd_data, length, data_length, delta_f, f_low
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
    length: int
        the length of the PSD to produce
    data_length: int
        the length of data being used for the analysis
    delta_f: float
        the difference in frequency samples
    f_low: float
        low frequency cut-off to use for PSD generation
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
    psd = {}
    psa = flat_unity(length, delta_f, f_low)
    for ifo, path in psd_data.items():
        p = pycbc.psd.read.from_txt(
            path, data_length, delta_f, f_low, **_psd_kwargs
        )
        psd[ifo] = copy.deepcopy(psa)
        psd[ifo][0:len(p)] = p[:]
    hm_psd = 3. / sum([1. / item for item in psd.values()])
    psd["hm"] = hm_psd
    return psd


def find_peak(
    trigger_parameters, strain_f, psd, approximant, f_low, t_start, t_end,
    dx_directions=["chirp_mass", "symmetric_mass_ratio", "chi_align"],
    fixed_directions=["distance"], method="scipy"
):
    """Find the peak template given a starting point

    Parameters
    ----------
    trigger_parameters: dict
        dictionary of trigger parameters to use as starting point
    strain_f: dict
        dictionary of frequency domain strain data
    psd: dict
        dictionary of PSDs
    approximant: str
        approximant to use for the analysis
    f_low: float
        low frequency cutoff to use for the analysis
    t_start: float
        time to start the analysis.
    t_end: float
        time to end the analysis.
    dx_directions: list, optional
        directions to calculate the metric. Default
        ["chirp_mass", "symmetric_mass_ratio", "chi_align"]
    fixed_directions: list, optional
        directions to keep fixed when generating the metric. Default
        ["distance"]
    method: str, optional
        the method to use for optimising the template parameters. Default
        'scipy'
    """
    _psd = psd.copy()
    _psd.pop("hm", None)
    event_info = {
        k: trigger_parameters[k] for k in dx_directions + fixed_directions
    }
    x_peak, snr_peak = filter.find_peak_snr(
        list(strain_f.keys()), strain_f, _psd, t_start, t_end, event_info, 
        dx_directions, f_low, approximant, method=method
    )
    x_peak = pe.convert(x_peak, disable_remnant=True)
    peak_template = pe.SimplePESamples(x_peak)
    peak_template.generate_spin_z()
    # if necessary move away from equal mass
    if peak_template["mass_1"] == peak_template["mass_2"]:
        peak_template["mass_1"] += 0.1
        peak_template["mass_2"] -= 0.1
    peak_template.update(
       {
           key: value for key, value in trigger_parameters.items() if key in
           ["ra", "dec", "psi", "time"]
       }
    )
    event_snr = {"network": snr_peak}
    return peak_template, event_snr


def calculate_higher_multipole_snr(
    peak_template, psd, approximant, strain_f, f_low, t_start, t_end,
    multipoles=['22', '21', '33', '44']
):
    """Calculate the SNR in each of the higher order multipoles for the
    peak template

    Parameters
    ----------
    peak_template: dict
        dictionary of parameters correspond to peak template
    psd: dict
        dictionary of PSDs
    approximant: str
        approximant to use when calculating the SNR
    strain_f: dict
        dictionary of frequency domain strain data
    f_low: float
        low frequency cutoff to use for the analysis
    t_start: float
        time to start the analysis.
    t_end: float
        time to end the analysis.
    multipoles: list, optional
        list of multipoles to calculate the SNR for. Default
        ['22', '21', '33', '44']
    """
    z_hm = {}
    z_hm_perp = {}
    ifos = [key for key in psd if key != "hm"]
    for ifo in ifos:
        h, h_perp, sigmas, zetas = waveform_modes.calculate_hm_multipoles(
            peak_template["mass_1"], peak_template["mass_2"],
            peak_template["spin_1z"], peak_template["spin_2z"], psd["hm"],
            f_low, approximant, modes=multipoles
        )
        z_hm[ifo], z_hm_perp[ifo] = _calculate_mode_snr(
            strain_f[ifo], psd[ifo], t_start, t_end, f_low, multipoles,
            h, h_perp
        )
    _, _, _, hm_net_snr_perp = waveform_modes.network_mode_snr(
        z_hm, z_hm_perp, ifos, multipoles, dominant_mode='22'
    )
    return hm_net_snr_perp, z_hm


def calculate_precession_snr(
    peak_template, psd, strain_f, f_low, delta_f, f_high, t_start, t_end,
    fiducial_chi_p=0.05, harmonics=['0', '1']
):
    """Calculate the SNR from precession for the peak template

    Parameters
    ----------
    peak_template: dict
        dictionary of parameters correspond to peak template
    psd: dict
        dictionary of PSDs
    strain_f: dict
        dictionary of frequency domain strain data
    f_low: float
        low frequency cutoff to use for the analysis
    delta_f: float
        difference between frequency samples to use for the analysis
    f_high: float
        high frequency cutoff to use for the analysis
    t_start: float
        time to start the analysis.
    t_end: float
        time to end the analysis.
    fiducial_chi_p: float, optional
        fiducial chi_p to use when estimate the SNR in precession from an
        aligned spin template. Default 0.05
    harmonics: list, optional
        precession harmonics to calculate. Default ['0', '1']
    """
    _prec_parameters = pe.SimplePESamples(peak_template.copy())
    _prec_parameters.update(
        {
            "chi_p": np.array([fiducial_chi_p]), "phase": np.array([0.]),
            "f_ref": np.array([f_low])
        }
    )
    _prec_parameters.generate_prec_spin()
    _prec_parameters.generate_all_posterior_samples()
    hp = _calculate_precessing_harmonics(
        _prec_parameters["mass_1"][0], _prec_parameters["mass_2"][0],
        _prec_parameters["a_1"][0], _prec_parameters["a_2"][0],
        _prec_parameters["tilt_1"][0], _prec_parameters["tilt_2"][0],
        _prec_parameters["phi_12"][0], _prec_parameters["beta"][0],
        _prec_parameters["distance"][0], harmonics=[0, 1],
        approx="IMRPhenomPv2", df=delta_f, f_lower=f_low, f_final=f_high
    )
    hprec = {'0': hp[0], '1': hp[1]}
    z_prec = {}
    z_prec_perp = {}
    ifos = [key for key in psd if key != "hm"]
    for ifo in ifos:
        h_perp, sigma, zeta = waveform_modes.orthonormalize_modes(
            hprec, psd[ifo], f_low, harmonics, dominant_mode='0'
        )
        z_prec[ifo], z_prec_perp[ifo] = _calculate_mode_snr(
            strain_f[ifo], psd[ifo], t_start, t_end, f_low, harmonics,
            hprec, h_perp, dominant_mode='0'
        )
    _, _, _, prec_net_snr_perp = waveform_modes.network_mode_snr(
        z_prec, z_prec_perp, ifos, ['0','1'], dominant_mode='0'
    )
    return {"prec": prec_net_snr_perp['1']}


def _calculate_dominant_polarisation(
    peak_template, psd, approximant, f_low, delta_f, f_high
):
    """Calculate the dominant polarisation

    Parameters
    ----------
    peak_template: dict
        dictionary of parameters correspond to peak template
    psd: dict
        dictionary of PSDs
    approximant: str
        approximant to use for the analysis
    f_low: float
        low frequency cutoff to use for the analysis
    delta_f: float
        difference between frequency samples to use for the analysis
    f_high: float
        high frequency cutoff to use for the analysis
    """
    f_sig = []
    f_cplx = {}
    h = metric.make_waveform(
        peak_template, delta_f, f_low, len(list(psd.values())[0]),
        approximant=approximant
    )
    ifos = [key for key in psd if key != "hm"]
    for ifo in ifos:
        _fp, _fc = Detector(ifo).antenna_pattern(
            peak_template['ra'], peak_template['dec'],
            peak_template['psi'], peak_template['time'])
        sigma = np.sqrt(
            sigmasq(
                h, psd[ifo], low_frequency_cutoff=f_low,
                high_frequency_cutoff=f_high
            )
        )
        f_cplx[ifo] = sigma * (_fp + 1j * _fc)
        f_sig.append(sigma * np.array([_fp, _fc]))

    f_sig = np.array(f_sig)
    fp, fc, _ = fstat.dominant_polarization(f_sig)
    return fp, fc, f_cplx


def calculate_second_polarization_snr(
    peak_template, psd, approximant, strain_f, f_low, delta_f, f_high,
    dominant_waveform
):
    """Calculate the SNR in the second polarisation for the peak template

    Parameters
    ----------
    peak_template: dict
        dictionary of parameters correspond to peak template
    psd: dict
        dictionary of PSDs
    approximant: str
        approximant to use for the analysis
    strain_f: dict
        dictionary of frequency domain strain data
    f_low: float
        low frequency cutoff to use for the analysis
    delta_f: float
        difference between frequency samples to use for the analysis
    f_high: float
        high frequency cutoff to use for the analysis
    dominant_waveform: dict
        dictionary containing the 22 waveform as seen in each of the detectors
    """
    ifos = [key for key in psd if key != "hm"]
    fp, fc, f_cplx = _calculate_dominant_polarisation(
        peak_template, psd, approximant, f_low, delta_f, f_high
    )
    alpha = fc / fp
    z = [dominant_waveform[ifo] for ifo in ifos]
    fs = np.array([f_cplx[ifo] for ifo in ifos])
    fl = np.inner(z, fs.conj().T)
    fr = np.inner(z, fs.T)
    ff = np.linalg.norm(fs, axis=0)
    rho_not_l = np.sqrt(np.linalg.norm(z)**2 - np.abs(fl / ff)**2)
    rho_not_r = np.sqrt(np.linalg.norm(z)**2 - np.abs(fr / ff)**2)
    peak_template.update(
        {
            "f_plus": np.array([fp]), "f_cross": np.array([fc]),
            "net_alpha": np.array([alpha])
        }
    )
    return {"not_left": rho_not_l[0], "not_right": rho_not_r[0]}


def _calculate_mode_snr(
    strain_f, psd, t_start, t_end, f_low, harmonics, h, h_perp, **kwargs
):
    """Wrapper for the waveform_modes.calculate_mode_snr function

    Parameters
    ----------
    strain_f: dict
        dictionary of frequency domain strain data
    psd: dict
        dictionary of PSDs to use
    t_start: float
        time to start the analysis.
    t_end: float
        time to end the analysis.
    harmonics: list
        list of harmonics to calculate the SNR for
    h: pycbc.frequencyseries.FrequencySeries
        frequency domain waveform
    h_perp: pycbc.frequencyseries.FrequencySeries
        frequency domain perpendicular waveform
    """
    aligned = waveform_modes.calculate_mode_snr(
        strain_f, psd, h, t_start, t_end, f_low, harmonics, **kwargs
    )
    perp = waveform_modes.calculate_mode_snr(
        strain_f, psd, h_perp, t_start, t_end, f_low, harmonics, **kwargs
    )
    return aligned, perp


def estimate_face_on_distance(
    peak_template, event_snr, psd, approximant, f_low, delta_f, f_high
):
    """Estimate the face-on distance for the peak template

    Parameters
    ----------
    peak_template: dict
        dictionary of parameters correspond to peak template
    event_snr: dict
        dictionary containing the SNRs associated with the peak template
    psd: dict
        dictionary of PSDs
    approximant: str
        approximant to use for the analysis
    f_low: float
        low frequency cutoff to use for the analysis
    delta_f: float
        difference between frequency samples to use for the analysis
    f_high: float
        high frequency cutoff to use for the analysis
    """
    h = metric.make_waveform(
        peak_template, delta_f, f_low, len(list(psd.values())[0]),
        approximant=approximant
    )
    sigma_hm = np.sqrt(
        sigmasq(
            h, psd["hm"], low_frequency_cutoff=f_low, high_frequency_cutoff=f_high
        )
    )
    f_net = np.sqrt(
        peak_template["f_plus"]**2 + peak_template["f_cross"]**2
    ) / sigma_hm
    return {
        "distance_face_on": (
            peak_template['distance'] * f_net * sigma_hm  / event_snr['network']
        )
    }


def main(args=None):
    """Main interface for `simple_pe_filter`
    """
    parser = command_line()
    opts, _ = parser.parse_known_args(args=args)
    if not os.path.isdir(opts.outdir):
        os.mkdir(opts.outdir)
    trigger_parameters = _load_trigger_parameters_from_file(
        opts.trigger_parameters
    )
    strain, strain_f = _load_strain_data_from_file(
        trigger_parameters, opts.strain, opts.f_low, opts.f_high,
        minimum_data_length=opts.minimum_data_length
    )
    delta_f = list(strain_f.values())[0].delta_f
    psd = _load_psd_from_file(
        opts.psd, opts.asd, int(opts.f_high * 2 / (delta_f * 2) + 1),
        int(len(list(strain.values())[0]) / 2.) + 1, delta_f, opts.f_low,
    )
    t_start = trigger_parameters["time"] - 0.1 # this time window should be an option
    t_end = trigger_parameters["time"] + 0.1
    peak_parameters, event_snr = find_peak(
        trigger_parameters, strain_f, psd, opts.approximant, opts.f_low,
        t_start, t_end
    )
    _snrs, z_hm = calculate_higher_multipole_snr(
        peak_parameters, psd, opts.approximant, strain_f, opts.f_low,
        t_start, t_end
    )
    event_snr.update(_snrs)
    event_snr.update(
        calculate_precession_snr(
            peak_parameters, psd, strain_f, opts.f_low, delta_f, opts.f_high,
            t_start, t_end
        )
    )
    event_snr.update(
        calculate_second_polarization_snr(
            peak_parameters, psd, opts.approximant, strain_f, opts.f_low,
            delta_f, opts.f_high, {key: value['22'] for key, value in z_hm.items()}
        )
    )
    peak_parameters.update(
        estimate_face_on_distance(
            peak_parameters, event_snr, psd, opts.approximant, opts.f_low,
            delta_f, opts.f_high
        )
    )
    peak_parameters.write(
        outdir=opts.outdir, filename="peak_parameters.json", overwrite=True,
        file_format="json"
    )
    pe.SimplePESamples(
        {key: [value] for key, value in event_snr.items()}
    ).write(
        outdir=opts.outdir, filename="peak_snrs.json", overwrite=True,
        file_format="json"
    )


if __name__ == "__main__":
    main()

#! /usr/bin/env python

import os
from argparse import ArgumentParser
import numpy as np
from pesummary.core.command_line import DictionaryAction
from pesummary.gw.conversions.snr import _calculate_precessing_harmonics, \
    _mode_array_map
from pesummary.core.reweight import rejection_sampling
from pesummary.utils.samples_dict import SamplesDict
from pycbc.filter.matchedfilter import sigma
from simple_pe.detectors import calc_reach_bandwidth, Network
from simple_pe.localization import event
from simple_pe.param_est import filter, pe
from simple_pe.waveforms import make_waveform
from simple_pe import waveforms
from simple_pe import io

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
        "--seed",
        help="random seed to set for reproducibility",
        default=123456789,
        type=int
    )
    parser.add_argument(
        "--trigger_parameters",
        help=(
            "Either a json file containing the trigger parameters or a space "
            "separated dictionary giving the trigger parameters, e.g. "
            "mass1:10 mass2:5"
        ),
        action=DictionaryAction,
        default=None,
    )
    parser.add_argument(
        "--snr_threshold",
        help="SNR threshold to use for localization. Default 4",
        default=4,
        type=float
    )
    parser.add_argument(
        "--strain",
        help=(
            "Time domain strain data to analyse. Must be provided as a space "
            "separated dictionary with keys giving the ifo and items giving "
            "the path to the strain data you wish to analyse, e.g. "
            
            "H1:/path/to/file or a single json file giving the path to file "
            "and channel name for each ifo, e.g. {H1: {strain: /path/to/file,"
            "channel: channel}}. Strain data must be a gwf file"
        ),
        nargs="+",
        default=None,
        action=DictionaryAction,
    )
    parser.add_argument(
        "--channels",
        help=(
            "Channels to use when reading in strain data. Must be provided as "
            "a space separated dictionary with keys giving the ifo and items "
            "giving the channel name, e.g. H1:HWINJ_INJECTED. For GWOSC open "
            "data the dictionary items must be GWOSC, e.g. H1:GWOSC. If you "
            "wish to use simple-pe to produce an injection for you, the "
            "dictionary items must be INJ, e.g. H1:INJ. Only used if strain is "
            "not a json file"
        ),
        nargs="+",
        default={},
        action=DictionaryAction,
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
        "--time_window",
        help="Window around given time to search for SNR peak",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--outdir",
        help="Directory to store the output",
        default="./",
    )
    parser.add_argument(
        "--metric_directions",
        help="Directions to calculate metric",
        nargs="+",
        default=['chirp_mass', 'symmetric_mass_ratio', 'chi_align']
    )
    parser.add_argument(
        "--bayestar_localization",
        help="File containing details of Bayestar localization.  "
             "Default=None",
        default=None,
        type=str
    )
    return parser


def find_peak(
    trigger_parameters, strain_f, psd, approximant, f_low,
    time_window, dx_directions=None,
    fixed_directions=None, method="scipy"
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
    time_window: float
        width of time window around trigger time
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
    if not dx_directions:
        dx_directions = ["chirp_mass", "symmetric_mass_ratio", "chi_align"]
    if not fixed_directions:
        fixed_directions = ["distance", "chi_p"]

    if not trigger_parameters["_precessing"] and "chi_p" in fixed_directions:
        fixed_directions.remove("chi_p")

    t_start = trigger_parameters["time"] - time_window
    t_end = trigger_parameters["time"] + time_window
    ifos = list(strain_f.keys())
    delta_f = list(strain_f.values())[0].delta_f

    event_info = {
        k: trigger_parameters[k] for k in dx_directions + fixed_directions
    }
    # find dominant harmonic peak
    x_peak, snr_peak = filter.find_peak_snr(
        ifos, strain_f, psd, t_start, t_end, event_info,
        dx_directions, f_low, approximant, method=method,
        harm2=trigger_parameters["_precessing"]
    )  # Harm2 should probably be False here -- we do the 2-harm peak later

    x_peak = pe.convert(x_peak, disable_remnant=True)
    print("Found dominant harmonic peak with SNR = %.4f" % snr_peak)
    for k, v in x_peak.items():
        print("%s = %.4f" % (k, v))

    if trigger_parameters["_precessing"]:
        # find two-harmonic peak
        event_info = {
            k: x_peak[k] for k in dx_directions + fixed_directions
        }
        x_2h_peak, snr_peak = filter.find_peak_snr(
            ifos, strain_f, psd, t_start, t_end, event_info,
            dx_directions, f_low, approximant, method=method, harm2=True
        )
        x_2h_peak = pe.convert(x_2h_peak, disable_remnant=True)

        print("Found two harmonic peak with SNR = %.4f" % snr_peak)
        for k, v in x_2h_peak.items():
            print("%s = %.4f" % (k, v))

        peak_template = pe.SimplePESamples(x_2h_peak)

    else:
        peak_template = pe.SimplePESamples(x_peak)

    peak_template.add_fixed('phase', 0.)
    peak_template.add_fixed('f_ref', f_low)
    peak_template.add_fixed('theta_jn', 0.)
    peak_template.generate_spin_z()
    peak_template.generate_prec_spin()
    peak_template.generate_all_posterior_samples(f_low=f_low, f_ref=f_low,
                                                 delta_f=delta_f,
                                                 disable_remnant=True)

    return peak_template, snr_peak


def calculate_snrs_and_sigma(
        peak_template, psd, approximant, strain_f, f_low, t_start, t_end):
    """Calculate the individual ifo SNRs and times as well as network SNR

    Parameters
    ----------
    peak_template: dict
        dictionary of parameters corresponding to peak template
    psd: dict
        dictionary of PSDs
    approximant: approximant

    strain_f: dict
        dictionary of frequency domain strain data
    f_low: float
        low frequency cutoff to use for the analysis
    t_start: float
        time to start the analysis.
    t_end: float
        time to end the analysis.
    """
    ifos = list(strain_f.keys())
    delta_f = list(psd.values())[0].delta_f
    f_high = list(psd.values())[0].sample_frequencies[-1]

    h = make_waveform(
        peak_template, delta_f, f_low, len(list(psd.values())[0]),
        approximant=approximant
    )

    _sig = sigma(h, io.calculate_harmonic_mean_psd(psd),
                 low_frequency_cutoff=f_low,
                 high_frequency_cutoff=f_high)

    net_snr, ifo_snr, ifo_time = filter.matched_filter_network(
        ifos, strain_f, psd, t_start, t_end, h, f_low
    )
    _snr = {"ifo_snr": ifo_snr,
            "ifo_time": ifo_time,
            "network": net_snr}

    return _snr, _sig


def calculate_subdominant_snr(
    peak_template, psd, approximant, strain_f, f_low, t_start, t_end,
    multipoles=None
):
    """Calculate the SNR in each of the higher order multipoles for the
    peak template

    Parameters
    ----------
    peak_template: dict
        dictionary of parameters corresponding to peak template
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
        ['22', '33', '44']
    """
    if multipoles is None:
        multipoles = ['22', '33', '44']

    ifos = list(strain_f.keys())

    # if necessary move away from equal mass
    if peak_template["mass_1"] == peak_template["mass_2"]:
        peak_template["mass_1"] += 0.1
        peak_template["mass_2"] -= 0.1

    z_hm = {}
    z_hm_perp = {}

    h_hm, h_hm_perp, sigmas, zetas = waveforms.calculate_hm_multipoles(
        peak_template["mass_1"], peak_template["mass_2"],
        peak_template["spin_1z"], peak_template["spin_2z"],
        io.calculate_harmonic_mean_psd(psd),
        f_low, approximant, multipoles, '22',
        peak_template["spin_1x"], peak_template["spin_1y"],
        peak_template["spin_2x"], peak_template["spin_2y"]
    )
    for ifo in ifos:
        z_hm[ifo], z_hm_perp[ifo] = _calculate_mode_snr(
            strain_f[ifo], psd[ifo], t_start, t_end, f_low, multipoles,
            h_hm, h_hm_perp
        )
    _, hm_net_snr_perp = waveforms.network_mode_snr(
        z_hm_perp, ifos, multipoles, dominant_mode='22'
    )
    _snr = {}
    for lm in multipoles:
        _snr[lm] = hm_net_snr_perp[lm]
        _snr['overlap_{lm}'] = abs(zetas[lm])

    return _snr


def calculate_precession_snr(
    peak_template, psd, approximant, strain_f, f_low, t_start, t_end,
    harmonics=None
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
    t_start: float
        time to start the analysis.
    t_end: float
        time to end the analysis.
    harmonics: list, optional
        precession harmonics to calculate. Default ['0', '1']
    """
    if harmonics is None:
        harmonics = [0, 1]

    delta_f = list(strain_f.values())[0].delta_f
    f_high = list(psd.values())[0].sample_frequencies[-1]
    ifos = list(strain_f.keys())

    try:
        # only works for FD approximants
        mode_array = _mode_array_map('22', approximant)
        hp = _calculate_precessing_harmonics(
            peak_template["mass_1"][0], peak_template["mass_2"][0],
            peak_template["a_1"][0], peak_template["a_2"][0],
            peak_template["tilt_1"][0], peak_template["tilt_2"][0],
            peak_template["phi_12"][0], peak_template["beta"][0],
            peak_template["distance"][0], harmonics=harmonics,
            approx=approximant, mode_array=mode_array, df=delta_f,
            f_lower=f_low, f_final=f_high
        )
    except Exception:
        mode_array = _mode_array_map('22', "IMRPhenomPv2")
        hp = _calculate_precessing_harmonics(
            peak_template["mass_1"][0], peak_template["mass_2"][0],
            peak_template["a_1"][0], peak_template["a_2"][0],
            peak_template["tilt_1"][0], peak_template["tilt_2"][0],
            peak_template["phi_12"][0], peak_template["beta"][0],
            peak_template["distance"][0], harmonics=harmonics,
            approx="IMRPhenomPv2", mode_array=mode_array, df=delta_f,
            f_lower=f_low, f_final=f_high
        )

    z_prec = {}
    z_prec_perp = {}
    overlap_prec = {}
    for ifo in ifos:
        h_perp, sig, zeta = waveforms.orthonormalize_modes(
            hp, psd[ifo], f_low, [0, 1], dominant_mode=0
        )
        overlap_prec[ifo] = zeta[1]
        z_prec[ifo], z_prec_perp[ifo] = _calculate_mode_snr(
            strain_f[ifo], psd[ifo], t_start, t_end, f_low, [0, 1], hp, h_perp,
            dominant_mode=0
        )
    _, prec_net_snr_perp = waveforms.network_mode_snr(
        z_prec_perp, ifos, [0, 1], 0
    )
    _snr = {"prec": prec_net_snr_perp[1],
            "overlap_prec": abs(zeta[1])}

    return _snr


def calculate_localisation_information(
        peak_template, psd, approximant, f_low,
        event_snr, threshold, trigger_parameters, bayestar_localization=None
):
    """Calculate the localization information for the event and use this
    to provide the network sensitivity, the sensitivity to the second
    polarization and the SNR in the second polarization for the peak template

    Parameters
    ----------
    peak_template: SimplePESamples
        dictionary of parameters that correspond to peak template
    psd: dict
        dictionary of PSDs
    approximant: str
        approximant to use for the analysis
    f_low: float
        low frequency cutoff to use for the analysis
    event_snr: dict
        dictionary of snrs of the candidate GW
    threshold:
        threshold for individual ifo to contribute to localization
    trigger_parameters: dict
        dictionary of trigger parameters to use as starting point
    bayestar_localization: string
        name of file containing Bayestar localization information
    """
    # Check consistency of localization request:
    single_sky_point = all(param in trigger_parameters for
                           param in ["ra", "dec"])
    if any(param in trigger_parameters for param in ["ra", "dec"]) and \
            not single_sky_point:
        raise ValueError("Please provide samples for both ra and dec")
    elif bayestar_localization and single_sky_point:
        raise ValueError(
            "Please specify either 'bayestar_localization' or provide "
            "an estimate of 'ra' and 'dec' but not both")

    # generate network and sensitivity
    ifos = list(event_snr['ifo_snr'].keys())
    net = Network(threshold=10.)

    for ifo in ifos:
        hor, f_mean, f_band = calc_reach_bandwidth(
            peak_template['mass_1'], peak_template['mass_2'],
            peak_template["chi_align"],
            approximant, psd[ifo], f_low
        )
        net.add_ifo(ifo, hor, f_mean, f_band, bns_range=False,
                    loc_thresh=threshold)

    if bayestar_localization or single_sky_point:
        # Use the localization information provided
        if single_sky_point:
            ra = [trigger_parameters['ra']]
            dec = [trigger_parameters['dec']]

        else:
            from ligo.skymap.io.fits import read_sky_map
            import astropy_healpix as ah
            import healpy as hp

            probs, _ = read_sky_map(bayestar_localization)
            npix = len(probs)
            nside = ah.npix_to_nside(npix)
            codec, ra = hp.pix2ang(nside, np.arange(npix))
            pts = SamplesDict({'ra': ra, 'dec': np.pi/2 - codec})
            pts = rejection_sampling(pts, probs).downsample(int(1e3))
            ra = pts['ra']
            dec = pts['dec']
            # should add distance??

        # calculate network sensitivity at points
        f_sig = np.zeros_like(ra)
        alpha_net = np.zeros_like(ra)
        snr_left = np.zeros_like(ra)
        snr_right = np.zeros_like(ra)

        for i, (_ra, _dec) in enumerate(zip(ra, dec)):
            ee = event.Event.from_snrs(
                net, event_snr["ifo_snr"], event_snr["ifo_time"],
                peak_template['chirp_mass'], _ra, _dec
            )
            ee.calculate_sensitivity()
            f_sig[i] = ee.sensitivity
            alpha_net[i] = ee.alpha_net()
            # need to account for time offsets in calculating SNRs
            snr_left[i] = np.linalg.norm(ee.projected_snr('left'))
            snr_right[i] = np.linalg.norm(ee.projected_snr('right'))

    else:
        # Localize based upon measured SNRs and times
        ev = event.Event.from_snrs(
            net, event_snr["ifo_snr"], event_snr["ifo_time"],
            peak_template['chirp_mass']
        )
        ev.calculate_mirror()
        ev.localize_all(methods=['coh', 'left', 'right'])

        if ev.localized >= 3:
            # generate points from localization region to calculate F+ and Fx
            coh = ev.localization['coh']
            if ev.mirror and (ev.mirror_loc['coh'].snr >
                              ev.localization['coh'].snr):
                coh = ev.mirror_loc['coh']
            # calculate max left/right SNR, allowing offset from central point
            # to maximize SNR
            snr_left = ev.localization['left'].calculate_max_snr()
            snr_right = ev.localization['right'].calculate_max_snr()

            ra, dec = coh.generate_samples(npts=int(1e3), sky_weight=True)

        elif ev.localized == 1:
            # one detector so snr is consistent with left/right polarization
            snr_left = event_snr['network']
            snr_right = event_snr['network']
            # source is only localized by the antenna pattern
            npts = int(1e4)
            ra = np.random.uniform(0, 2 * np.pi, npts)
            dec = np.pi / 2 - np.arccos(np.random.uniform(-1, 1, npts))
            f = np.zeros_like(ra)
            det = ev.__getattribute__(ev.ifos[0])
            for i, (r, d) in enumerate(zip(ra, dec)):
                fp, fc = det.antenna_pattern(r, np.pi / 2 - d, ev.psi, ev.gps)
                f[i] = np.sqrt(fp ** 2 + fc ** 2)
            pts = SamplesDict({'ra': ra, 'dec': dec})
            pts = rejection_sampling(pts, f ** 3).downsample(int(1e3))
            ra = pts['ra']
            dec = pts['dec']

        else:
            raise KeyError(
                f"Unable to localize event from SNRs. This could be because "
                f"you are considering a network with less than 3 detectors, or "
                f"because the IFO SNRs are <{threshold}. The recovered IFO "
                f"SNRs are {', '.join([ifo + ':' + str(abs(event_snr['ifo_snr'][ifo])) for ifo in ifos])}"
            )

        # calculate network sensitivity at points
        f_sig = np.zeros_like(ra)
        alpha_net = np.zeros_like(ra)

        for i, (_ra, _dec) in enumerate(zip(ra, dec)):
            ee = event.Event.from_snrs(
                net, event_snr["ifo_snr"], event_snr["ifo_time"],
                peak_template['chirp_mass'], _ra, _dec
            )
            ee.calculate_sensitivity()
            f_sig[i] = ee.sensitivity
            alpha_net[i] = ee.alpha_net()

    # add information to peak template:
    peak_template.add_fixed('f_net', np.mean(f_sig) / peak_template['sigma'])
    peak_template.add_fixed("response_sigma",
                            np.std(f_sig) / np.mean(f_sig))
    peak_template.add_fixed("net_alpha", np.mean(alpha_net))
    peak_template.add_fixed("distance_face_on",
                            peak_template["distance"] * peak_template["f_net"] *
                            peak_template['sigma'] / event_snr["network"]
                            )
    _snrs = {"not_left": np.sqrt(np.min(event_snr['network']**2 -
                                        snr_left ** 2)),
             "not_right": np.sqrt(np.min(event_snr['network']**2 -
                                         snr_right ** 2))
             }

    return _snrs


def _calculate_mode_snr(
    strain_f, psd, t_start, t_end, f_low, harmonics, h, h_perp, **kwargs
):
    """Wrapper for the waveforms.calculate_mode_snr function

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
    aligned, _ = waveforms.calculate_mode_snr(
        strain_f, psd, h, t_start, t_end, f_low, harmonics, **kwargs
    )
    perp, _ = waveforms.calculate_mode_snr(
        strain_f, psd, h_perp, t_start, t_end, f_low, harmonics, **kwargs
    )
    return aligned, perp


def main(args=None):
    """Main interface for `simple_pe_filter`
    """
    parser = command_line()
    opts, _ = parser.parse_known_args(args=args)
    np.random.seed(opts.seed)
    if not os.path.isdir(opts.outdir):
        os.mkdir(opts.outdir)
    if isinstance(opts.trigger_parameters, list):
        opts.trigger_parameters = "".join(opts.trigger_parameters)
    trigger_parameters = io.load_trigger_parameters_from_file(
        opts.trigger_parameters, opts.approximant
    )
    strain, strain_f = io.load_strain_data_from_file(
        trigger_parameters, opts.strain, opts.channels, opts.f_low, opts.f_high,
        minimum_data_length=opts.minimum_data_length
    )
    delta_f = list(strain_f.values())[0].delta_f

    psd = io.load_psd_from_file(
        opts.psd, opts.asd, delta_f, opts.f_low, opts.f_high
    )

    peak_parameters, event_snr = find_peak(
        trigger_parameters, strain_f, psd, opts.approximant,
        opts.f_low, opts.time_window, dx_directions=opts.metric_directions
    )

    trig_start = trigger_parameters["time"] - opts.time_window
    trig_end = trigger_parameters["time"] + opts.time_window

    event_snr, _sigma = calculate_snrs_and_sigma(
        peak_parameters, psd, opts.approximant, strain_f, opts.f_low,
        trig_start, trig_end
    )
    peak_parameters.add_fixed("sigma", _sigma)

    _snrs = calculate_subdominant_snr(
        peak_parameters, psd, opts.approximant, strain_f, opts.f_low,
        trig_start, trig_end
    )
    event_snr.update(_snrs)

    _snrs = calculate_precession_snr(
        peak_parameters, psd, opts.approximant, strain_f, opts.f_low,
        trig_start, trig_end
    )
    event_snr.update(_snrs)

    _snrs = calculate_localisation_information(
        peak_parameters, psd, opts.approximant, opts.f_low,
        event_snr, opts.snr_threshold, trigger_parameters,
        opts.bayestar_localization
    )
    event_snr.update(_snrs)

    peak_parameters.write(
        outdir=opts.outdir, filename="peak_parameters.json", overwrite=True,
        file_format="json"
    )

    # cast the ifo SNRs to reals
    event_snr['ifo_snr_phase'] = {}
    for k, v in event_snr['ifo_snr'].items():
        event_snr['ifo_snr'][k] = abs(v)
        event_snr['ifo_snr_phase'][k] = np.angle(v)

    pe.SimplePESamples(
        {key: [value] for key, value in event_snr.items()}
    ).write(
        outdir=opts.outdir, filename="peak_snrs.json", overwrite=True,
        file_format="json"
    )


if __name__ == "__main__":
    main()

import numpy as np
import copy
from pycbc.waveform import td_approximants, fd_approximants, get_fd_waveform
from pycbc.filter.matchedfilter import sigma, overlap_cplx, matched_filter

mode_array_dict = {
    '22': [[2, 2], [2, -2]],
    '21': [[2, 1], [2, -1]],
    '33': [[3, 3], [3, -3]],
    '32': [[3, 2], [3, -2]],
    '44': [[4, 4], [4, -4]],
    '43': [[4, 3], [4, -3]]
}


def mode_array(mode, approx):
    """
    Return the mode array for a given approximant.  This returns either the
    mode or the positive and negative m modes, depending upon the approximant

    Parameters
    ----------
    mode: str
        the mode of interest
    approx: str
        the waveform approximant

    Returns
    -------
    mode_array: list
        list of modes corresponding to desired mode
    """
    # don't include negative m modes in Phenom models as will throw an error
    # - they are automatically added by these models anyway
    if approx in ['IMRPhenomPv3HM', 'IMRPhenomHM']:
        mode_array_idx = -1
    else:
        mode_array_idx = None

    return mode_array_dict[mode][:mode_array_idx]


def calculate_hm_multipoles(mass1, mass2, spin1z, spin2z, ifo_psd, f_low,
                            approximant, modes, dominant_mode='22',
                            spin1x=0., spin1y=0., spin2x=0., spin2y=0.):
    """
    Calculate the higher harmonic waveforms for given set of modes.
    Return waveforms and parts orthogonal to the (2,2)

    Parameters
    ----------
    mass1: float
        mass of primary
    mass2: float
        mass of secondary
    spin1z: float
        z-component of primary spin
    spin2z: float
        z-component of primary spin
    ifo_psd: psd
        to use when orthogonalizing
    f_low: float
        low frequency cutoff
    approximant: str
        waveform to use
    modes: list
        modes to consider
    dominant_mode: str
        mode to use when orthogonalizing, default is 22 mode
    spin1x: float
        spin1x
    spin1y: float
        spin1y
    spin2x: float
        spin2x
    spin2y: float
        spin2y

    Returns
    -------
    h: dict
        normalized waveform modes
    h_perp: dict
        orthonormalized waveform modes
    sigmas: dict
        waveform normalizations
    zetas: dict
        complex overlap with dominant mode
    """
    h = {}
    all_modes = copy.deepcopy(modes)

    if dominant_mode in all_modes:
        pass
    else:
        all_modes.append(dominant_mode)

    # generate the waveforms and normalize
    for lm in all_modes:
        if lm == '22':
            inc = 0.
        else:
            inc = np.pi / 2
        if approximant in fd_approximants():
            if approximant == "IMRPhenomPv3HM" and \
                    all(_ for _ in [spin1x, spin1y, spin2x, spin2y]):
                # Pv3 required 22.  But, as we're doing spin aligned,
                # HM only is OK
                approximant = "IMRPhenomHM"
            if approximant == "IMRPhenomPv3HM":
                h[lm], _ = get_fd_waveform(approximant=approximant,
                                           mass1=mass1, mass2=mass2,
                                           spin1x=spin1x, spin1y=spin1y,
                                           spin1z=spin1z,
                                           spin2x=spin2x, spin2y=spin2y,
                                           spin2z=spin2z,
                                           distance=1, delta_f=ifo_psd.delta_f,
                                           f_lower=f_low,
                                           f_final=ifo_psd.sample_frequencies[
                                               -1],
                                           inclination=inc,
                                           mode_array=mode_array('22',
                                                                 approximant))
                if lm != '22':
                    _h, _ = get_fd_waveform(approximant=approximant,
                                            mass1=mass1, mass2=mass2,
                                            spin1x=spin1x, spin1y=spin1y,
                                            spin1z=spin1z,
                                            spin2x=spin2x, spin2y=spin2y,
                                            spin2z=spin2z,
                                            distance=1, delta_f=ifo_psd.delta_f,
                                            f_lower=f_low,
                                            f_final=ifo_psd.sample_frequencies[
                                                -1],
                                            inclination=inc,
                                            mode_array=mode_array(lm,
                                                                  approximant) +
                                                       mode_array('22',
                                                                  approximant))
                    h[lm] = _h - h[lm]
            else:
                h[lm], _ = get_fd_waveform(approximant=approximant,
                                           mass1=mass1, mass2=mass2,
                                           spin1x=spin1x, spin1y=spin1y,
                                           spin1z=spin1z,
                                           spin2x=spin2x, spin2y=spin2y,
                                           spin2z=spin2z,
                                           distance=1, delta_f=ifo_psd.delta_f,
                                           f_lower=f_low,
                                           f_final=ifo_psd.sample_frequencies[
                                               -1],
                                           inclination=inc,
                                           mode_array=mode_array(lm,
                                                                 approximant))
        elif approximant in td_approximants():
            from simple_pe.waveforms.waveform import make_waveform
            params = {
                "mass_1": [mass1], "mass_2": [mass2], "spin_1z": [spin1z],
                "spin_2z": [spin2z], "distance": [1.], "phase": [0.],
                "f_ref": [f_low], "inc": [inc]
            }
            npts = int(ifo_psd.sample_frequencies[-1] / ifo_psd.delta_f) + 1
            h[lm] = make_waveform(
                params, ifo_psd.delta_f, f_low, npts, approximant=approximant,
                modes=mode_array(lm, approximant)
            )
        else:
            print("Bad approximant")
            return -1

    h_perp, sigmas, zetas = orthonormalize_modes(h, ifo_psd, f_low, all_modes,
                                                 dominant_mode)
    return h, h_perp, sigmas, zetas


def orthonormalize_modes(h, ifo_psd, f_low, modes, dominant_mode='22'):
    """
    Orthonormalize a set of waveforms for a given PSD
    Return normalized waveforms orthogonal to the dominant mode,
    sigmas and (complex) overlaps of original waveforms

    Parameters
    ----------
    h: dict
        dictionary of waveform modes
    ifo_psd: pycbc.Frequency_Series
        PSD to use for orthonormalization
    f_low: float
        low frequency cutoff
    modes: list
        modes to consider
    dominant_mode: str
        mode to use for orthonormalization

    Returns
    -------
    h_perp: dict
    orthonormalized waveforms
    sigmas: dict
        waveform normalizations, pre-orthogonalization
    zetas: dict
        complex overlap with dominant mode
    """
    sigmas = {}
    for mode in modes:

        try:
            sigmas[mode] = sigma(h[mode], ifo_psd, low_frequency_cutoff=f_low,
                                 high_frequency_cutoff=
                                 ifo_psd.sample_frequencies[-1])
            h[mode] /= sigmas[mode]
        except:
            print("No power in mode %s" % mode)
            h.pop(mode)

    zetas = {}
    h_perp = {}
    for mode in modes:
        zetas[mode] = overlap_cplx(h[dominant_mode], h[mode], psd=ifo_psd,
                                   low_frequency_cutoff=f_low,
                                   high_frequency_cutoff=
                                   ifo_psd.sample_frequencies[-1],
                                   normalized=True)

        # generate the orthogonal waveform
        if mode == dominant_mode:
            h_perp[mode] = h[mode]
        else:
            h_perp[mode] = (h[mode] - zetas[mode] * h[dominant_mode]) / \
                           (np.sqrt(1 - np.abs(zetas[mode]) ** 2))

    return h_perp, sigmas, zetas


def calculate_alpha_lm_and_overlaps(mass1, mass2, spin1z, spin2z, ifo_psd,
                                    f_low,
                                    approximant, modes, dominant_mode='22',
                                    spin1x=0., spin1y=0., spin2x=0., spin2y=0.):
    """
    Calculate the higher harmonic waveforms for given set of modes.
    Return waveforms and parts orthogonal to the (2,2)

    Parameters
    ----------
    mass1: float
        mass1
    mass2: float
        mass2
    spin1z: float
        spin1z
    spin2z: float
        spin2z
    ifo_psd: pycbc.Frequency_Series
        PSD to use when orthogonalizing
    f_low: float
        low frequency cutoff
    approximant: str
        waveform to use
    modes: list
        modes to consider
    dominant_mode: str
        mode to use when orthogonalizing
    spin1x: float
        spin1x (optional)
    spin1y: float
        spin1y (optional)
    spin2x: float
        spin2x (optional)
    spin2y: float
        spin2y (optional)

    Returns
    -------
    alpha_lm: dict
        relative amplitudes of modes
    overlap_lm: dict
        overlap of lm with 22
    """
    h, h_perp, sigmas, zetas = calculate_hm_multipoles(mass1, mass2, spin1z,
                                                       spin2z, ifo_psd, f_low,
                                                       approximant, modes,
                                                       dominant_mode,
                                                       spin1x, spin1y,
                                                       spin2x, spin2y)

    alpha_lm = {}
    overlap_lm = {}
    for k, s in sigmas.items():
        alpha_lm[k] = s / sigmas[dominant_mode]
        overlap_lm[k] = abs(zetas[k])

    return alpha_lm, overlap_lm


def calculate_mode_snr(strain_data, ifo_psd, waveform_modes, t_start, t_end,
                       f_low, modes, dominant_mode='22'):
    """
    Calculate the SNR in each of the modes, and also the orthogonal SNR
    strain_data: time series data from ifo

    Parameters
    ----------
    strain_data: pycbc.Time_Series
        the ifo data
    ifo_psd: pycbc.Frequency_Series
        PSD for ifo
    waveform_modes: dict
        dictionary of waveform modes (time/frequency series)
    t_start: float
        beginning of time window to look for SNR peak
    t_end: float
        end of time window to look for SNR peak
    f_low: float
        low frequency cutoff
    modes: list
        the modes to calculate SNR for
    dominant_mode: str
        mode that is used to define the peak time

    Returns
    -------
    z: dict
        dictionary of complex SNRs for each mode
    t: float
        the time of the max SNR
    """

    if dominant_mode not in waveform_modes.keys():
        print("Please give the waveform for the dominant mode")
        return

    s = matched_filter(waveform_modes[dominant_mode], strain_data, ifo_psd,
                       low_frequency_cutoff=f_low)
    snr = s.crop(t_start - s.start_time, s.end_time - t_end)

    # find the peak and use this for the other modes later
    i_max = snr.abs_arg_max()
    t_max = snr.sample_times[i_max]

    z = {}
    for mode in modes:
        s = matched_filter(waveform_modes[mode], strain_data, psd=ifo_psd,
                           low_frequency_cutoff=f_low,
                           high_frequency_cutoff=ifo_psd.sample_frequencies[-1],
                           sigmasq=None)
        snr_ts = s.crop(t_start - s.start_time, s.end_time - t_end)
        z[mode] = snr_ts[i_max]

    return z, t_max


def network_mode_snr(z, z_perp, ifos, modes, dominant_mode='22'):
    """
    Calculate the SNR in each of the modes for the network and also project
    onto space proportional to dominant mode

    Parameters
    ----------
    z: dict
        dictionary of dictionaries of SNRs in each mode (in each ifo)
    z_perp: dict
        dictionary of dictionaries of perpendicular SNRs in each mode (
        in each ifo)
    ifos: list
        list of ifos
    modes: list
        list of modes
    dominant_mode: str
        the mode with most power (for orthogonalization)

    Returns
    -------
    rss_snr: dict
        the root sum squared SNR in each mode
    rss_snr_perp: dict
        the root sum squared SNR orthogonal to the dominant mode
    net_snr: dict
        the SNR in each mode that is consistent (in amplitude and
        phase) with the dominant mode SNR
    net_perp_snr: dict
        the orthogonal SNR in each mode consistent with the dominant mode
    """

    z_array = {}
    z_perp_array = {}

    rss_snr = {}
    rss_snr_perp = {}

    for mode in modes:
        z_array[mode] = np.array([z[ifo][mode] for ifo in ifos])
        z_perp_array[mode] = np.array([z_perp[ifo][mode] for ifo in ifos])
        rss_snr[mode] = np.linalg.norm(z_array[mode])
        rss_snr_perp[mode] = np.linalg.norm(z_perp_array[mode])

    net_snr = {}
    net_snr_perp = {}

    for mode in modes:
        net_snr[mode] = np.abs(np.inner(z_array[dominant_mode],
                                        z_array[mode].conjugate())) / \
                        rss_snr[dominant_mode]
        net_snr_perp[mode] = \
            np.abs(np.inner(z_perp_array[dominant_mode],
                            z_perp_array[mode].conjugate())) / \
            rss_snr_perp[dominant_mode]

    return rss_snr, rss_snr_perp, net_snr, net_snr_perp

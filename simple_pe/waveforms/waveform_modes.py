import numpy as np
from pycbc.waveform import td_approximants, fd_approximants, get_fd_waveform, get_td_waveform
from pycbc.filter.matchedfilter import sigma, overlap_cplx, matched_filter

mode_array_dict = {
        '22': [[2, 2]],# [2, -2]]
        '44': [[4, 4]], #, [4, -4]],
        '33': [[3, 3]], #[3, -3]],
        '21': [[2, 1]], #[2, -1]]
    }

def calculate_hm_multipoles(mass1, mass2, spin1z, spin2z, ifo_psd, f_low = 20.,
                            modes=['22', '33', '44'], approximant="IMRPhenomXPHM",
                            dominant_mode='22',
                            spin1x=0., spin1y=0., spin2x=0., spin2y=0.):
    """
    Calculate the higher harmonic waveforms for given set of modes.
    Return waveforms and parts orthogonal to the (2,2)
    mass1: mass1
    mass2: mass2
    spin1z: spin1z
    spin2z: spin2z
    ifo_psd: psd to use when orthogonalizing
    f_low: low frequency cutoff
    modes: modes to consider
    approximant: waveform to use
    dominant_mode: mode to use when orthogonalizing
    spin1x:
    spin1y:
    spin2x:
    spin2y:
    """
    h = {}

    if dominant_mode in modes:
        pass
    else:
        modes.append(dominant_mode)

    # generate the waveforms and normalize
    for lm in modes:
        if approximant in fd_approximants():
            if approximant == "IMRPhenomPv3HM" and all(_ for _ in [spin1x, spin1y, spin2x, spin2y]):
                # Pv3 required 22.  But, as we're doing spin aligned, HM only is OK
                approximant = "IMRPhenomHM"
            if approximant == "IMRPhenomPv3HM":
                h[lm], _ =get_fd_waveform(approximant=approximant,
                                                    mass1=mass1, mass2=mass2, spin1x=spin1x, spin1y=spin1y,
                                                    spin1z=spin1z,
                                                    spin2x=spin2x, spin2y=spin2y, spin2z=spin2z,
                                                    distance=1, delta_f=ifo_psd.delta_f, f_lower=f_low,
                                                    f_final=ifo_psd.sample_frequencies[-1],
                                                    inclination=np.pi / 3, mode_array=[[2, 2]])

                if lm != '22':
                    _h, _ = get_fd_waveform(approximant=approximant,
                                                     mass1=mass1, mass2=mass2, spin1x=spin1x, spin1y=spin1y,
                                                     spin1z=spin1z,
                                                     spin2x=spin2x, spin2y=spin2y, spin2z=spin2z,
                                                     distance=1, delta_f=ifo_psd.delta_f, f_lower=f_low,
                                                     f_final=ifo_psd.sample_frequencies[-1],
                                                     inclination=np.pi / 3, mode_array=mode_array_dict[lm] + [[2, 2]])
                    h[lm] = _h - h[lm]
            else:
                h[lm], _ = get_fd_waveform(approximant=approximant,
                                                    mass1=mass1, mass2=mass2, spin1x=spin1x, spin1y=spin1y,
                                                    spin1z=spin1z,
                                                    spin2x=spin2x, spin2y=spin2y, spin2z=spin2z,
                                                    distance=1, delta_f=ifo_psd.delta_f, f_lower=f_low,
                                                    f_final=ifo_psd.sample_frequencies[-1],
                                                    inclination=np.pi / 3, mode_array=mode_array_dict[lm])
        elif approximant in td_approximants():
            h[lm], _ = get_td_waveform(approximant=approximant,
                                                mass1=mass1, mass2=mass2, spin1x=spin1x, spin1y=spin1y, spin1z=spin1z,
                                                spin2x=spin2x, spin2y=spin2y, spin2z=spin2z,
                                                distance=1, delta_t=ifo_psd.delta_t, f_lower=f_low,
                                                inclination=np.pi / 3, mode_array=mode_array_dict[lm])
            h[lm].resize(2 * (len(ifo_psd) - 1))
        else:
            print("Bad approximant")
            return -1

    h_perp, zeta = orthogonalize_modes(modes, h, ifo_psd, f_low, dominant_mode)

    return h, zeta, h_perp


def orthogonalize_modes(modes, h, ifo_psd, f_low = 20., dominant_mode='22'):
    """
    Orthogonalize a set of waveforms for a give PSD
    Return waveforms orthogonal to the dominant mode and overlaps
    modes: modes to consider
    h: dictionary of waveform modes
    ifo_psd: psd to use when orthogonalizing
    f_low: low frequency cutoff
    dominant_mode: mode to use when orthogonalizing
    """
    for mode in modes:

        try:
            sig = sigma(h[mode], ifo_psd, low_frequency_cutoff=f_low,
                        high_frequency_cutoff=ifo_psd.sample_frequencies[-1])
            h[mode] /= sig
        except:
            print("No power in mode %s" % mode)
            h.pop(mode)

    zeta = {}
    h_perp = {}
    for mode in modes:
        zeta[mode] = overlap_cplx(h[dominant_mode], h[mode], psd=ifo_psd, low_frequency_cutoff=f_low,
                                high_frequency_cutoff=ifo_psd.sample_frequencies[-1], normalized=True)

        # generate the orthogonal waveform
        if mode == dominant_mode:
            h_perp[mode] = h[mode]
        else:
            h_perp[mode] = (h[mode] - zeta[mode] * h[dominant_mode]) / (np.sqrt(1 - np.abs(zeta[mode]) ** 2))

    return h_perp, zeta


def calculate_mode_snr(strain_data, ifo_psd, waveform_modes, t_start, t_end, f_low=20.,
                     dominant_mode='22'):
    """
    Calculate the SNR in each of the modes, and also the orthogonal SNR
    strain_data: time series data from ifo
    ifo_psd: PSD for ifo
    waveform_modes: dictionary of waveform modes (time/frequency series)
    t_start: beginning of time window to look for SNR peak
    t_end: end of time window to look for SNR peak
    f_low: low frequency cutoff
    dominant_mode: mode that is used to define the peak time
    """

    if dominant_mode not in waveform_modes.keys():
        print("Please give the waveform for the dominant mode")
        return

    s = matched_filter(waveform_modes[dominant_mode], strain_data, ifo_psd, low_frequency_cutoff=f_low)
    snr = s.crop(t_start - s.start_time, s.end_time - t_end)

    # find the peak and use this for the other modes later
    i_max = snr.abs_arg_max()

    z = {}
    for mode, wav in waveform_modes.items():
        s = matched_filter(wav, strain_data, psd=ifo_psd, low_frequency_cutoff=f_low,
                           high_frequency_cutoff=ifo_psd.sample_frequencies[-1], sigmasq=None)
        snr_ts = s.crop(t_start - s.start_time, s.end_time - t_end)
        z[mode] = snr_ts[i_max]

    return z


def network_mode_snr(ifos, modes, z, z_perp, dominant_mode='22'):
    """
    Calculate the SNR in each of the modes for the network and also project onto space proportional to dominant mode
    ifos: list of ifos
    z: dictionary of dictionaries of SNRs in each mode (in each ifo)
    z_perp: dictionary of dictionaries of perpendicular SNRs in each mode (in each ifo)
    dominant_mode: the mode with most power (for orthogonalization)
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
        net_snr[mode] = np.abs(np.inner(z_array[dominant_mode], z_array[mode].conjugate()))/rss_snr[dominant_mode]
        net_snr_perp[mode] = np.abs(np.inner(z_perp_array[dominant_mode], z_perp_array[mode].conjugate()))/rss_snr_perp[dominant_mode]

    return rss_snr, rss_snr_perp, net_snr, net_snr_perp

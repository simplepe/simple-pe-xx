import numpy as np
from pycbc.filter.matchedfilter import matched_filter
from simple_pe.param_est import metric, pe
from scipy import optimize
from pesummary.utils.samples_dict import SamplesDict


def matched_filter_network(ifos, data, psds, t_start, t_end, h, f_low, return_times=False):
    """
    Find the maximum SNR in the network for a waveform h within a given time range

    :param ifos: list of ifos
    :param data: a dictionary containing data from the ifos
    :param psds: a dictionary containing psds from the given ifos
    :param t_start: start time to consider SNR peak
    :param t_end: end time to consider SNR peak
    :param h: waveform
    :param f_low: low frequency cutoff
    :param return_times: return the time of the peak in each ifo
    :return snr: the network snr
    :return smax: the max snr in each ifo
    :return tmax: the time of max snr in each ifo
     """
    snrsq = 0
    smax = {}
    tmax = {}
    for ifo in ifos:
        snr = matched_filter(h, data[ifo], psd=psds[ifo], low_frequency_cutoff=f_low,
                             high_frequency_cutoff=psds[ifo].sample_frequencies[-1])
        snr_cut = snr.crop(t_start - snr.start_time, snr.end_time - t_end)
        smax[ifo] = snr_cut.abs_max_loc()[0]
        tmax[ifo] = snr_cut.sample_times[snr_cut.abs_arg_max()]
        snrsq += smax[ifo] ** 2

    if return_times:
        return np.sqrt(snrsq), smax, tmax
    else:
        return np.sqrt(snrsq), smax


def _neg_net_snr(x, x_directions, ifos, data, psds, t_start, t_end, f_low, approximant, fixed_pars=None):
    """
    Calculate the negative waveform match, taking x as the values and
    dx as the parameters.  This is in a format that's appropriate
    for scipy.optimize

    :param x: list of values
    :param x_directions: list of parameters for which to calculate waveform variations
    :param ifos: list of ifos to use
    :param data: a dictionary of data from the given ifos
    :param psds: a dictionary of power spectra for the ifos
    :param t_start: start time to consider SNR peak
    :param t_end: end time to consider SNR peak
    :param f_low: low frequency cutoff
    :param approximant: the approximant to use
    :param fixed_pars: a dictionary of fixed parameters and values
    :return -snr: the negative of the match at this point
    """
    s = dict(zip(x_directions, x))
    if fixed_pars is not None:
        s.update(fixed_pars)

    try:
        h = metric.make_waveform(
            s, psds[ifos[0]].delta_f, f_low, len(psds[ifos[0]]), approximant
        )
    except RuntimeError:
        return np.inf

    s = matched_filter_network(ifos, data, psds, t_start, t_end, h, f_low)[0]

    return -s


def find_peak_snr(ifos, data, psds, t_start, t_end, x, dx_directions,
                  f_low, approximant="IMRPhenomD", method='metric', initial_mismatch=0.03, final_mismatch=0.001,
                  tolerance=0.01):
    """
    A function to find the maximum SNR.
    Either calculate a metric at the point x in dx_directions and walk to peak, or use
    scipy optimization regime

    :param ifos: list of ifos to use
    :param data: a dictionary of data from the given ifos
    :param psds: a dictionary of power spectra for the ifos
    :param t_start: start time to consider SNR peak
    :param t_end: end time to consider SNR peak
    :param x: dictionary with parameter values for initial point
    :param dx_directions: list of parameters for which to calculate waveform variations
    :param f_low: low frequency cutoff
    :param approximant: the approximant to use
    :param method: how to find the maximum
    :param initial_mismatch: the mismatch for calculating the metric
    :param final_mismatch: the mismatch required to stop iteration
    :param tolerance: the allowed error in the metric is (tolerance * mismatch)
    :return x_prime: the point in the grid with the highest snr
    :return snr_peak: the SNR squared at this point
    """

    snr_peak = 0

    if (method != 'metric') and (method != 'scipy'):
        print('Have only implemented metric and scipy optimize based methods')
        return
    
    elif method == 'scipy':
        bounds = [(metric.param_mins[k], metric.param_maxs[k]) for k in dx_directions]
        x0 = np.array([x[k] for k in dx_directions]).flatten()
        fixed_pars = {k: float(v) for k, v in x.items() if k not in dx_directions}

        # generate bounds on spins:
        chia = "chi_eff" if "chi_eff" in x.keys() else "chi_align"
        chip = "chi_p2" if "chi_p2" in x.keys() else "chi_p"
        if chip == "chi_p2":
            n = 1
        else:
            n = 2

        if (chia in x) and (chip in x) and ((chia in dx_directions) or (chip in dx_directions)):
            # need bounds based on spin limits
            if (chia in dx_directions) and (chip in dx_directions):
                con = lambda y: y[dx_directions.index(chia)]**2 + y[dx_directions.index(chip)]**n
            elif chia in dx_directions:
                con = lambda y: y[dx_directions.index(chia)]**2 + x[chip]**n
            else:
                con = lambda y: x[chia]**2 + y[dx_directions.index(chip)]**n

            nlc = optimize.NonlinearConstraint(con, pe.param_mins['a_1'], pe.param_maxs['a_1'])

            out = optimize.minimize(_neg_net_snr, x0,
                                    args=(
                                    dx_directions, ifos, data, psds, t_start, t_end, f_low, approximant, fixed_pars),
                                    bounds=bounds,
                                    constraints=nlc)

        else:
            out = optimize.minimize(_neg_net_snr, x0,
                                    args=(
                                    dx_directions, ifos, data, psds, t_start, t_end, f_low, approximant, fixed_pars),
                                    bounds=bounds)

        x = {}
        for dx, val in zip(dx_directions, out.x):
            x[dx] = val
        x.update(fixed_pars)

        snr_peak = -out.fun

    elif method == 'metric':
        mismatch = initial_mismatch

        while mismatch > final_mismatch:
            x, snr_peak = _metric_find_peak(ifos, data, psds, t_start, t_end, x, dx_directions,
                                            f_low, approximant, mismatch, tolerance)
            print("Found peak, reducing mismatch to refine")
            mismatch /= 4

    return x, snr_peak


def _metric_find_peak(ifos, data, psds, t_start, t_end, x, dx_directions, f_low, approximant,
                      mismatch, tolerance=0.01):
    """
    A function to find the maximum SNR for a given metric mismatch
    Calculate a metric at the point x in dx_directions and walk to peak

    :param ifos: list of ifos to use
    :param data: a dictionary of data from the given ifos
    :param psds: a dictionary of power spectra for the ifos
    :param t_start: start time to consider SNR peak
    :param t_end: end time to consider SNR peak
    :param x: dictionary with parameter values for initial point
    :param dx_directions: list of parameters for which to calculate waveform variations
    :param f_low: low frequency cutoff
    :param approximant: the approximant to use
    :param mismatch: the mismatch for calculating the metric
    :param tolerance: the allowed error in the metric is (tolerance * mismatch)
    :return x_prime: the point in the grid with the highest snr
    :return snr_peak: the SNR squared at this point
    """
    psd_harm = len(ifos) / sum([1. / psds[ifo] for ifo in ifos])
    g = metric.Metric(x, dx_directions, mismatch, f_low, psd_harm, approximant, tolerance)
    g.iteratively_update_metric()

    while True:
        h = metric.make_waveform(x, g.psd.delta_f, g.f_low, len(g.psd), g.approximant)
        snr_0 = matched_filter_network(ifos, data, psds, t_start, t_end, h, f_low)[0]

        snrs = np.zeros([g.ndim, 2])
        alphas = np.zeros([g.ndim, 2])

        for i in range(g.ndim):
            for j in range(2):
                alphas[i, j] = metric.check_physical(x, g.normalized_evecs()[i:i + 1], (-1) ** j)
                h = metric.make_offset_waveform(x, g.normalized_evecs()[i:i + 1], alphas[i, j] * (-1) ** j,
                                                g.psd.delta_f, g.f_low, len(g.psd),
                                                g.approximant)
                snrs[i, j] = matched_filter_network(ifos, data, psds, t_start, t_end, h, f_low)[0]

        if snrs.max() > snr_0:
            # maximum isn't at the centre so update location
            i, j = np.unravel_index(np.argmax(snrs), snrs.shape)
            for k, dx_val in g.normalized_evecs()[i:i + 1].items():
                x[k] += float(alphas[i, j] * (-1) ** j * dx_val)

        else:
            # maximum is at the centre
            break

    s = (snrs[:, 0] - snrs[:, 1]) * 0.25 / \
        (snr_0 - 0.5 * (snrs[:, 0] + snrs[:, 1]))
    delta_x = SamplesDict(dx_directions, np.matmul(g.normalized_evecs().samples, s))
    alpha = metric.check_physical(x, delta_x, 1)

    h = metric.make_offset_waveform(x, delta_x, alpha, g.psd.delta_f, f_low, len(g.psd), approximant)
    snr_peak = matched_filter_network(ifos, data, psds, t_start, t_end, h, f_low)[0]

    for k, dx_val in delta_x.items():
        x[k] += float(alpha * dx_val)

    return x, snr_peak

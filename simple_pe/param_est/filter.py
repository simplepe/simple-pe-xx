import numpy as np
from pycbc.filter.matchedfilter import matched_filter
from simple_pe.param_est import metric
from scipy import optimize
from pesummary.utils.samples_dict import SamplesDict


def matched_filter_network(ifos, data, psds, t_start, t_end, h, f_low):
    """
    Find the maximum SNR in the network for a waveform h within a given time range

    :param ifos: list of ifos
    :param data: a dictionary containing data from the ifos
    :param psds: a dictionary containing psds from the given ifos
    :param t_start: start time to consider SNR peak
    :param t_end: end time to consider SNR peak
    :param h: waveform
    :param f_low: low frequency cutoff
    :return snr: the network snr
    :return smax: the max snr in each ifo
     """
    snrsq = 0
    smax = {}
    for ifo in ifos:
        snr = matched_filter(h, data[ifo], psd=psds[ifo], low_frequency_cutoff=f_low,
                             high_frequency_cutoff=psds[ifo].sample_frequencies[-1])
        smax[ifo] = max(abs(snr[(snr.sample_times > t_start) &
                                (snr.sample_times < t_end)]))

        snrsq += smax[ifo] ** 2

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

    h = metric.make_waveform(s, psds[ifos[0]].delta_f, f_low, len(psds[ifos[0]]), approximant)

    s = matched_filter_network(ifos, data, psds, t_start, t_end, h, f_low)[0]

    return -s


def find_peak_snr(ifos, data, psds, t_start, t_end, x, dx_directions,
                  f_low, approximant="IMRPhenomD", method='metric', mismatch=0.03, tolerance=0.01):
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
    :param mismatch: the mismatch for calculating the metric
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
        x0 = np.array([x[k] for k in dx_directions])
        fixed_pars = {k: v for k, v in x.items() if k not in dx_directions}

        out = optimize.minimize(_neg_net_snr, x0,
                                args=(dx_directions, ifos, data, psds, t_start, t_end, f_low, approximant, fixed_pars),
                                bounds=bounds)

        x = {}
        for dx, val in zip(dx_directions, out.x):
            x[dx] = np.array([val])
        x.update(fixed_pars)

        snr_peak = -out.fun

    elif method == 'metric':
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

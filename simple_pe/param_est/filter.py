import numpy as np
from simple_pe.param_est import metric, pe
from simple_pe.waveforms import waveform_modes
from scipy import optimize
from pesummary.utils.samples_dict import SamplesDict
import copy


def matched_filter_network(ifos, data, psds, t_start, t_end, h, f_low, dominant_mode=0):
    """
    Find the maximum SNR in the network for a waveform h within a given time range

    :param ifos: list of ifos
    :param data: a dictionary containing data from the ifos
    :param psds: a dictionary containing psds from the given ifos
    :param t_start: start time to consider SNR peak
    :param t_end: end time to consider SNR peak
    :param h: waveform (either a time series or dictionary of time series)
    :param f_low: low frequency cutoff
    :param dominant_mode: the dominant waveform mode (if a dictionary was passed)
    :return snr: the network snr
    :return smax: the max snr in each ifo
    """
    if not isinstance(h, dict):
        h = {0: h}
    modes = list(h.keys())
    snrsq = 0
    smax = {}
    for ifo in ifos:   
        h_perp, _, _ = waveform_modes.orthonormalize_modes(h, psds[ifo], f_low, modes, dominant_mode)
        z_dict = waveform_modes.calculate_mode_snr(data[ifo], psds[ifo], h_perp, t_start, t_end, f_low,
                                                   h.keys(), dominant_mode)
        smax[ifo] = np.linalg.norm(np.array(list(z_dict.values())))
        snrsq += smax[ifo] ** 2
    return np.sqrt(snrsq), smax


def _neg_net_snr(x, dx_directions, ifos, data, psds, t_start, t_end, f_low, approximant, fixed_pars=None,
                 harm2=False, verbose=False):
    """
    Calculate the negative waveform match, taking x as the values and
    dx as the parameters.  This is in a format that's appropriate
    for scipy.optimize

    :param x: list of values
    :param dx_directions: list of parameters for which to calculate waveform variations
    :param ifos: list of ifos to use
    :param data: a dictionary of data from the given ifos
    :param psds: a dictionary of power spectra for the ifos
    :param t_start: start time to consider SNR peak
    :param t_end: end time to consider SNR peak
    :param f_low: low frequency cutoff
    :param approximant: the approximant to use
    :param fixed_pars: a dictionary of fixed parameters and values
    :param harm2: if True then generate two harmonics and filter both
    :param verbose: if True then print info
    :return -snr: the negative of the match at this point
    """
    s = dict(zip(dx_directions, x))
    if fixed_pars is not None:
        s.update(fixed_pars)

    if verbose:
        print('making waveform at parameters')
        print(s)
    try:
        h = metric.make_waveform(
            s, psds[ifos[0]].delta_f, f_low, len(psds[ifos[0]]), approximant,
            harm2=harm2
        )
    except RuntimeError:
        print('error making waveform')
        return np.inf

    s = matched_filter_network(ifos, data, psds, t_start, t_end, h, f_low)[0]

    if verbose:
        print('snr = %.4f' % s)

    return -s


def find_peak_snr(ifos, data, psds, t_start, t_end, x, dx_directions,
                  f_low, approximant="IMRPhenomD", method='scipy', harm2=False,
                  initial_mismatch=0.03, final_mismatch=0.001, tolerance=0.01, verbose=False):
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
    :param method: how to find the maximum (either 'scipy' or 'metric')
    :param harm2: use SNR from second harmonic (only with method='scipy')
    :param initial_mismatch: the mismatch for calculating the metric
    :param final_mismatch: the mismatch required to stop iteration
    :param tolerance: the allowed error in the metric is (tolerance * mismatch)
    :param verbose: if True then print info
    :return x_prime: the point in the grid with the highest snr
    :return snr_peak: the SNR squared at this point
    """
    snr_peak = 0

    if (method != 'metric') and (method != 'scipy'):
        print('Have only implemented metric and scipy optimize based methods')
        return
    
    elif method == 'scipy':
        mins = copy.deepcopy(metric.param_mins)
        maxs = copy.deepcopy(metric.param_maxs)

        # generate bounds on spins:
        chia = "chi_eff" if "chi_eff" in x.keys() else "chi_align"
        chip = "chi_p2" if "chi_p2" in x.keys() else "chi_p"
        if chip == "chi_p2":
            n = 1
        else:
            n = 2

        nlc = None
        if (chia in x) and (chip in x) and ((chia in dx_directions) or (chip in dx_directions)):
            # need bounds based on spin limits
            if (chia in dx_directions) and (chip in dx_directions):
                con = lambda y: y[dx_directions.index(chia)] ** 2 + y[dx_directions.index(chip)] ** n
                nlc = optimize.NonlinearConstraint(con, pe.param_mins['a_1'], pe.param_maxs['a_1'])

            if chia in dx_directions:
                mins[chia] = - np.sqrt(mins[chia] ** 2 - x[chip] ** n)
                maxs[chia] = np.sqrt(mins[chia] ** 2 - x[chip] ** n)
            if chip in dx_directions:
                maxs[chip] = (maxs[chip]**n - x[chia]**2) ** (1/n)
                if harm2:
                    # need to have nonzero chi_p to generate 2 harmonics
                    mins[chip] = mins['prec'] ** n

        bounds = [(mins[k], maxs[k]) for k in dx_directions]
        x0 = np.array([x[k] for k in dx_directions]).flatten()
        fixed_pars = {k: float(v) for k, v in x.items() if k not in dx_directions}

        if nlc is not None:
            out = optimize.minimize(_neg_net_snr, x0,
                                    args=(dx_directions, ifos, data, psds, t_start, t_end,
                                          f_low, approximant, fixed_pars, harm2, verbose),
                                    bounds=bounds,
                                    constraints=nlc)

        else:
            out = optimize.minimize(_neg_net_snr, x0,
                                    args=(dx_directions, ifos, data, psds, t_start, t_end,
                                          f_low, approximant, fixed_pars, harm2, verbose),
                                    bounds=bounds)

        x = {}
        for dx, val in zip(dx_directions, out.x):
            x[dx] = val
        x.update(fixed_pars)

        snr_peak = -out.fun

    elif method == 'metric':
        if harm2:
            print('2nd harmonic not implemented for metric')
        mismatch = initial_mismatch

        while mismatch > final_mismatch:
            x, snr_peak = _metric_find_peak(ifos, data, psds, t_start, t_end, x, dx_directions,
                                            f_low, approximant, mismatch, tolerance)
            if verbose:
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

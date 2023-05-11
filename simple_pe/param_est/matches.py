import numpy as np
from pycbc.filter import match
from scipy import optimize
from simple_pe.waveforms import waveform
from simple_pe.param_est import metric, pe
from pesummary.utils.samples_dict import SamplesDict


def _neg_wf_match(x, x_directions, data, f_low, psd, approximant, fixed_pars=None):
    """
    Calculate the negative waveform match, taking x as the values and
    dx as the parameters.  This is in a format that's appropriate
    for scipy.optimize

    :param x: list of values
    :param x_directions: list of parameters for which to calculate waveform variations
    :param data: the data containing the waveform of interest
    :param f_low: low frequency cutoff
    :param psd: the power spectrum to use in calculating the match
    :param approximant: the approximant to use
    :param fixed_pars: a dictionary of fixed parameters and values
    :return -m: the negative of the match at this point
    """
    s = dict(zip(x_directions, x))
    if fixed_pars is not None:
        s.update(fixed_pars)

    h = waveform.make_waveform(s, psd.delta_f, f_low, len(psd), approximant)

    m = match(data, h, psd, low_frequency_cutoff=f_low, subsample_interpolation=True)[0]

    return -m


def _log_wf_mismatch(x, x_directions, data, f_low, psd, approximant, fixed_pars=None):
    """
    Calculate the negative waveform match, taking x as the values and
    dx as the parameters.  This is in a format that's appropriate
    for scipy.optimize

    :param x: list of values
    :param x_directions: list of parameters for which to calculate waveform variations
    :param data: the data containing the waveform of interest
    :param f_low: low frequency cutoff
    :param psd: the power spectrum to use in calculating the match
    :param approximant: the approximant to use
    :param fixed_pars: a dictionary of fixed parameters and values
    :return -m: the negative of the match at this point
    """
    s = dict(zip(x_directions, x))
    if fixed_pars is not None:
        s.update(fixed_pars)

    h = waveform.make_waveform(s, psd.delta_f, f_low, len(psd), approximant)

    m = match(data, h, psd, low_frequency_cutoff=f_low, subsample_interpolation=True)[0]

    return np.log10(1 - m)


def find_best_match(data, x, dx_directions, f_low, psd, approximant="IMRPhenomD",
                    method="metric", mismatch=0.03, tolerance=0.01):
    """
    A function to find the maximum match.
    This is done in two steps, first by finding the point in the grid defined
    by the metric gij (and given desired_mismatch) that gives the highest match.
    Second, we approximate the match as quadratic and find the maximum.

    :param data: the data containing the waveform of interest
    :param x: dictionary with parameter values for initial point
    :param dx_directions: list of parameters for which to calculate waveform variations
    :param f_low: low frequency cutoff
    :param psd: the power spectrum to use in calculating the match
    :param approximant: the approximant to use
    :param method: how to find the maximum
    :param mismatch: the mismatch for calculating the metric
    :param tolerance: the allowed error in the metric is (tolerance * mismatch)
    :return x_prime: the point in the grid with the highest match
    :return m_0: the match at this point
    """
    if (method != 'metric') and (method != 'scipy'):
        print('Have only implemented metric and scipy optimize based methods')
        m_peak = -1

    elif method == 'scipy':
        bounds = waveform.param_bounds(x, dx_directions, harm2=False)
        x0 = np.array([x[k] for k in dx_directions])
        fixed_pars = {k: v for k, v in x.items() if k not in dx_directions}

        out = optimize.minimize(_log_wf_mismatch, x0,
                                args=(dx_directions, data, f_low, psd, approximant, fixed_pars),
                                bounds=bounds)

        x = dict(zip(dx_directions, out.x))
        m_peak = 1 - 10 ** out.fun

    elif method == 'metric':
        g = metric.Metric(x, dx_directions, mismatch, f_low, psd, approximant, tolerance)
        g.iteratively_update_metric()

        while True:
            h = waveform.make_waveform(x, g.psd.delta_f, g.f_low, len(g.psd), g.approximant)

            m_0, _ = match(data, h, g.psd, low_frequency_cutoff=g.f_low, subsample_interpolation=True)
            matches = np.zeros([g.ndim, 2])
            alphas = np.zeros([g.ndim, 2])

            for i in range(g.ndim):
                for j in range(2):
                    alphas[i, j] = waveform.check_physical(x, g.normalized_evecs()[i:i + 1], (-1) ** j)
                    h = waveform.make_offset_waveform(x, g.normalized_evecs()[i:i + 1], alphas[i, j] * (-1) ** j,
                                             g.psd.delta_f, g.f_low, len(g.psd),
                                             g.approximant)
                    matches[i, j] = match(data, h, g.psd, low_frequency_cutoff=g.f_low,
                                          subsample_interpolation=True)[0]

            if matches.max() > m_0:
                # maximum isn't at the centre so update location
                i, j = np.unravel_index(np.argmax(matches), matches.shape)
                for k, dx_val in g.normalized_evecs()[i:i + 1].items():
                    x[k] += float(alphas[i, j] * (-1) ** j * dx_val)

            else:
                # maximum is at the centre
                break

        s = (matches[:, 0] - matches[:, 1]) * 0.25 / \
            (m_0 - 0.5 * (matches[:, 0] + matches[:, 1]))
        delta_x = pe.SimplePESamples(SamplesDict(dx_directions, np.matmul(g.normalized_evecs().samples, s)))
        alpha = waveform.check_physical(x, delta_x, 1)

        h = waveform.make_offset_waveform(x, delta_x, alpha, psd.delta_f, f_low, len(psd), approximant)
        m_peak = match(data, h, psd, low_frequency_cutoff=f_low,
                       subsample_interpolation=True)[0]

        for k, dx_val in delta_x.items():
            x[k] += float(alpha * dx_val)

    return x, m_peak

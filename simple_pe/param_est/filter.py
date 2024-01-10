import numpy as np
from simple_pe.param_est import metric, pe
from simple_pe.waveforms import waveform_modes, waveform, parameter_bounds
from scipy import optimize
from pesummary.utils.samples_dict import SamplesDict


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
    :return tmax: return the time of max snr in each ifo
    """
    if not isinstance(h, dict):
        h = {0: h}
    modes = list(h.keys())
    snrsq = 0
    smax = {}
    tmax = {}
    for ifo in ifos:   
        h_perp, _, _ = waveform_modes.orthonormalize_modes(h, psds[ifo], f_low, modes, dominant_mode)
        z_dict, tmax[ifo] = waveform_modes.calculate_mode_snr(data[ifo], psds[ifo], h_perp, t_start, t_end, f_low,
                                                   h.keys(), dominant_mode)
        if len(modes) > 1:
            # return the RSS SNR
            smax[ifo] = np.linalg.norm(np.array(list(z_dict.values())))
        else:
            # return the complex SNR for the 1 mode
            smax[ifo] = z_dict[modes[0]]

        snrsq += np.abs(smax[ifo]) ** 2
    return np.sqrt(snrsq), smax, tmax


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
        h = waveform.make_waveform(
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


def filter_grid_components(event_info, grid_data, strain_f, f_low, f_high, delta_f, psds, approximant, t_start, t_end, ifos, prec_snr=True, hm_snr=True):
    # filter over a grid in mc and eta space, keeping other parameters given in
    # event_info fixed can filter higher harmonics and precession as well
    import tqdm
    import copy
    from pesummary.gw.conversions.snr import _calculate_precessing_harmonics
    from simple_pe.waveforms import prec_33
    snrs = {}
    overlaps = {}

    dirs = grid_data.keys()
    data_size = grid_data[list(dirs)[0]].shape

    snrs = np.zeros(data_size)

    if prec_snr:
        prec_modes = [0, 1]
    else:
        prec_modes = [0]

    for ind in tqdm.tqdm(range(len(snrs.flatten())) ):
        i = np.unravel_index(ind, data_size)
        ei = copy.deepcopy(event_info)
        for d in dirs:
            ei[d] = grid_data[d][i]
        ei = pe.SimplePESamples(ei)
        ei.add_fixed('phase', 0.)
        ei.add_fixed('f_ref', f_low)
        ei.add_fixed('theta_jn', 0.5)
        ei.generate_prec_spin()
        ei.generate_all_posterior_samples(
            f_low=f_low, f_ref=f_low, delta_f=delta_f, disable_remnant=True
        )

        hp = _calculate_precessing_harmonics(
            ei["mass_1"][0], ei["mass_2"][0], ei["a_1"][0], ei["a_2"][0],
            ei["tilt_1"][0], ei["tilt_2"][0], ei["phi_12"][0], ei["beta"][0],
            ei["distance"][0], harmonics=prec_modes, approx=approximant,
            mode_array=[[2,2]], df=delta_f, f_low=f_low, f_final=f_high
        )
        snrs[i], _, _ = matched_filter_network(ifos, strain_f, psds, t_start, t_end, hp, f_low) 
        #h_perp, sigma, zeta = waveform_modes.orthonormalize_modes(
        #    hp, psd, f_low, prec_modes, dominant_mode=0
        #)
        #z = waveform_modes.calculate_mode_snr(
        #    strain_f, psd, h_perp, t_start, t_end, f_low,
        #    prec_modes, dominant_mode=0
        #)[0]
        #snrs[dom][i] = np.abs(z[0])

        #if prec_snr:
        #    overlaps[prec][i] = zeta[1]
        #    snrs[prec][i] = np.abs(z[1])

        #if hm_snr:
        #    h33 = prec_33.calculate_precessing_harmonics(
        #        ei["mass_1"][0], ei["mass_2"][0], ei["a_1"][0], ei["a_2"][0],
        #        ei["tilt_1"][0], ei["tilt_2"][0], ei["phi_12"][0],
        #        ei["beta"][0], ei["distance"][0], harmonics=[0],
        #        approx=approximant, df=delta_f, f_low=f_low,
        #        f_ref=f_low, f_final=f_high
        #    )

        #    h_hm = {dom: h_perp[0], hm: h33[0]}
        #    z = waveform_modes.calculate_mode_snr(
        #        strain_f, psd, h_hm, t_start, t_end, f_low,
        #        h_hm.keys(), dom
        #    )[0]
        #    snrs[hm][i] = np.abs(z[hm])

    #snrsq_tot = None
    #for k, s in snrs.items():
    #    if snrsq_tot is None:
    #        snrsq_tot = np.zeros_like(s)

    #    snrsq_tot += s**2

    #snrs['Total'] = snrsq_tot**0.5

    return snrs


def find_peak_snr(ifos, data, psds, t_start, t_end, x, dx_directions,
                  f_low, approximant="IMRPhenomD", method='scipy', harm2=False, bounds=None,
                  initial_mismatch=0.03, final_mismatch=0.001, tolerance=0.01, verbose=False,
                  _net_snr=None):
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
    :param harm2: use SNR from second harmonic
    :param bounds: give initial bounds for the range of parameters to investigate
    :param initial_mismatch: the mismatch for calculating the metric
    :param final_mismatch: the mismatch required to stop iteration
    :param tolerance: the allowed error in the metric is (tolerance * mismatch)
    :param verbose: if True then print info
    :return x_prime: the point in the grid with the highest snr
    :return snr_peak: the SNR squared at this point
    """
    snr_peak = 0

    if method not in ["metric", "scipy", "grid"]:
        print('Have only implemented metric, scipy and grid optimize based methods')
        return
    
    elif method == "grid":
        g_ms = metric.find_metric_and_eigendirections(
            x, dx_directions, _net_snr, f_low, psds["L1"], approximant
        )
        scale=1.5
        npts = 11
        x_val = np.linspace(-scale, scale, npts)
        grid = np.meshgrid(x_val, x_val, x_val)
        n_evec = g_ms.normalized_evecs()
        grid_data = {}
        dx_data = np.tensordot(n_evec.samples, np.asarray(grid), axes=(1, 0))
        for i,d in enumerate(dx_directions):
            grid_data[d] = dx_data[i] + g_ms.x[d]
        snrs = filter_grid_components(
            x, grid_data, data, f_low, psds["L1"].sample_frequencies[-1], psds["L1"].delta_f, psds, approximant, t_start, t_end, ifos, prec_snr=True, hm_snr=True
        )
        #s = 'Total'
        amax = np.unravel_index(np.argmax(snrs), snrs.shape)
        snr_peak = snrs[amax]
        x = {}
        for k, i in grid_data.items():
            x[k] = i[amax]
        fixed_pars = {k: float(v) for k, v in x.items() if k not in dx_directions}
        x.update(fixed_pars)

    elif method == 'scipy':

        nlc = None
        if bounds is None:
            bounds = parameter_bounds.param_bounds(x, dx_directions, harm2)

            # generate constraint on spins:
            chia = "chi_eff" if "chi_eff" in x.keys() else "chi_align"
            chip = "chi_p2" if "chi_p2" in x.keys() else "chi_p"
            if chip == "chi_p2":
                n = 1
            else:
                n = 2

            if (chia in x) and (chip in x) and ((chia in dx_directions) or (chip in dx_directions)):
                # need bounds based on spin limits
                if (chia in dx_directions) and (chip in dx_directions):
                    con = lambda y: y[dx_directions.index(chia)] ** 2 + y[dx_directions.index(chip)] ** n
                    nlc = optimize.NonlinearConstraint(con, pe.param_mins['a_1'], pe.param_maxs['a_1'])

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
        mismatch = initial_mismatch
        x = pe.SimplePESamples(x)

        while mismatch > final_mismatch:
            x, snr_peak = _metric_find_peak(ifos, data, psds, t_start, t_end, x, dx_directions,
                                            f_low, approximant, mismatch, tolerance)
            if verbose:
                print("Found peak, reducing mismatch to refine")
            mismatch /= 4

    return x, snr_peak


def _metric_find_peak(ifos, data, psds, t_start, t_end, x, dx_directions, f_low, approximant,
                      mismatch, tolerance=0.01, harm2=False, verbose=False):
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
    :param harm2: use SNR from second harmonic
    :param verbose: if True then print info
    :return x_prime: the point in the grid with the highest snr
    :return snr_peak: the SNR squared at this point
    """
    psd_harm = len(ifos) / sum([1. / psds[ifo] for ifo in ifos])
    g = metric.Metric(x, dx_directions, mismatch, f_low, psd_harm, approximant, tolerance)
    g.iteratively_update_metric()

    while True:
        h = waveform.make_waveform(x, g.psd.delta_f, g.f_low, len(g.psd), g.approximant, harm2=harm2)
        snr_0 = matched_filter_network(ifos, data, psds, t_start, t_end, h, f_low)[0]

        snrs = np.zeros([g.ndim, 2])
        alphas = np.zeros([g.ndim, 2])

        for i in range(g.ndim):
            for j in range(2):
                alphas[i, j] = metric.check_physical(x, g.normalized_evecs()[i:i + 1], (-1) ** j, verbose=verbose)
                h = metric.make_offset_waveform(x, g.normalized_evecs()[i:i + 1], alphas[i, j] * (-1) ** j,
                                                g.psd.delta_f, g.f_low, len(g.psd),
                                                g.approximant, harm2=harm2)
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
    alpha = metric.check_physical(x, delta_x, 1, verbose=verbose)

    h = metric.make_offset_waveform(x, delta_x, alpha, g.psd.delta_f, f_low, len(g.psd), approximant, harm2=harm2)
    snr_peak = matched_filter_network(ifos, data, psds, t_start, t_end, h, f_low)[0]

    for k, dx_val in delta_x.items():
        x[k] += float(alpha * dx_val)

    return x, snr_peak

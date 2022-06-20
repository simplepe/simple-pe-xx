import numpy as np
from pycbc.waveform import get_fd_waveform
from pycbc.filter import match
from pycbc import conversions
from pycbc.filter.matchedfilter import matched_filter
from simple_pe.waveforms import waveform_modes
import copy
from scipy import optimize
from scipy.stats import chi2
from pesummary.gw.conversions import convert
from pesummary.utils.samples_dict import SamplesDict


def find_peak(data, xx, gij, basis, desired_mismatch, dist, f_low, psd,
          mass="chirp", approximant="IMRPhenomD", verbose=False):
    """
    A function to find the maximum match.
    This is done in two steps, first by finding the point in the grid defined
    by the metric gij (and given desired_mismatch) that gives the highest match.
    Second, we approximate the match as quadratic and find the maximum.

    :param data: the data containing the waveform of interest
    :param xx: the point in parameter space used to calculate the metric
    :param gij: the parameter space metric
    :param basis: the basis relating the directions of the metric to the physical space
    :param desired_mismatch: the desired mismatch
    :param dist: distance to the signal
    :param f_low: low frequency cutoff
    :param psd: the power spectrum to use in calculating the match
    :param mass: either use chirp or total mass
    :param approximant: the approximant generator to use
    :param verbose: print some debugging information
    :return x_prime: the point in the grid with the highest match
    :return m_0: the match at this point
    :return steps: the number of steps taken in each eigen-direction
    """

    x = copy.deepcopy(xx)
    evecs = calculate_evecs(gij, desired_mismatch)
    ndim = len(evecs)
    v_phys = np.matmul(basis, evecs)
    steps = np.zeros(ndim)

    while True:
        h = make_waveform(x, np.zeros_like(x), dist, psd.delta_f, f_low, len(psd), mass, approximant)
        m_0, _ = match(data, h, psd, low_frequency_cutoff=f_low)
        matches = np.zeros([ndim, 2])
        alphas = np.zeros([ndim, 2])

        for i in range(ndim):
            for j in range(2):
                dx = (-1) ** j * v_phys[i]
                alphas[i, j] = check_physical(x, dx)
                h = make_waveform(x, alphas[i, j] * dx, dist, psd.delta_f, f_low, len(psd),
                                  mass, approximant)
                matches[i, j], _ = match(data, h, psd,
                                         low_frequency_cutoff=f_low)

        if verbose:
            print("Central match %.3f; maximum offset match %.3f" %
                  (m_0, matches.max()))

        if matches.max() > m_0:
            # maximum isn't at the centre so update location
            i, j = np.unravel_index(np.argmax(matches), matches.shape)
            x += alphas[i, j] * (-1) ** j * v_phys[i]
            steps[i] += (-1) ** j
            if verbose:
                print("Moving in the %d eigendirection, %.2f units" %
                      (i, alphas[i, j] * (-1) ** j))
                print("New position"),
                print(x)
                print("")
        else:
            if verbose:
                print("Maximum at the centre, stopping")
            break

    s = (matches[:, 0] - matches[:, 1]) * 0.25 / \
        (m_0 - 0.5 * (matches[:, 0] + matches[:, 1]))
    steps += s
    delta_x = np.matmul(v_phys, s)
    alpha = check_physical(x, delta_x)
    delta_x *= alpha

    if verbose:
        print("Using matches to find peak")
        print("Moving in the eigendirections distance of"),
        print("%.2g" % alpha * s)
        print("New position"),
        print(x + delta_x)

    h = make_waveform(x, delta_x, dist, psd.delta_f, f_low, len(psd), mass, approximant)
    m_peak = match(data, h, psd, low_frequency_cutoff=f_low)[0]
    x_peak = x + delta_x

    return x_peak, m_peak, steps


def find_peak_snr(data, psd, ifos, t_start, t_end, xx, gij, basis, desired_mismatch, dist, f_low, f_high,
                  mass="chirp", approximant="IMRPhenomD", verbose=False):
    """
    A function to find the maximum SNR.
    Start at the point with parameters xx and use the metric gij to calculate eigen-directions.
    Calculate the SNR along each of the eigen-directions at a given step size (as defined by the
    initial metric).  If any are higher, then move.  If not, then reduce step size.
    Repeat until step size reaches requested mismatch.

    :param data: the data containing the waveform of interest
    :param psd: the power spectrum to use in calculating the match
    :param ifos: list of ifos to use
    :param t_start: start time to consider SNR peak
    :param t_end: end time to consider SNR peak
    :param xx: the point in parameter space used to calculate the metric
    :param gij: the parameter space metric
    :param basis: the basis relating the directions of the metric to the physical space
    :param desired_mismatch: the desired mismatch
    :param dist: distance to the signal
    :param f_low: low frequency cutoff
    :param f_high: high frequency cutoff
    :param mass: either use chirp or total mass
    :param approximant: the approximant generator to use
    :param verbose: print debugging information (if True)
    :return x_prime: the point in the grid with the highest snr
    :return snrsq_peak: the match at this point
    :return steps: the number of steps taken in each eigendirection
    """

    x = copy.deepcopy(xx)
    evecs = calculate_evecs(gij, desired_mismatch)
    ndim = len(evecs)
    v_phys = np.matmul(evecs, basis)
    steps = np.zeros(ndim)
    df = psd[ifos[0]].delta_f

    flen = int(len(data[ifos[0]]) / 2 + 1)

    while True:
        h = make_waveform(x, np.zeros_like(x), dist, df, f_low, flen, mass, approximant)
        snrsq_0 = 0
        for ifo in ifos:
            snr = matched_filter(h, data[ifo], psd=psd[ifo], low_frequency_cutoff=f_low,
                                 high_frequency_cutoff=f_high)
            smax = max(abs(snr[(snr.sample_times > t_start) &
                               (snr.sample_times < t_end)]))
            snrsq_0 += smax ** 2

        snrsqs = np.zeros([ndim, 2])
        alphas = np.zeros([ndim, 2])

        for i in range(ndim):
            for j in range(2):
                dx = (-1) ** j * v_phys[i]
                alphas[i, j] = check_physical(x, dx)
                h = make_waveform(x, alphas[i, j] * dx, dist, df, f_low, flen, mass,
                                  approximant)
                for ifo in ifos:
                    snr = matched_filter(h, data[ifo], psd=psd[ifo], low_frequency_cutoff=f_low,
                                         high_frequency_cutoff=f_high)
                    snrsqs[i, j] += max(abs(snr[(snr.sample_times > t_start) &
                                                (snr.sample_times < t_end)])) ** 2

        if verbose:
            print("Central snr %.3f; maximum offset snr %.3f" %
                  (np.sqrt(snrsq_0), np.sqrt(snrsqs.max())))

        if snrsqs.max() > snrsq_0:
            # maximum isn't at the centre so update location
            i, j = np.unravel_index(np.argmax(snrsqs), snrsqs.shape)
            x += alphas[i, j] * (-1) ** j * v_phys[i]
            steps[i] += (-1) ** j
            if verbose:
                print("Moving in the %d eigendirection, %.2f units" %
                      (i, alphas[i, j] * (-1) ** j))
                print("New position"),
                print(x)
                print("")
        else:
            if verbose:
                print("Maximum at the centre, stopping")
            break

    # Use the calculated SNRs to find the approximate peak:
    snrs = np.sqrt(snrsqs)
    s = (snrs[:, 1] - snrs[:, 0]) * 0.25 / \
        (np.sqrt(snrsq_0) - 0.5 * (snrs[:, 0] + snrs[:, 1]))
    if max(abs(s)) > 1:
        print("Step is too large -- there's an issue")
        print(s)

    delta_x = np.matmul(s, v_phys)
    alpha = check_physical(x, delta_x)
    delta_x *= alpha

    if verbose:
        print("Using SNRS to find approximate peak")
        print("Moving in the eigendirections distance of"),
        print(alpha)
        print(s)

        # print("%.2g" % alpha * s)
        print("New position"),
        print(x + delta_x)

    h = make_waveform(x, delta_x, dist, df, f_low, flen, mass, approximant)
    snrsq_peak = 0
    for ifo in ifos:
        snr = matched_filter(h, data[ifo], psd=psd[ifo], low_frequency_cutoff=f_low,
                             high_frequency_cutoff=f_high)
        snrsq_peak += max(abs(snr[(snr.sample_times > t_start) &
                                  (snr.sample_times < t_end)])) ** 2

    if snrsq_peak > snrsq_0:
        if verbose:
            print("moving to inferred peak")
        x_peak = x + delta_x
        steps += s
    else:
        if verbose:
            print("original point is better")
        x_peak = x
        snrsq_peak = snrsq_0

    return x_peak, np.sqrt(snrsq_peak), steps


def maximize_network_snr(mc_initial, eta_initial, chi_eff_initial, ifos, strain, psd, t_start, t_end, f_low,
                         mass, approximant):
    """
    A function to find the maximum SNR in the network within a given time range

    :param mc_initial: starting position for chirp mass
    :param eta_initial: starting position for eta
    :param chi_eff_initial: starting position for chi_eff
    :param ifos: list of ifos
    :param strain: dictionary of strain data from the ifos
    :param psd: the power spectra for the ifos
    :param t_start: start time to consider SNR peak
    :param t_end: end time to consider SNR peak
    :param f_low: low frequency cutoff
    :param mass: either chirp or total
    :param approximant: the approximant to use
    :return x: the parameters of the peak
    :return snr: the network snr
    :return ifo_snr: the max snr in each ifo
    """

    x = np.asarray([mc_initial, eta_initial, chi_eff_initial])
    trigger_snr, ifo_snr = network_snr(x, strain, psd, t_start, t_end, f_low)

    # initial directions and initial spacing
    v0 = np.array([(1., 0., 0.),
                   (0., 0.01, 0.),
                   (0., 0., 0.1)])
    scale = 2.1  # appropriate for 3 dimensions
    dist = 1.0
    desired_mismatch = scale / trigger_snr ** 2
    min_mis = 1e-2 / trigger_snr ** 2
    tolerance = 0.05
    max_iter = 8

    hm_psd = len(ifos) / sum([1. / psd[ifo] for ifo in ifos])
    f_high = hm_psd.sample_frequencies[-1]

    while desired_mismatch > min_mis:
        basis = scale_vectors(x, v0, dist, desired_mismatch, f_low, hm_psd, mass, approximant)
        gij = calculate_metric(x, basis, dist, f_low, hm_psd, mass, approximant)
        gij, v, t = iteratively_update_metric(x, gij, basis, desired_mismatch, tolerance,
                                              dist, f_low, hm_psd, mass, approximant, max_iter=max_iter)
        x, snr_now, steps = find_peak_snr(strain, psd, ifos, t_start, t_end, x, gij, basis, desired_mismatch, dist,
                                          f_low, f_high, mass, approximant)

        if x[1] == 0.25:
            x -= np.dot([0., 0.001, 0.], basis)
            # break
        if np.linalg.norm(steps) < 1:
            # didn't move much so reduce step size
            desired_mismatch /= 4.

    h = make_waveform(x, np.zeros_like(x), dist, hm_psd.delta_f, f_low, len(hm_psd), mass,
                      approximant)
    snr, ifo_snr = matched_filter_network(strain, psd, ifos, t_start, t_end, h, f_low, f_high)

    return x, snr, ifo_snr


def matched_filter_network(data, psd, ifos, t_start, t_end, h, f_low, f_high):
    """
     A function to find the maximum SNR in the network within a given time range

     :param data: the data containing the waveform of interest
     :param psd: the power spectrum to use in calculating the match
     :param ifos: list of ifos
     :param t_start: start time to consider SNR peak
     :param t_end: end time to consider SNR peak
     :param h: waveform
     :param f_low: low frequency cutoff
     :param f_high: high frequency cutoff-
     :return snr: the network snr
     :return smax: the max snr in each ifo
     """
    snrsq = 0
    smax = {}
    for ifo in ifos:
        snr = matched_filter(h, data[ifo], psd=psd[ifo], low_frequency_cutoff=f_low,
                             high_frequency_cutoff=f_high)
        smax[ifo] = max(abs(snr[(snr.sample_times > t_start) &
                                (snr.sample_times < t_end)]))

        snrsq += smax[ifo] ** 2

    return np.sqrt(snrsq), smax


def network_snr(mc_eta_sz, data, psd, t_start, t_end, f_low, approximant="IMRPhenomD"):
    """
     A function to find the network SNR for a given chirp mass, eta, spin_z
     in the network within a given time range

     :param mc_eta_sz: vector containing (chirp mass, eta, spin_1z [optionally spin2z])
     :param data: the data containing the waveform of interest
     :param psd: the power spectrum to use in calculating the match
     :param t_start: start time to consider SNR peak
     :param t_end: end time to consider SNR peak
     :param f_low: low frequency cutoff
     :param approximant: the approximant to use
     :return snr: the network snr
     """

    mc = mc_eta_sz[0]
    eta = mc_eta_sz[1]
    s1z = mc_eta_sz[2]
    if len(mc_eta_sz) == 3:
        s2z = mc_eta_sz[2]
    else:
        s2z = mc_eta_sz[3]

    mass1 = conversions.mass1_from_mchirp_eta(mc, eta)
    mass2 = conversions.mass2_from_mchirp_eta(mc, eta)
    distance = 1.

    ifos = list(data.keys())

    f_high = psd[ifos[0]].sample_frequencies[-1]
    df = psd[ifos[0]].delta_f

    hp, hc = get_fd_waveform(approximant=approximant,
                             mass1=mass1, mass2=mass2, spin1z=s1z, spin2z=s2z,
                             distance=distance,
                             delta_f=df, f_lower=f_low,
                             f_final=f_high)

    return matched_filter_network(data, psd, ifos, t_start, t_end, hp, f_low, f_high)

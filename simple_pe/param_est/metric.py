import numpy as np
from pycbc.waveform import get_fd_waveform
from pycbc.filter import match
from pycbc import conversions
from pycbc.filter.matchedfilter import matched_filter
import copy
from scipy import optimize


def make_waveform(x, dx, dist, df, f_low, flen, waveform="IMRPhenomD"):
    """
    This function makes a waveform for the given parameters and
    returns h_plus generated at value (x + dx).

    :param x: np.array with four values assumed to be mchirp, eta, s1z, s2z
    :param dx: same as x.
    :param dist: distance to the signal
    :param df: frequency spacing of points
    :param f_low: low frequency cutoff
    :param flen: length of the frequency domain array to generate
    :param waveform: the waveform generator to use
    :return h_plus: waveform as a frequency series with the requested df, flen
    """

    mc = x[0] + dx[0]
    eta = x[1] + dx[1]
    m1 = conversions.mass1_from_mchirp_eta(mc, eta)
    m2 = conversions.mass2_from_mchirp_eta(mc, eta)
    h_plus, h_cross = get_fd_waveform(approximant=waveform, mass1=m1, mass2=m2, spin1z=x[2] + dx[2],
                                      spin2z=x[2] + dx[2], delta_f=df, distance=dist, f_lower=f_low)
    h_plus.resize(flen)
    return h_plus


def scale_vectors(x, vec, dist, mismatch, f_low, psd,
                  waveform="IMRPhenomD", tolerance=1e-2):
    """
    This function scales the input vectors so that the mismatch between
    a waveform at point x and one at x + v[i] is equal to the specified
    mismatch, up to the specified tolerance.

    :param x: np.array with four values assumed to be mchirp, eta, s1z, s2z
    :param vec: an array of directions dx in which to vary the waveform parameters
    :param dist: distance to the signal
    :param mismatch: the desired mismatch (1 - match)
    :param f_low: low frequency cutoff
    :param psd: the power spectrum to use in calculating the match
    :param waveform: the waveform generator to use
    :param tolerance: the maximum fractional error in the mismatch
    :return v: A set of vectors in the directions given by v but normalized to give the
    desired mismatch
    """
    ndim = len(vec)
    v = copy.deepcopy(vec)
    for i in range(ndim):
        opt = optimize.root_scalar(lambda a: average_mismatch(x, a * v[i], dist,
                                                              f_low, psd, waveform) - mismatch,
                                   bracket=[0, 20], method='brentq', rtol=tolerance)
        v[i] *= opt.root

    return v


def check_physical(x, dx, maxs=[1e4, 0.25, 0.98, 0.98],
                   mins=[1., 0.05, -0.98, -0.98]):
    """
    A function to check whether ther point described by the positions x + dx is
    physically permitted.  If not, rescale and return the scaling factor

    :param x : np.array with four values assumed to be mchirp, eta, s1z, s2z
    :param dx: same as x.
    :param maxs: the maximum permitted values of the physical parameters
    :param mins: the minimum physical values of the physical parameters
    :return alpha: the scaling factor required to make x + dx physically permissable
    """
    alpha = 1.
    for i in range(3):
        if (x + dx)[i] < mins[i]: alpha = min(alpha, (x[i] - mins[i]) / abs(dx[i]))
        if (x + dx)[i] > maxs[i]: alpha = min(alpha, (maxs[i] - x[i]) / abs(dx[i]))
    return alpha


def scale_match(m_alpha, alpha):
    """
    A function to scale the match calculated at an offset alpha to the
    match at unit offset

    :param m_alpha: the match at an offset alpha
    :param alpha: the value of alpha
    :return m : the match at unit offset
    """
    m = (alpha ** 2 - 1 + m_alpha) / alpha ** 2
    return m


def average_mismatch(x, dx, dist, f_low, psd,
                     waveform="IMRPhenomD", verbose=False):
    """
    This function calculated the average match for steps of +dx and -dx
    It also takes care of times when one of the steps moves beyond the
    edge of the physical parameter space

    :param x : np.array with four values assumed to be mchirp, eta, chi_eff
    :param dx: the change in the values x
    :param dist: distance to the signal
    :param f_low: low frequency cutoff
    :param flen: length of the frequency domain array to generate
    :param psd: the power spectrum to use in calculating the match
    :param waveform: the waveform generator to use
    :return m: The average match from steps of +/-dx
    """
    a = {}
    m = {}
    h0 = make_waveform(x, np.zeros_like(x), dist, psd.delta_f, f_low, len(psd), waveform)
    for s in [1., -1.]:
        a[s] = check_physical(x, s * dx)
        h = make_waveform(x, s * a[s] * dx, dist, psd.delta_f, f_low, len(psd), waveform)
        m[s], _ = match(h0, h, psd, low_frequency_cutoff=f_low)
    if verbose:
        print("Had to scale steps to %.2f, %.2f" % (a[-1], a[1]))
        print("Mismatches %.3f, %.3f" % (1 - m[-1], 1 - m[1]))
    if (min(a.values()) < 1e-2):
        if verbose:
            print("we're really close to the boundary, so down-weight match contribution")
        mm = (2 - m[1] - m[-1]) / (a[1] ** 2 + a[-1] ** 2)
    else:
        mm = 1 - 0.5 * (scale_match(m[1], a[1]) + scale_match(m[-1], a[-1]))
    return mm


def calculate_metric(x, vec, dist, f_low, psd, waveform="IMRPhenomD"):
    """
    A function to calculate the metric at a point x, associated to a given set
    of variations in the directions given by vec.
    :param x : np.array with three values assumed to be mchirp, eta, sz
    :param vec: an array of directions dx in which to vary the waveform parameters
    :param dist: distance to the signal
    :param f_low: low frequency cutoff
    :param psd: the power spectrum to use in calculating the match
    :param waveform: the waveform generator to use
    :return gij: a square matrix, with size given by the length of vec, that gives the
         metric at x along the directions given by vec
    """
    ndim = len(vec)
    gij = np.zeros([ndim, ndim])

    # diagonal components
    # g_ii = 1 - 0.5 [m(dx_i) + m(-dx_i)]
    for i in range(ndim):
        dx = vec[i]

        gij[i, i] += average_mismatch(x, dx, dist, f_low, psd, waveform)

    # off diagonal
    # g_ij = 0.25 * [- m(1/sqrt(2) (dx_i + dx_j)) - m(-1/sqrt(2) (dx_i + dx_j))
    #               + m(1/sqrt(2) (dx_i - dx_j)) - m(-1/sqrt(2) (dx_i - dx_j))]
    for i in range(ndim):
        for j in range(i + 1, ndim):
            for s in ([[1, 1], [1, -1]]):
                dx = (s[0] * vec[i] + s[1] * vec[j]) / np.sqrt(2)
                gij[i, j] += 0.5 * s[0] / s[1] * \
                             average_mismatch(x, dx, dist, f_low, psd, waveform)
            gij[j, i] = gij[i, j]

    return gij


def physical_metric(gij, basis):
    """
    A function to calculate the metric in physical coordinates

    :param gij: the metric calculated with respect to a set of basis vectors
    :param basis: the basis vectors expressed in terms of the physical coordinates
    :return gphys: the metric in physical coordinates
    """
    vnorm = np.linalg.norm(basis, axis=1)
    ghatij = gij / vnorm / vnorm.reshape((-1, 1))
    return ghatij


def calculate_evecs(gij, mismatch):
    """
    A function to calculate the eigenvectors of the metric gij normalized
    so that the match along the eigen-direction is given by mismatch


    :param gij: the metric
    :param mismatch: the required mismatch
    :return v: the appropriately scaled eigenvectors
    """
    evals, evec = np.linalg.eig(gij)
    # remove any negative evals
    evals[evals <= 0] = 1e-8
    v = (evec * np.sqrt((mismatch) / evals)).T
    return v


def update_metric(x, gij, basis, mismatch, dist, f_low, psd,
                  waveform="IMRPhenomD", tolerance=1e-2):
    """
    A function to re-calculate the metric gij based on the matches obtained
    for the eigenvectors of the original metric

    :param x: the point in parameter space used to calculate the metric
    :param gij: the original metric
    :param basis: the basis relating the directions of the metric to the physical space
    :param mismatch: the desired mismatch
    :param dist: distance to the signal
    :param f_low: low frequency cutoff
    :param psd: the power spectrum to use in calculating the match
    :param waveform: the waveform generator to use
    :return gij_prime: the updated metric
    :return ev_scale: a scaled set of eigenvectors
    """
    evecs = calculate_evecs(gij, mismatch)
    v_phys = np.inner(evecs, basis.T)
    v_scale = scale_vectors(x, v_phys, dist, mismatch, f_low, psd,
                            waveform, tolerance)
    ev_scale = (evecs.T *
                np.linalg.norm(v_scale, axis=1) / np.linalg.norm(v_phys, axis=1)).T
    g_prime = calculate_metric(x, v_scale, dist, f_low, psd, waveform)
    evec_inv = np.linalg.inv(ev_scale)
    gij_prime = np.inner(np.inner(evec_inv, g_prime), evec_inv)
    return gij_prime, ev_scale


def metric_error(gij, evecs, mismatch):
    """
    A function to calculate the inner products between the evecs and check
    they are orthogonal and correctly normalized

    :param gij: the metric
    :param evecs: the eigenvectors
    :param mismatch: the desired mismatch (equivalently, norm of evecs)
    :return max_err: the maximum error in the inner products
    """
    vgv = np.inner(np.inner(evecs, gij), evecs)
    off_diag = np.max(abs(vgv[~np.eye(gij.shape[0], dtype=bool)]))
    diag = np.max(abs(np.diag(vgv)) - mismatch)
    max_err = max(off_diag, diag)
    return max_err


def iteratively_update_metric(x, gij, basis, mismatch, tolerance, dist,
                              f_low, psd, waveform="IMRPhenomD", max_iter=20, verbose=False):
    """
    A function to re-calculate the metric gij based on the matches obtained
    for the eigenvectors of the original metric

    :param x: the point in parameter space used to calculate the metric
    :param gij: the original metric
    :param basis: the basis relating the directions of the metric to the physical space
    :param mismatch: the desired mismatch
    :param tolerance: the allowed error in the metric is (tolerance * mismatch)
    :param dist: distance to the signal
    :param f_low: low frequency cutoff
    :param psd: the power spectrum to use in calculating the match
    :param waveform: the waveform generator to use
    :return g_prime: the updated metric
    :return v: scaled eigenvectors
    """
    g = gij
    v = np.eye(len(basis))
    tol = tolerance * mismatch
    err = metric_error(g, v, mismatch)
    if verbose:
        print("Initial error in metric: %.2g" % err)

    op = 0
    while (err > tol) and (op < max_iter):
        g, v = update_metric(x, g, basis, mismatch, dist,
                             f_low, psd, waveform, tolerance)
        err = metric_error(g, v, mismatch)
        op += 1
        if verbose:
            print("Iteration %d, desired error=%.2g, max error=%.2g" % (op, tol, err))
            print(g)

    if (err > tol):
        print("Failed to achieve requested tolerance.  Requested: %.2g; achieved %.2g" % (tol, err))

    return g, v, tol


def find_peak(data, xx, gij, basis, mismatch, dist, f_low, psd,
              waveform="IMRPhenomD", verbose=False):
    """
    A function to find the maximum match.
    This is done in two steps, first by finding the point in the grid defined
    by the metric gij (and given mismatch) that gives the highest match.
    Second, we approximate the match as quadratic and find the maximum.

    :param data: the data containing the waveform of interest
    :param xx: the point in parameter space used to calculate the metric
    :param gij: the parameter space metric
    :param basis: the basis relating the directions of the metric to the physical space
    :param mismatch: the desired mismatch
    :param dist: distance to the signal
    :param f_low: low frequency cutoff
    :param psd: the power spectrum to use in calculating the match
    :param waveform: the waveform generator to use
    :return x_prime: the point in the grid with the highest match
    :return m_0: the match at this point
    :return steps: the number of steps taken in each eigen-direction
    """

    x = copy.deepcopy(xx)
    evecs = calculate_evecs(gij, mismatch)
    ndim = len(evecs)
    v_phys = np.inner(evecs, basis.T)
    steps = np.zeros(ndim)

    while True:
        h = make_waveform(x, np.zeros_like(x), dist, psd.delta_f, f_low, len(psd), waveform)
        m_0, _ = match(data, h, psd, low_frequency_cutoff=f_low)
        matches = np.zeros([ndim, 2])
        alphas = np.zeros([ndim, 2])

        for i in range(ndim):
            for j in range(2):
                dx = (-1) ** j * v_phys[i]
                alphas[i, j] = check_physical(x, dx)
                h = make_waveform(x, alphas[i, j] * dx, dist, psd.delta_f, f_low, len(psd),
                                  waveform)
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
            if verbose: print("Maximum at the centre, stopping")
            break

    s = (matches[:, 0] - matches[:, 1]) * 0.25 / \
        (m_0 - 0.5 * (matches[:, 0] + matches[:, 1]))
    steps += s
    delta_x = np.inner(s, v_phys.T)
    alpha = check_physical(x, delta_x)
    delta_x *= alpha

    if verbose:
        print("Using matches to find peak")
        print("Moving in the eigendirections distance of"),
        print("%.2g" % alpha * s)
        print("New position"),
        print(x + delta_x)

    h = make_waveform(x, delta_x, dist, psd.delta_f, f_low, len(psd), waveform)
    m_peak = match(data, h, psd, low_frequency_cutoff=f_low)[0]
    x_peak = x + delta_x

    return x_peak, m_peak, steps


def find_peak_snr(data, psd, ifos, t_start, t_end, xx, gij, basis, mismatch, dist, f_low, f_high,
                  waveform="IMRPhenomD", verbose=False):
    """
    A function to find the maximum SNR.
    Start at the point with parameters xx and use the metric gij to calculate eigen-directions.
    Calculate the SNR along each of the eigen-directions at a given step size (as defined by the
    initial metric).  If any are higher, then move.  If not, then reduce step size.
    Repeat until step size reaches requested mismatch.

    :param data: the data containing the waveform of interest
    :param psd: the power spectrum to use in calculating the match
    :param t_start: start time to consider SNR peak
    :param t_end: end time to consider SNR peak
    :param xx: the point in parameter space used to calculate the metric
    :param gij: the parameter space metric
    :param basis: the basis relating the directions of the metric to the physical space
    :param mismatch: the desired mismatch
    :param dist: distance to the signal
    :param f_low: low frequency cutoff
    :param f_high: high frequency cutoff
    :param waveform: the waveform generator to use
    :param verbose: print debugging information (if True)
    :return x_prime: the point in the grid with the highest snr
    :return snrsq_peak: the match at this point
    :return steps: the number of steps taken in each eigendirection
    """

    x = copy.deepcopy(xx)
    evecs = calculate_evecs(gij, mismatch)
    ndim = len(evecs)
    v_phys = np.inner(evecs, basis.T)
    steps = np.zeros(ndim)
    df = psd[ifos[0]].delta_f

    flen = int(len(data[ifos[0]]) / 2 + 1)

    while True:
        h = make_waveform(x, np.zeros_like(x), dist, df, f_low, flen, waveform)
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
                h = make_waveform(x, alphas[i, j] * dx, dist, df, f_low, flen,
                                  waveform)
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

    delta_x = np.inner(s, v_phys.T)
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

    h = make_waveform(x, delta_x, dist, df, f_low, flen, waveform)
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
                         approximant):
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
    :param approximant: the waveform to use
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
    mismatch = scale / trigger_snr ** 2
    min_mis = 1e-2 / trigger_snr ** 2
    tolerance = 0.05
    max_iter = 8

    hm_psd = len(ifos) / sum([1. / psd[ifo] for ifo in ifos])
    f_high = hm_psd.sample_frequencies[-1]


    while mismatch > min_mis:
        basis = scale_vectors(x, v0, dist, mismatch, f_low, hm_psd, waveform=approximant)
        gij = calculate_metric(x, basis, dist, f_low, hm_psd, waveform=approximant)
        gij, v, t = iteratively_update_metric(x, gij, basis, mismatch, tolerance,
                                              dist, f_low, hm_psd, waveform=approximant, max_iter=max_iter)
        x, snr_now, steps = find_peak_snr(strain, psd, ifos, t_start, t_end, x, gij, basis, mismatch, dist,
                                          f_low, f_high,
                                          waveform=approximant)

        if x[1] == 0.25:
            x -= np.dot([0., 0.001, 0.], basis)
            # break
        if np.linalg.norm(steps) < 1:
            # didn't move much so reduce step size
            mismatch /= 4.

    h = make_waveform(x, np.zeros_like(x), dist, hm_psd.delta_f, f_low, len(hm_psd),
                      waveform=approximant)
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

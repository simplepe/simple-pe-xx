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
    returns h_plus generated at value (x + dx).  Currently defaults to

    Parameters
    ----------
    x : np.array with four values assumed to be mchirp, eta, s1z, s2z
    dx: same as x.
    dist: distance to the signal
    df: frequency spacing of points
    f_low: low frequency cutoff
    flen: length of the frequency domain array to generate
    waveform: the waveform generator to use

    Returns
    -------
    h_plus: waveform as a frequency series with the requested df, flen
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

    Parameters
    ----------
    x : np.array with four values assumed to be mchirp, eta, s1z, s2z
    vec: an array of directions dx in which to vary the waveform parameters
    dist: distance to the signal
    mismatch: the desired mismatch (1 - match)
    f_low: low frequency cutoff
    psd: the power spectrum to use in calculating the match
    waveform: the waveform generator to use
    tolerance: the maximum fractional error in the mismatch

    Returns
    -------
    v: A set of vectors in the directions given by v but normalized to give the
    desired mismatch
    """
    ndim = len(vec)
    v = copy.deepcopy(vec)
    for i in range(ndim):
        opt = optimize.root_scalar(lambda a: average_mismatch(x, a * v[i], dist,
                                                              f_low, psd, waveform) - mismatch,
                                   bracket=[0, 5], method='brentq', rtol=tolerance)
        v[i] *= opt.root

    return v


def check_physical(x, dx, maxs=[1e4, 0.25, 0.98, 0.98],
                   mins=[0, 0.05, -0.98, -0.98]):
    """
    A function to check whether ther point described by the positions x + dx is
    physically permitted.  If not, rescale and return the scaling factor

    Parameters
    ----------
    x : np.array with four values assumed to be mchirp, eta, s1z, s2z
    dx: same as x.
    maxs: the maximum permitted values of the physical parameters
    mins: the minimum physical values of the physical parameters

    Returns
    -------
    alpha: the scaling factor required to make x + dx physically permissable
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

    Parameters
    ----------
    m_alpha: the match at an offset alpha
    alpha: the value of alpha

    Returns
    -------
    m : the match at unit offset
    """
    m = (alpha ** 2 - 1 + m_alpha) / alpha ** 2
    return m


def average_mismatch(x, dx, dist, f_low, psd,
                     waveform="IMRPhenomD", verbose=False):
    """
    This function calculated the average match for steps of +dx and -dx
    It also takes care of times where one of the steps moves beyond the
    edge of the physical paramter space

    Parameters
    ----------
    x : np.array with four values assumed to be mchirp, eta, chi_eff
    dx: the change in the values x
    dist: distance to the signal
    f_low: low frequency cutoff
    flen: length of the frequency domain array to generate
    psd: the power spectrum to use in calculating the match
    waveform: the waveform generator to use

    Returns
    -------
    m: The average match from steps of +/-dx
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

    Parameters
    ----------
    x : np.array with three values assumed to be mchirp, eta, sz
    vec: an array of directions dx in which to vary the waveform parameters
    dist: distance to the signal
    f_low: low frequency cutoff
    psd: the power spectrum to use in calculating the match
    waveform: the waveform generator to use

    Returns
    -------
    gij: a square matrix, with size given by the length of vec, that gives the
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

    Parameters
    ----------
    gij: the metric calculated with respect to a set of basis vectors
    basis: the basis vectors expressed in terms of the physical coordinates

    Returns
    -------
    gphys: the metric in physical coordinates
    """
    vnorm = np.linalg.norm(basis, axis=1)
    ghatij = gij / vnorm / vnorm.reshape((-1, 1))
    return ghatij


def calculate_evecs(gij, mismatch):
    """
    A function to calculate the eigenvectors of the metric gij normalized
    so that the match along the eigendirection is given by mismatch

    Parameters
    ----------
    gij: the metric
    mismatch: the required mismatch

    Returns
    -------
    v: the appropriately scaled eigenvectors
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

    Parameters
    ----------
    x: the point in parameter space used to calculate the metric
    gij: the original metric
    basis: the basis relating the directions of the metric to the physical space
    mismatch: the desired mismatch
    dist: distance to the signal
    f_low: low frequency cutoff
    psd: the power spectrum to use in calculating the match
    waveform: the waveform generator to use

    Returns
    -------
    gij_prime: the updated metric
    ev_scale: a scaled set of eigenvectors
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

    Parameters
    ----------
    gij: the metric
    evecs: the eigenvectors
    misnatch: the desired mismatch (equivalently, norm of evecs)

    Returns
    -------
    max_err: the maximum error in the inner products
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

    Parameters
    ----------
    x: the point in parameter space used to calculate the metric
    gij: the original metric
    basis: the basis relating the directions of the metric to the physical space
    mismatch: the desired mismatch
    tolerance: the allowed error in the metric is (tolerance * mismatch)
    dist: distance to the signal
    f_low: low frequency cutoff
    psd: the power spectrum to use in calculating the match
    waveform: the waveform generator to use

    Returns
    -------
    g_prime: the updated metric
    v: scaled eigenvectors
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

    Parameters
    ----------
    data: the data containing the waveform of interest
    xx: the point in parameter space used to calculate the metric
    gij: the parameter space metric
    basis: the basis relating the directions of the metric to the physical space
    mismatch: the desired mismatch
    dist: distance to the signal
    f_low: low frequency cutoff
    psd: the power spectrum to use in calculating the match
    waveform: the waveform generator to use

    Returns
    -------
    x_prime: the point in the grid with the highest match
    m_0: the match at this point
    steps: the number of steps taken in each eigendirection
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

    Parameters
    ----------
    data: the data containing the waveform of interest
    psd: the power spectrum to use in calculating the match
    t_start: start time to consider SNR peak
    t_end: end time to consider SNR peak
    xx: the point in parameter space used to calculate the metric
    gij: the parameter space metric
    basis: the basis relating the directions of the metric to the physical space
    mismatch: the desired mismatch
    dist: distance to the signal
    f_low: low frequency cutoff
    f_high: high frequency cutoff
    waveform: the waveform generator to use
    verbose: print debugging information (if True)

    Returns
    -------
    x_prime: the point in the grid with the highest snr
    snrsq_peak: the match at this point
    steps: the number of steps taken in each eigendirection
    """

    x = copy.deepcopy(xx)
    evecs = calculate_evecs(gij, mismatch)
    ndim = len(evecs)
    v_phys = np.inner(evecs, basis.T)
    steps = np.zeros(ndim)
    df = psd[ifos[0]].delta_f

    flen = len(psd[ifos[0]])

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


def matched_filter_network(data, psd, ifos, t_start, t_end, h, f_low, f_high):
    """
     A function to find the maximum SNR in the network within a given time range

     Parameters
     ----------
     data: the data containing the waveform of interest
     psd: the power spectrum to use in calculating the match
     ifos: list of ifos
     t_start: start time to consider SNR peak
     t_end: end time to consider SNR peak
     f_low: low frequency cutoff
     f_high: high frequency cutoff

     Returns
     -------
     snr: the network snr
     smax: the max snr in each ifo
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

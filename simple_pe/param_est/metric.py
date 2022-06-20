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


def make_waveform(x, dx, scaling, dist, df, f_low, flen, approximant="IMRPhenomD"):
    """
    This function makes a waveform for the given parameters and
    returns h_plus generated at value (x + dx).

    :param x: dictionary with parameter values for initial point
    :param dx: dictionary with parameter variations
    :param scaling: the scaling to apply to dx
    :param dist: distance to the signal
    :param df: frequency spacing of points
    :param f_low: low frequency cutoff
    :param flen: length of the frequency domain array to generate
    :param approximant: the approximant generator to use
    :return h_plus: waveform at parameter space point x + scaling * dx
    """

    tmp_x = copy.deepcopy(x)
    for k, dx_val in dx.items():
        tmp_x[k] += scaling * dx_val

    if 'chi_eff' in x.keys():
        tmp_x['spin1z'] = tmp_x['chi_eff']
        tmp_x['spin2z'] = tmp_x['chi_eff']

    data = convert(tmp_x)

    h_plus, h_cross = get_fd_waveform(mass1=data['mass_1'], mass2=data['mass_2'],
                                      spin1z=data['spin1z'],
                                      spin2z=data['spin2z'],
                                      delta_f=df, distance=dist, f_lower=f_low,
                                      approximant=approximant,
                                      mode_array=waveform_modes.mode_array('22', approximant))
    h_plus.resize(flen)
    return h_plus


def check_physical(x, dx, scaling, maxs=None, mins=None):
    """
    A function to check whether the point described by the positions x + dx is
    physically permitted.  If not, rescale and return the scaling factor

    :param x: dictionary with parameter values for initial point
    :param dx: dictionary with parameter variations
    :param scaling: the scaling to apply to dx
    :param maxs: the maximum permitted values of the physical parameters
    :param mins: the minimum physical values of the physical parameters
    :return alpha: the scaling factor required to make x + scaling * dx physically permissible
    """
    if mins is None:
        mins = {'chirp_mass': 1.,
                'total_mass': 2.,
                'symmetric_mass_ratio': 0.08,
                'chi_eff': -0.98,
                'spin1z': -0.98,
                'spin2z': -0.98
                }

    if maxs is None:
        maxs = {'chirp_mass': 1e4,
                'total_mass': 1e4,
                'symmetric_mass_ratio': 0.25,
                'chi_eff': 0.98,
                'spin1z': 0.98,
                'spin2z': 0.98
                }

    alpha = 1.

    for k, dx_val in dx.items():
        if k in maxs and (x[k] + scaling * dx[k]) < mins[k]:
            alpha = min(alpha, (x[k] - mins[k]) / abs(scaling * dx[k]))
        if k in mins and (x[k] + scaling * dx[k]) > maxs[k]:
            alpha = min(alpha, (maxs[k] - x[k]) / abs(scaling * dx[k]))

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


def average_mismatch(x, dx, scaling, dist, f_low, psd,
                     approximant="IMRPhenomD", verbose=False):
    """
    This function calculates the average match for steps of +dx and -dx
    It also takes care of times when one of the steps moves beyond the
    edge of the physical parameter space

    :param x : np.array with three values assumed to be mchirp, eta, chi_eff
    :param dx: the change in the values x
    :param scaling: the scaling to apply to dx
    :param dist: distance to the signal
    :param f_low: low frequency cutoff
    :param psd: the power spectrum to use in calculating the match
    :param mass: use either chirp or total mass
    :param approximant: the approximant generator to use
    :param verbose: print debugging information
    :return m: The average match from steps of +/- scaling * dx
    """
    a = {}
    m = {}
    h0 = make_waveform(x, dx, 0., dist, psd.delta_f, f_low, len(psd), approximant)
    for s in [1., -1.]:
        a[s] = check_physical(x, dx, s * scaling)
        h = make_waveform(x, dx, s * a[s] * scaling, dist, psd.delta_f, f_low, len(psd), approximant)
        m[s] = match(h0, h, psd, low_frequency_cutoff=f_low)[0]
    if verbose:
        print("Had to scale steps to %.2f, %.2f" % (a[-1], a[1]))
        print("Mismatches %.3f, %.3f" % (1 - m[-1], 1 - m[1]))
    if min(a.values()) < 1e-2:
        if verbose:
            print("we're really close to the boundary, so down-weight match contribution")
        mm = (2 - m[1] - m[-1]) / (a[1] ** 2 + a[-1] ** 2)
    else:
        mm = 1 - 0.5 * (scale_match(m[1], a[1]) + scale_match(m[-1], a[-1]))
    return mm


def scale_dx(x, dx, desired_mismatch, dist, f_low, psd,
                  approximant="IMRPhenomD", tolerance=1e-2):
    """
    This function scales the input vectors so that the mismatch between
    a waveform at point x and one at x + v[i] is equal to the specified
    mismatch, up to the specified tolerance.

    :param x: dictionary with parameter values for initial point
    :param dx: a dictionary with the initial change in parameters
    :param desired_mismatch: the desired mismatch (1 - match)
    :param dist: distance to the signal
    :param f_low: low frequency cutoff
    :param psd: the power spectrum to use in calculating the match
    :param approximant: the approximant generator to use
    :param tolerance: the maximum fractional error in the mismatch
    :return scale: The required scaling of dx to achieve the desired mismatch
    """
    opt = optimize.root_scalar(lambda a: average_mismatch(x, dx, a, dist, f_low, psd,
                                                          approximant=approximant) - desired_mismatch,
                                   bracket=[0, 20], method='brentq', rtol=tolerance)
    scale = opt.root

    return scale


def scale_dx_array(x, dxs, desired_mismatch, dist, f_low, psd,
                  approximant="IMRPhenomD", tolerance=1e-2):
    """
    This function scales the input vectors so that the mismatch between
    a waveform at point x and one at x + v[i] is equal to the specified
    mismatch, up to the specified tolerance.

    :param x: dictionary with parameter values for initial point
    :param dxs: a dictionary with the a set of changes in parameters
    :param desired_mismatch: the desired mismatch (1 - match)
    :param dist: distance to the signal
    :param f_low: low frequency cutoff
    :param psd: the power spectrum to use in calculating the match
    :param approximant: the approximant generator to use
    :param tolerance: the maximum fractional error in the mismatch
    :return scale: The required scaling of dx to achieve the desired mismatch
    """
    nsamp = dxs.number_of_samples
    scale = np.zeros(nsamp)
    for i in range(nsamp):
        dx = dxs[i:i + 1]
        scale[i] = scale_dx(x, dx, desired_mismatch, dist, f_low, psd, approximant, tolerance)

    scaled_dx_array = SamplesDict(dxs.keys(), dxs.samples * scale)

    return scaled_dx_array


def calculate_metric(x, dxs, dist, f_low, psd, approximant="IMRPhenomD", verbose=False):
    """
    A function to calculate the metric at a point x, associated to a given set
    of variations in the directions given by dxs.
    :param x: dictionary with parameter values for initial point
    :param dxs: a dictionary of arrays with the initial change in parameters
    :param dist: distance to the signal
    :param f_low: low frequency cutoff
    :param psd: the power spectrum to use in calculating the match
    :param approximant: the approximant generator to use
    :return gij: a square matrix, with size given by the length of dxs, that gives the
         metric at x along the directions given by dxs
    """
    ndim = len(dxs.keys())

    scaling = 1.
    gij = np.zeros([ndim, ndim])

    # diagonal components
    # g_ii = 1 - 0.5 [m(dx_i) + m(-dx_i)]
    for i in range(ndim):
        gij[i, i] += average_mismatch(x, dxs[i:i+1], scaling, dist, f_low, psd, approximant)

    # off diagonal
    # g_ij = 0.25 * [+ m(1/sqrt(2) (+ dx_i + dx_j)) + m(1/sqrt(2) (- dx_i - dx_j))
    #                - m(1/sqrt(2) (+ dx_i - dx_j)) - m(1/sqrt(2) (- dx_i + dx_j))]
    for i in range(ndim):
        for j in range(i + 1, ndim):
            for s in ([1, -1]):
                dx = {}
                for k, vals in dxs.items():
                    dx[k] = (vals[i] + s * vals[j]) / np.sqrt(2)
                gij[i, j] += 0.5 * s * average_mismatch(x, dx, scaling, dist, f_low, psd, approximant)
            gij[j, i] = gij[i, j]

    # this gives the metric in coordinates given by the dxs.
    # Let's instead return the metric in physical parameter space
    gphys_ij = physical_metric(gij, dxs)

    if verbose:
        print("coordinate metric")
        print(gij)
        print("coordinate variations")
        print(dxs)
        print("physical metric")
        print(gphys_ij)

    return gphys_ij


def physical_metric(gij, dxs):
    """
    A function to calculate the metric in physical coordinates

    :param gij: the metric calculated with respect to a set of basis vectors
    :param dxs: the basis vectors expressed in terms of the physical coordinates
    :return gphys: the metric in physical coordinates
    """
    dx_inv = np.linalg.inv(dxs.samples)
    gphys_ij = np.matmul(dx_inv.T, np.matmul(gij, dx_inv))

    return gphys_ij


def metric_error(gij, basis, desired_mismatch):
    """
    We are looking for a metric and corresponding basis for which the
    basis is orthogonal and whose normalization is given by the desired mismatch
    This function checks for the largest error

    :param gij: the metric
    :param basis: the basis vectors used to calculate the metric
    :param desired_mismatch: the desired mismatch (equivalently, norm of evecs)
    :return max_err: the maximum error in the inner products
    """
    vgv = np.matmul(basis.samples.T, np.matmul(gij, basis.samples))
    off_diag = np.max(abs(vgv[~np.eye(gij.shape[0], dtype=bool)]))
    diag = np.max(abs(np.diag(vgv)) - desired_mismatch)
    max_err = max(off_diag, diag)
    return max_err


def calculate_evecs(gij, desired_mismatch):
    """
    A function to calculate the eigenvectors of the metric gij normalized
    so that the match along the eigen-direction is given by desired_mismatch

    :param gij: the metric
    :param desired_mismatch: the required desired_mismatch
    :return v: the appropriately scaled eigenvectors
    """
    evals, evec = np.linalg.eig(gij)
    # remove any negative evals
    evals[evals <= 0] = 1.
    v = (evec * np.sqrt(desired_mismatch / evals))
    return v


def update_metric(x, gij, dx_directions, desired_mismatch, dist, f_low, psd,
                  approximant="IMRPhenomD", tolerance=1e-2, verbose=False):
    """
    A function to re-calculate the metric gij based on the matches obtained
    for the eigenvectors of the original metric

    :param x: the point in parameter space used to calculate the metric
    :param gij: the original metric
    :param dx_directions: list of physical parameters corresponding to metric components
    :param desired_mismatch: the desired mismatch
    :param dist: distance to the signal
    :param f_low: low frequency cutoff
    :param psd: the power spectrum to use in calculating the match
    :param approximant: the approximant generator to use
    :param tolerance: the required accuracy in rescaling the basis vectors
    :return gij_prime: the updated metric
    :return ev_scale: a scaled set of eigenvectors
    """
    # calculate the eigendirections of the matrix
    evecs = np.linalg.eig(gij)[1]
    dx0 = SamplesDict(dx_directions, evecs)
    # scale so that they actually have the desired mismatch
    dxs = scale_dx_array(x, dx0, desired_mismatch, dist, f_low, psd, approximant, tolerance)
    g_prime = calculate_metric(x, dxs, dist, f_low, psd, approximant, verbose)

    return g_prime, dxs


def iteratively_update_metric(x, gij, dxs, desired_mismatch, dist,
                              f_low, psd, approximant="IMRPhenomD", tolerance=1e-2,
                              max_iter=20, verbose=False):
    """
    A function to re-calculate the metric gij based on the matches obtained
    for the eigenvectors of the original metric

    :param x: the point in parameter space used to calculate the metric
    :param gij: the original metric
    :param dxs: a dictionary of arrays with the initial change in parameters
    :param desired_mismatch: the desired mismatch
    :param dist: distance to the signal
    :param f_low: low frequency cutoff
    :param psd: the power spectrum to use in calculating the match
    :param approximant: the approximant generator to use
    :param tolerance: the allowed error in the metric is (tolerance * desired_mismatch)
    :param max_iter: maximum number of iterations before stopping
    :param verbose: print debugging information
    :return g_prime: the updated metric
    :return v: scaled eigenvectors
    :return tol: tolerance achieved
    """
    tol = tolerance * desired_mismatch

    dx_directions = dxs.keys()
    err = metric_error(gij, dxs, desired_mismatch)
    if verbose:
        print("Initial error in metric: %.2g" % err)

    op = 0
    g = gij

    while (err > tol) and (op < max_iter):
        g, dxs = update_metric(x, g, dx_directions, desired_mismatch, dist,
                             f_low, psd, approximant, tolerance, verbose)
        err = metric_error(g, dxs, desired_mismatch)
        op += 1
        if verbose:
            print("Iteration %d, desired error=%.2g, max error=%.2g" % (op, tol, err))
            print(g)

    if (err > tol):
        print("Failed to achieve requested tolerance.  Requested: %.2g; achieved %.2g" % (tol, err))

    return g, dxs, tol


def find_metric_and_eigendirections(x, dx_directions, snr, f_low, psd, approximant="IMRPhenomD",
                         tolerance=0.05, max_iter=20, verbose=False):
    """
    Calculate the eigendirections in parameter space, normalized to enclose a 90% confidence
    region at the requested SNR

    :param x: dictionary with parameter values for initial point
    :param dx_directions: list of parameters for which to calculate waveform variations
    :param snr: the observed SNR
    :param f_low: low frequency cutoff
    :param psd: the power spectrum to use in calculating the match
    :param mass: either chirp or total
    :param approximant: the approximant to use
    :param tolerance: the allowed error in the metric is (tolerance * mismatch)
    :param verbose: print debugging information
    :param max_iter: the maximum number of iterations
    :return v_phys: the scaled eigendirections in physical space
    :return gij: the mismatch metric in physical space
    """
    # initial directions and initial spacing
    ndim = len(dx_directions)
    n_sigmasq = chi2.isf(0.1, ndim)
    desired_mismatch = n_sigmasq / (2 * snr ** 2)
    dist = 1.

    dx0 = SamplesDict(dx_directions, np.eye(ndim))

    dxs = scale_dx_array(x, dx0, desired_mismatch, dist, f_low, psd,
                         approximant, tolerance)

    gij = calculate_metric(x, dxs, dist, f_low, psd, approximant, verbose)
    g, v, t = iteratively_update_metric(x, gij, dxs, desired_mismatch,
                                        dist, f_low, psd, approximant, tolerance, max_iter, verbose)

    return g, v


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

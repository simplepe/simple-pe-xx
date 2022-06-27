import numpy as np
from pycbc.waveform import get_fd_waveform
from pycbc.filter import match
from simple_pe.waveforms import waveform_modes
import copy
from scipy import optimize
from scipy.stats import chi2
from pesummary.gw.conversions import convert
from pesummary.utils.samples_dict import SamplesDict


class Metric:
    """
    A class to store the parameter space metric at a point x in parameter space,
    where the variations are given by dxs
    :param x: dictionary with parameter values for initial point
    :param dxs: a dictionary of arrays with the initial change in parameters
    :param dist: distance to the signal
    :param f_low: low frequency cutoff
    :param psd: the power spectrum to use in calculating the match
    :param approximant: the approximant generator to use
    :return gij: a square matrix, with size given by the length of dxs, that gives the
         metric at x along the directions given by dxs
    """
    def __init__(self, x, dx_directions, mismatch, f_low, psd, approximant="IMRPhenomD", tolerance=1e-2):
        """
        :param x: dictionary with parameter values for initial point
        :param dx_directions: a list of directions to vary
        :param mismatch: the mismatch value to use when calculating metric
        :param f_low: low frequency cutoff
        :param psd: the power spectrum to use in calculating the match
        :param approximant: the approximant generator to use
        :param tolerance: tolerance for scaling vectors
        """
        self.x = x
        self.dx_directions = dx_directions
        self.ndim = len(dx_directions)
        self.dxs = SamplesDict(self.dx_directions, np.eye(self.ndim))
        self.mismatch = mismatch
        self.f_low = f_low
        self.psd = psd
        self.approximant = approximant
        self.distance = 1.
        self.tolerance = tolerance
        self.coordinate_metric = None
        self.metric = None
        self.evals = None
        self.evec = None
        self.err = None
        self.projected_directions = None
        self.projected_metric = None
        self.projection = None

        self.scale_dxs()
        self.calculate_metric()

    def scale_dxs(self):
        """
        This function scales the vectors dxs so that the mismatch between
        a waveform at point x and one at x + dxs[i] is equal to the specified
        mismatch, up to the specified tolerance.
        """
        scale = np.zeros(self.ndim)
        for i in range(self.ndim):
            dx = self.dxs[i:i + 1]
            scale[i] = scale_dx(self.x, dx, self.mismatch, self.distance, self.f_low, self.psd,
                                self.approximant, self.tolerance)

        self.dxs = SamplesDict(self.dx_directions, self.dxs.samples * scale)

    def calculate_metric(self):
        scaling = 1.
        gij = np.zeros([self.ndim, self.ndim])

        # diagonal components
        # g_ii = 1 - 0.5 [m(dx_i) + m(-dx_i)]
        for i in range(self.ndim):
            gij[i, i] += average_mismatch(self.x, self.dxs[i:i+1], scaling, self.distance, self.f_low,
                                          self.psd, self.approximant)

        # off diagonal
        # g_ij = 0.25 * [+ m(1/sqrt(2) (+ dx_i + dx_j)) + m(1/sqrt(2) (- dx_i - dx_j))
        #                - m(1/sqrt(2) (+ dx_i - dx_j)) - m(1/sqrt(2) (- dx_i + dx_j))]
        for i in range(self.ndim):
            for j in range(i + 1, self.ndim):
                for s in ([1, -1]):
                    dx = {}
                    for k, vals in self.dxs.items():
                        dx[k] = (vals[i] + s * vals[j]) / np.sqrt(2)
                    gij[i, j] += 0.5 * s * average_mismatch(self.x, dx, scaling, self.distance,
                                                            self.f_low, self.psd, self.approximant)
                gij[j, i] = gij[i, j]

        # this gives the metric in coordinates given by the dxs.
        # Let's instead return the metric in physical parameter space
        self.coordinate_metric = gij
        self.physical_metric()

    def physical_metric(self):
        """
        A function to calculate the metric in physical coordinates

        :param gij: the metric calculated with respect to a set of basis vectors
        :param dxs: the basis vectors expressed in terms of the physical coordinates
        :return gphys: the metric in physical coordinates
        """
        dx_inv = np.linalg.inv(self.dxs.samples)
        self.metric = np.matmul(dx_inv.T, np.matmul(self.coordinate_metric, dx_inv))

    def calculate_evecs(self):
        """
        A function to calculate the eigenvectors and eigenvalues of the metric gij
        """
        self.evals, self.evec = np.linalg.eig(self.metric)

    def normalized_evecs(self):
        """
        Return the evecs normalized to give the desired mismatch
        """
        return self.evec * np.sqrt(self.mismatch/self.evals)

    def calc_metric_error(self):
        """
        We are looking for a metric and corresponding basis for which the
        basis is orthogonal and whose normalization is given by the desired mismatch
        This function checks for the largest error

        """
        vgv = np.matmul(self.dxs.samples.T, np.matmul(self.metric, self.dxs.samples))
        off_diag = np.max(abs(vgv[~np.eye(self.metric.shape[0], dtype=bool)]))
        diag = np.max(abs(np.diag(vgv)) - self.mismatch)
        self.err = max(off_diag, diag)

    def update_metric(self):
        """
        A function to re-calculate the metric gij based on the matches obtained
        for the eigenvectors of the original metric

        :param tolerance: the required accuracy in rescaling the basis vectors
        """
        # calculate the eigendirections of the matrix
        self.calculate_evecs()
        self.dxs = SamplesDict(self.dx_directions, self.evec)
        # scale so that they actually have the desired mismatch
        self.scale_dxs()
        self.calculate_metric()

    def iteratively_update_metric(self, max_iter=20):
        """
        A method to re-calculate the metric gij based on the matches obtained
        for the eigenvectors of the original metric
        """
        tol = self.tolerance * self.mismatch
        self.calc_metric_error()

        op = 0

        while (self.err > tol) and (op < max_iter):
            self.update_metric()
            self.calc_metric_error()
            op += 1

        if self.err > tol:
            print("Failed to achieve requested tolerance.  Requested: %.2g; achieved %.2g" % (tol, self.err))

    def project_metric(self, projected_directions):
        """
        Project out the unwanted directions of the metric

        :param projected_directions: list of parameters that we want to keep
        """
        self.projected_directions = projected_directions
        kept = np.ones(self.ndim, dtype=bool)
        for i,x in enumerate(self.dx_directions):
            if x not in projected_directions:
                kept[i] = False

        # project out unwanted: g'_ab = g_ab - g_ai (ginv)^ij g_jb (where a,b run over kept, i,j over removed)
        # matrix to calculate the values of the maximized directions: x^i = (ginv)^ij g_jb
        ginv = np.linalg.inv(self.metric[~kept][:, ~kept])
        self.projection = - np.matmul(ginv, self.metric[~kept][:, kept])
        self.projected_metric = self.metric[kept][:, kept] + np.matmul(self.metric[kept][:, ~kept], self.projection)

    def projected_evecs(self):
        """
        Return the evecs normalized to give the desired mismatch
        """
        evals, evec = np.linalg.eig(self.projected_metric)
        return evec * np.sqrt(self.mismatch/evals)


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
        tmp_x[k] += float(scaling * dx_val)

    if 'chi_eff' in x.keys():
        tmp_x['spin_1z'] = tmp_x['chi_eff']
        tmp_x['spin_2z'] = tmp_x['chi_eff']

    data = convert(tmp_x, disable_remnant=True)

    h_plus, h_cross = get_fd_waveform(mass1=data['mass_1'], mass2=data['mass_2'],
                                      spin1z=data['spin_1z'],
                                      spin2z=data['spin_2z'],
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
                'mass_1': 1.,
                'mass_2': 1.,
                'symmetric_mass_ratio': 0.08,
                'chi_eff': -0.98,
                'spin_1z': -0.98,
                'spin_2z': -0.98
                }

    if maxs is None:
        maxs = {'chirp_mass': 1e4,
                'total_mass': 1e4,
                'mass_1': 1e4,
                'mass_2': 1e4,
                'symmetric_mass_ratio': 0.25,
                'chi_eff': 0.98,
                'spin_1z': 0.98,
                'spin_2z': 0.98
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


def find_metric_and_eigendirections(x, dx_directions, snr, f_low, psd, approximant="IMRPhenomD",
                         tolerance=0.05, max_iter=20):
    """
    Calculate the eigendirections in parameter space, normalized to enclose a 90% confidence
    region at the requested SNR

    :param x: dictionary with parameter values for initial point
    :param dx_directions: list of parameters for which to calculate waveform variations
    :param snr: the observed SNR
    :param f_low: low frequency cutoff
    :param psd: the power spectrum to use in calculating the match
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
    g = Metric(x, dx_directions, desired_mismatch, f_low, psd, approximant, tolerance)
    g.iteratively_update_metric(max_iter)
    return g


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


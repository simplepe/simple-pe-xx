import numpy as np
from pycbc.waveform import get_fd_waveform
from pycbc.filter import match
from simple_pe.waveforms import waveform_modes
import copy
from scipy import optimize
from scipy.stats import chi2
from simple_pe.param_est.pe import convert
from pesummary.utils.samples_dict import SamplesDict
from pesummary.gw import conversions


class Metric:
    """
    A class to store the parameter space metric at a point x in parameter space,
    where the variations are given by dxs
    :param x: dictionary with parameter values for initial point
    :param dxs: a dictionary of arrays with the initial change in parameters
    :param f_low: low frequency cutoff
    :param psd: the power spectrum to use in calculating the match
    :param approximant: the approximant generator to use
    :return gij: a square matrix, with size given by the length of dxs, that gives the
    metric at x along the directions given by dxs
    """

    def __init__(self, x, dx_directions, mismatch, f_low, psd,
                 approximant="IMRPhenomD", tolerance=1e-2, n_sigma=None):
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
        if 'distance' not in self.x:
            self.x['distance'] = 1.
        self.dx_directions = dx_directions
        self.ndim = len(dx_directions)
        self.dxs = SamplesDict(self.dx_directions, np.eye(self.ndim))
        self.mismatch = mismatch
        self.f_low = f_low
        self.psd = psd
        self.approximant = approximant
        self.tolerance = tolerance
        self.n_sigma = n_sigma
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
            scale[i] = scale_dx(self.x, dx, self.mismatch, self.f_low, self.psd,
                                self.approximant, self.tolerance)

        self.dxs = SamplesDict(self.dx_directions, self.dxs.samples * scale)

    def calculate_metric(self):
        scaling = 1.
        gij = np.zeros([self.ndim, self.ndim])

        # diagonal components
        # g_ii = 1 - 0.5 [m(dx_i) + m(-dx_i)]
        for i in range(self.ndim):
            gij[i, i] += average_mismatch(self.x, self.dxs[i:i + 1], scaling, self.f_low,
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
                    gij[i, j] += 0.5 * s * average_mismatch(self.x, dx, scaling,
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
        if self.evec is None:
            self.calculate_evecs()
        return SamplesDict(self.dx_directions, self.evec * np.sqrt(self.mismatch / self.evals))

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
        for i, x in enumerate(self.dx_directions):
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
        return SamplesDict(self.projected_directions, evec * np.sqrt(self.mismatch / evals))

    def generate_ellipse(self, npts=100, projected=False, mismatch=None):
        """
        Generate an ellipse of points of constant mismatch

        :param projected: use the projected metric if True, else use metric
        :param npts: number of points in
        :param mismatch: the mismatch at which to place the ellipse (if None, then use the value associated with the metric)
        :return ellipse_dict: SamplesDict with ellipse of points
        """
        if projected:
            dx_dirs = self.projected_directions
            n_evec = self.projected_evecs()
        else:
            dx_dirs = self.dx_directions
            n_evec = self.normalized_evecs()

        if len(dx_dirs) != 2:
            print("We're expecting to plot a 2-d ellipse")
            return -1

        if mismatch:
            r = np.sqrt(mismatch / self.mismatch)
        else:
            r = 1

        # generate points on a circle
        phi = np.linspace(0, 2 * np.pi, npts)
        xx = r * np.cos(phi)
        yy = r * np.sin(phi)
        pts = np.array([xx, yy])

        # project onto eigendirections
        dx = SamplesDict(dx_dirs, np.matmul(n_evec.samples, pts))

        # scale to be physical
        alphas = np.zeros(npts)
        for i in range(npts):
            alphas[i] = check_physical(self.x, dx[i:i + 1], 1.)

        ellipse_dict = SamplesDict(dx_dirs,
                                   np.array([self.x[dx] for dx in dx_dirs]).reshape([2, 1]) + dx.samples * alphas)

        return ellipse_dict

    def generate_match_grid(self, npts=10, projected=False, mismatch=None):
        """
        Generate an ellipse of points of constant mismatch

        :param projected: use the projected metric if True, else use metric
        :param npts: number of points in each dimension of the grid
        :param mismatch: the mismatch of the greatest extent of the grid (if None, then use the value associated with the metric)
        :return ellipse_dict: SamplesDict with ellipse of points
        """
        if projected:
            dx_dirs = self.projected_directions + [x for x in self.dx_directions if x not in self.projected_directions]
            n_evec = self.projected_evecs()
        else:
            dx_dirs = self.dx_directions
            n_evec = self.normalized_evecs()

        if mismatch:
            r = np.sqrt(mismatch / self.mismatch)
        else:
            r = 1

        grid = np.mgrid[-r:r:npts * 1j, -r:r:npts * 1j]
        dx_data = np.tensordot(n_evec.samples, grid, axes=(1, 0))

        if projected:
            dx_extra = np.tensordot(self.projection, dx_data, axes=(1, 0))
            dx_data = np.append(dx_data, dx_extra, 0)

        dx = SamplesDict(dx_dirs, dx_data.reshape(len(dx_dirs), npts ** 2))

        nsamp = dx.number_of_samples
        m = np.zeros(nsamp)

        h0 = make_waveform(self.x, self.psd.delta_f, self.f_low, len(self.psd), self.approximant)

        for i in range(nsamp):
            if check_physical(self.x, dx[i:i + 1], 1.) < 1:
                m[i] = 0
            else:
                h1 = make_offset_waveform(self.x, dx[i:i + 1], 1.,
                                          self.psd.delta_f, self.f_low, len(self.psd), self.approximant)
                m[i] = match(h0, h1, self.psd, self.f_low, subsample_interpolation=True)[0]

        matches = SamplesDict(dx_dirs + ['match'],
                              np.append(
                                  dx.samples + np.array([self.x[dx] for dx in dx_dirs]).reshape([len(dx_dirs), 1]),
                                  m.reshape([1, npts ** 2]), 0).reshape([len(dx_dirs) + 1, npts, npts]))

        return matches

    def generate_samples(self, npts=int(1e5)):
        """
        Generate an ellipse of points of constant mismatch

        :param npts: number of points to generate
        :return phys_samples: SamplesDict with samples
        """
        pts = np.random.normal(0, 1, [2 * npts, self.ndim])

        sample_pts = SamplesDict(self.dx_directions,
                                 (np.array([self.x[dx] for dx in self.normalized_evecs().keys()])
                                  + np.matmul(pts, self.normalized_evecs().samples.T / self.n_sigma)).T)

        phys_samples = trim_unphysical(sample_pts)
        if phys_samples.number_of_samples > npts:
            phys_samples = phys_samples.downsample(npts)

        return phys_samples


def generate_spin_z(samples):
    """
    Generate z-component spins from chi_eff
    :param samples: SamplesDict with PE samples containing chi_eff
    :return new_samples: SamplesDict with spin z-components
    """
    if any(_ in samples.keys() for _ in ["chi_eff", "chi_align"]):
        param = "chi_eff" if "chi_eff" in samples.keys() else "chi_align"
        # put chi_eff/chi_align on both BHs
        new_samples = SamplesDict(samples.keys() + ['spin_1z', 'spin_2z'],
                              np.append(samples.samples, np.array([samples[param], samples[param]]), 0))
    else:
        print("Need to specify 'chi_eff'")
        return -1
    return new_samples


def generate_prec_spin(samples):
    """
    Generate component spins from chi_eff and chi_p
    :param samples: SamplesDict with PE samples containing chi_eff and chi_p
    :return new_samples: SamplesDict with spin components where chi_eff goes on both BH, chi_p on BH 1
    """
    if ('chi_eff' not in samples.keys()) and ('chi_align' not in samples.keys()):
        print("Need to specify 'chi_eff' or 'chi_align'")
        return -1
    if ('chi_p' not in samples.keys()):
        print("Need to specify 'chi_p'")
        return -1

    param = "chi_eff" if "chi_eff" in samples.keys() else "chi_align"

    a_1 = np.sqrt(samples["chi_p"] ** 2 + samples[param] ** 2)
    a_2 = np.abs(samples[param])
    tilt_1 = np.arctan2(samples["chi_p"], samples[param])
    tilt_2 = np.arccos(np.sign(samples[param]))
    phi_12 = np.zeros_like(a_1)
    phi_jl = np.zeros_like(a_1)
    new_samples = SamplesDict(samples.keys() + ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl'],
                              np.append(samples.samples, np.array([a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl]), 0))

    return new_samples


def make_waveform(x, df, f_low, flen, approximant="IMRPhenomD"):
    """
    This function makes a waveform for the given parameters and
    returns h_plus generated at value x.

    :param x: dictionary with parameter values for waveform generation
    :param df: frequency spacing of points
    :param f_low: low frequency cutoff
    :param flen: length of the frequency domain array to generate
    :param approximant: the approximant generator to use
    :return h_plus: waveform at parameter space point x
    """
    if 'phase' not in x.keys():
        x['phase'] = np.zeros_like(list(x.values())[0])
    x['f_ref'] = f_low * np.ones_like(list(x.values())[0])

    modes = waveform_modes.mode_array('22', approximant)
    data = convert(x, disable_remnant=True)

    if 'chi_p' in data.keys():
        # generate the leading harmonic of the precessing waveform
        data = generate_prec_spin(data)
        data.generate_all_posterior_samples(disable_remnant=True)
        h_plus = conversions.snr._calculate_precessing_harmonics(data["mass_1"][0], data["mass_2"][0],
                                                                 data["a_1"][0], data["a_2"][0],
                                                                 data["tilt_1"][0], data["tilt_2"][0],
                                                                 data["phi_12"][0],
                                                                 data["beta"][0], data["distance"][0],
                                                                 harmonics=[0], approx=approximant,
                                                                 mode_array=modes,
                                                                 df=df, f_low=f_low,
                                                                 f_ref=data["f_ref"][0])[0]

    else:
        if ('spin_1z' not in data.keys()) or ('spin_2z' not in data.keys()):
            data = generate_spin_z(data)

        h_plus, h_cross = get_fd_waveform(mass1=data['mass_1'], mass2=data['mass_2'],
                                          spin1z=data['spin_1z'],
                                          spin2z=data['spin_2z'],
                                          delta_f=df, distance=data['distance'], f_lower=f_low,
                                          approximant=approximant,
                                          mode_array=modes)

    h_plus.resize(flen)
    return h_plus


def make_offset_waveform(x, dx, scaling, df, f_low, flen, approximant="IMRPhenomD"):
    """
    This function makes a waveform for the given parameters and
    returns h_plus generated at value (x + scaling * dx).

    :param x: dictionary with parameter values for initial point
    :param dx: dictionary with parameter variations (can be a subset of the parameters in x)
    :param scaling: the scaling to apply to dx
    :param df: frequency spacing of points
    :param f_low: low frequency cutoff
    :param flen: length of the frequency domain array to generate
    :param approximant: the approximant generator to use
    :return h_plus: waveform at parameter space point x + scaling * dx
    """
    tmp_x = copy.deepcopy(x)

    for k, dx_val in dx.items():
        tmp_x[k] += float(scaling * dx_val)

    h_plus = make_waveform(tmp_x, df, f_low, flen, approximant)

    return h_plus


param_mins = {'chirp_mass': 1.,
              'total_mass': 2.,
              'mass_1': 1.,
              'mass_2': 1.,
              'symmetric_mass_ratio': 0.04,
              'chi_eff': -0.98,
              'chi_align': -0.7,
              'chi_p': 0.,
              'spin_1z': -0.98,
              'spin_2z': -0.98
              }

param_maxs = {'chirp_mass': 1e4,
              'total_mass': 1e4,
              'mass_1': 1e4,
              'mass_2': 1e4,
              'symmetric_mass_ratio': 0.25,
              'chi_eff': 0.98,
              'chi_align': 0.7,
              'chi_p': 0.98,
              'spin_1z': 0.98,
              'spin_2z': 0.98
              }


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
        mins = param_mins

    if maxs is None:
        maxs = param_maxs

    alpha = 1.

    for k, dx_val in dx.items():
        if k in maxs and (x[k] + scaling * dx[k]) < mins[k]:
            alpha = min(alpha, (x[k] - mins[k]) / abs(scaling * dx[k]))
        if k in mins and (x[k] + scaling * dx[k]) > maxs[k]:
            alpha = min(alpha, (maxs[k] - x[k]) / abs(scaling * dx[k]))

    return alpha


def trim_unphysical(samples, maxs=None, mins=None):
    """
    Trim unphysical points from a SamplesDict

    :param samples: SamplesDict containing sample points
    :param maxs: the maximum permitted values of the physical parameters
    :param mins: the minimum physical values of the physical parameters
    :return physical_samples: SamplesDict with points outside the param max and min given
    """
    if mins is None:
        mins = param_mins

    if maxs is None:
        maxs = param_maxs

    keep = np.ones(samples.number_of_samples, bool)
    for d, v in samples.items():
        if d in maxs:
            keep *= (v < maxs[d])
        if d in mins:
            keep *= (v > mins[d])

    return SamplesDict(samples.keys(), samples.samples[:, keep])


def scale_match(m_alpha, alpha):
    """
    A function to scale the match calculated at an offset alpha to the
    match at unit offset

    :param m_alpha: the match at an offset alpha
    :param alpha: the value of alpha
    :return m: the match at unit offset
    """
    m = (alpha ** 2 - 1 + m_alpha) / alpha ** 2

    return m


def average_mismatch(x, dx, scaling, f_low, psd,
                     approximant="IMRPhenomD", verbose=False):
    """
    This function calculates the average match for steps of +dx and -dx
    It also takes care of times when one of the steps moves beyond the
    edge of the physical parameter space

    :param x : dictionary with the initial point
    :param dx: dictionary with parameter variations
    :param scaling: the scaling to apply to dx
    :param f_low: low frequency cutoff
    :param psd: the power spectrum to use in calculating the match
    :param approximant: the approximant generator to use
    :param verbose: print debugging information
    :return m: The average match from steps of +/- scaling * dx
    """
    a = {}
    m = {}
    h0 = make_waveform(x, psd.delta_f, f_low, len(psd), approximant)
    for s in [1., -1.]:
        a[s] = check_physical(x, dx, s * scaling)
        h = make_offset_waveform(x, dx, s * a[s] * scaling, psd.delta_f, f_low, len(psd), approximant)
        m[s] = match(h0, h, psd, low_frequency_cutoff=f_low, subsample_interpolation=True)[0]
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


def scale_dx(x, dx, desired_mismatch, f_low, psd,
             approximant="IMRPhenomD", tolerance=1e-2):
    """
    This function scales the input vectors so that the mismatch between
    a waveform at point x and one at x + v[i] is equal to the specified
    mismatch, up to the specified tolerance.

    :param x: dictionary with parameter values for initial point
    :param dx: a dictionary with the initial change in parameters
    :param desired_mismatch: the desired mismatch (1 - match)
    :param f_low: low frequency cutoff
    :param psd: the power spectrum to use in calculating the match
    :param approximant: the approximant generator to use
    :param tolerance: the maximum fractional error in the mismatch
    :return scale: The required scaling of dx to achieve the desired mismatch
    """
    opt = optimize.root_scalar(lambda a: average_mismatch(x, dx, a, f_low, psd,
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
    :param max_iter: the maximum number of iterations
    :return g: the mismatch metric in physical space
    """
    # initial directions and initial spacing
    ndim = len(dx_directions)
    n_sigmasq = chi2.isf(0.1, ndim)
    desired_mismatch = n_sigmasq / (2 * snr ** 2)
    g = Metric(x, dx_directions, desired_mismatch, f_low, psd, approximant, tolerance,
               n_sigma=np.sqrt(n_sigmasq))
    g.iteratively_update_metric(max_iter)
    return g


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

    h = make_waveform(s, psd.delta_f, f_low, len(psd), approximant)

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

    h = make_waveform(s, psd.delta_f, f_low, len(psd), approximant)

    m = match(data, h, psd, low_frequency_cutoff=f_low, subsample_interpolation=True)[0]

    return np.log10(1-m)


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
        bounds = [(param_mins[k], param_maxs[k]) for k in dx_directions]
        x0 = np.array([x[k] for k in dx_directions])
        fixed_pars = {k: v for k, v in x.items() if k not in dx_directions}

        # out = optimize.minimize(_neg_wf_match, x0,
        #                         args=(dx_directions, data, f_low, psd, approximant, fixed_pars),
        #                         bounds=bounds)
        #
        # x = dict(zip(dx_directions, out.x))
        # m_peak = -out.fun

        out = optimize.minimize(_log_wf_mismatch, x0,
                                args=(dx_directions, data, f_low, psd, approximant, fixed_pars),
                                bounds=bounds)

        x = dict(zip(dx_directions, out.x))
        m_peak = 1-10**(out.fun)

    elif method == 'metric':
        g = Metric(x, dx_directions, mismatch, f_low, psd, approximant, tolerance)
        g.iteratively_update_metric()

        while True:
            h = make_waveform(x, g.psd.delta_f, g.f_low, len(g.psd), g.approximant)

            m_0, _ = match(data, h, g.psd, low_frequency_cutoff=g.f_low, subsample_interpolation=True)
            matches = np.zeros([g.ndim, 2])
            alphas = np.zeros([g.ndim, 2])

            for i in range(g.ndim):
                for j in range(2):
                    alphas[i, j] = check_physical(x, g.normalized_evecs()[i:i + 1], (-1) ** j)
                    h = make_offset_waveform(x, g.normalized_evecs()[i:i + 1], alphas[i, j] * (-1) ** j,
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
        delta_x = SamplesDict(dx_directions, np.matmul(g.normalized_evecs().samples, s))
        alpha = check_physical(x, delta_x, 1)

        h = make_offset_waveform(x, delta_x, alpha, psd.delta_f, f_low, len(psd), approximant)
        m_peak = match(data, h, psd, low_frequency_cutoff=f_low,
                       subsample_interpolation=True)[0]

        for k, dx_val in delta_x.items():
            x[k] += float(alpha * dx_val)

    return x, m_peak

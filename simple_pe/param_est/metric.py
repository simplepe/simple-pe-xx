import numpy as np
from pycbc.filter import match
from scipy import optimize
from scipy.stats import chi2
from simple_pe.param_est.pe import SimplePESamples
from pesummary.utils.samples_dict import SamplesDict
from simple_pe.waveforms import waveform


class Metric:
    """
    A class to store the parameter space metric at a point x in parameter space,
    where the variations are given by dx_directions

    :param x: dictionary with parameter values for initial point
    :param dx_directions: a list of directions to vary
    :param mismatch: the mismatch value to use when calculating metric
    :param f_low: low frequency cutoff
    :param psd: the power spectrum to use in calculating the match
    :param approximant: the approximant generator to use
    :param tolerance: tolerance for scaling vectors
    :param prob: probability to enclose within ellipse
    :param snr: snr of signal, used in scaling mismatch
    """

    def __init__(self, x, dx_directions, mismatch, f_low, psd,
                 approximant="IMRPhenomD", tolerance=1e-2, prob=None, snr=None):
        """
        """
        self.x = SimplePESamples(x)
        if 'distance' not in self.x:
            self.x['distance'] = np.ones_like(list(x.values())[0])
        self.dx_directions = dx_directions
        self.ndim = len(dx_directions)
        self.dxs = SimplePESamples(SamplesDict(self.dx_directions, np.eye(self.ndim)))
        self.mismatch = mismatch
        self.f_low = f_low
        self.psd = psd
        self.approximant = approximant
        self.tolerance = tolerance
        self.prob = prob
        self.snr = snr
        if self.prob:
            self.n_sigma = np.sqrt(chi2.isf(1 - self.prob, self.ndim))
            if self.snr:
                self.mismatch = self.n_sigma ** 2 / (2 * self.snr ** 2)
        self.coordinate_metric = None
        self.metric = None
        self.evals = None
        self.evec = None
        self.err = None
        self.projected_directions = None
        self.projected_metric = None
        self.projection = None
        self.projected_mismatch = None

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

        self.dxs = SimplePESamples(SamplesDict(self.dx_directions, self.dxs.samples * scale))

    def calculate_metric(self):
        """
        Calculate the metric using the existing basis vectors, stored as self.dxs.
        """
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
        Calculate the metric in physical coordinates
        """
        dx_inv = np.linalg.inv(self.dxs.samples)
        self.metric = np.matmul(dx_inv.T, np.matmul(self.coordinate_metric, dx_inv))

    def calculate_evecs(self):
        """
        Calculate the eigenvectors and eigenvalues of the metric gij
        """
        self.evals, self.evec = np.linalg.eig(self.metric)

    def normalized_evecs(self):
        """
        Return the eigenvectors normalized to give the desired mismatch
        """
        if self.evec is None:
            self.calculate_evecs()
        # always force to be positive to avoid negatives in sqrt
        self.evals[self.evals < 0] = np.abs(self.evals[self.evals < 0])
        return SimplePESamples(SamplesDict(self.dx_directions, self.evec *
                                           np.sqrt(self.mismatch / self.evals)))

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
        Re-calculate the metric gij based on the matches obtained
        for the eigenvectors of the original metric
        """
        # calculate the eigendirections of the matrix
        self.calculate_evecs()
        self.dxs = SimplePESamples(SamplesDict(self.dx_directions, self.evec))
        # scale so that they actually have the desired mismatch
        self.scale_dxs()
        self.calculate_metric()

    def iteratively_update_metric(self, max_iter=20, verbose=True):
        """
        A method to re-calculate the metric gij based on the matches obtained
        for the eigenvectors of the original metric
        :param max_iter: maximum number of iterations
        :param verbose: print information messages during update
        """
        tol = float(self.tolerance * self.mismatch)
        self.calc_metric_error()

        op = 0
        if verbose:
            from tqdm import tqdm
            base_desc = "Calculating the metric | iteration {} < {} | error {:.2g} > {:.2g}"
            pbar = tqdm(
                np.arange(max_iter), bar_format="{desc}",
                desc=base_desc.format(op, max_iter, self.err, tol)
            )
        while (self.err > tol) and (op < max_iter):
            self.update_metric()
            self.calc_metric_error()
            op += 1
            if verbose:
                pbar.set_description(base_desc.format(op, max_iter, self.err, tol))
                pbar.update(1)

        if self.err > tol:
            pbar.set_description(
                f"Failed to achieve requested tolerance.  Requested: {tol:.2g} "
                f"achieved {self.err:.2g}"
            )

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

        if self.prob and self.snr:
            # calculate equivalent mismatch given number of remaining dimensions
            n_sigmasq = chi2.isf(1 - self.prob, len(projected_directions))
            self.projected_mismatch = n_sigmasq / (2 * self.snr ** 2)

    def projected_evecs(self):
        """
        Return the evecs normalized to give the desired mismatch
        """
        evals, evec = np.linalg.eig(self.projected_metric)
        return SimplePESamples(SamplesDict(self.projected_directions, evec * np.sqrt(self.mismatch / evals)))

    def generate_ellipse(self, npts=100, projected=False, scale=1.):
        """
        Generate an ellipse of points of the stored mismatch.  Scale the radius by a factor `scale'
        If the metric is projected, and we know the snr, then scale the mismatch appropriately
        for the number of projected dimensions.
        :param npts: number of points in the ellipse
        :param projected: use the projected metric if True, else use metric
        :param scale: scaling to apply to the mismatch
        :return ellipse_dict: SimplePESamples with ellipse of points
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

        if projected and self.projected_mismatch:
            scale *= np.sqrt(self.projected_mismatch / self.mismatch)

        # generate points on a circle
        phi = np.linspace(0, 2 * np.pi, npts)
        xx = scale * np.cos(phi)
        yy = scale * np.sin(phi)
        pts = np.array([xx, yy])

        # project onto eigendirections
        dx = SimplePESamples(SamplesDict(dx_dirs, np.matmul(n_evec.samples, pts)))

        # scale to be physical
        alphas = np.zeros(npts)
        for i in range(npts):
            alphas[i] = waveform.check_physical(self.x, dx[i:i + 1], 1.)

        ellipse_dict = SimplePESamples(SamplesDict(dx_dirs,
                                                   np.array([self.x[dx] for dx in dx_dirs]).reshape(
                                                       [2, 1]) + dx.samples * alphas))

        return ellipse_dict

    def generate_match_grid(self, npts=10, projected=False, scale=1.):
        """
        Generate a grid of points with extent governed by stored mismatch.  
        If the metric is projected, scale to the projected number of dimensions
        Apply an overall scaling of scale

        :param npts: number of points in each dimension of the grid
        :param projected: use the projected metric if True, else use metric
        :param scale: a factor by which to scale the extent of the grid
        :return ellipse_dict: SimplePESamples with grid of points and match at each point
        """
        if projected:
            dx_dirs = self.projected_directions + [x for x in self.dx_directions if x not in self.projected_directions]
            n_evec = self.projected_evecs()
        else:
            dx_dirs = self.dx_directions
            n_evec = self.normalized_evecs()

        if projected and self.projected_mismatch:
            scale *= np.sqrt(self.projected_mismatch / self.mismatch)

        grid = np.mgrid[-scale:scale:npts * 1j, -scale:scale:npts * 1j]
        dx_data = np.tensordot(n_evec.samples, grid, axes=(1, 0))

        if projected:
            dx_extra = np.tensordot(self.projection, dx_data, axes=(1, 0))
            dx_data = np.append(dx_data, dx_extra, 0)

        dx = SimplePESamples(SamplesDict(dx_dirs, dx_data.reshape(len(dx_dirs), npts ** 2)))

        h0 = waveform.make_waveform(self.x, self.psd.delta_f, self.f_low, len(self.psd), self.approximant)

        m = np.zeros(dx.number_of_samples)
        for i in range(dx.number_of_samples):
            if waveform.check_physical(self.x, dx[i:i + 1], 1.) < 1:
                m[i] = 0
            else:
                h1 = waveform.make_offset_waveform(self.x, dx[i:i + 1], 1.,
                                          self.psd.delta_f, self.f_low, len(self.psd), self.approximant)
                m[i] = match(h0, h1, self.psd, self.f_low, subsample_interpolation=True)[0]

        matches = SimplePESamples(SamplesDict(dx_dirs + ['match'],
                                              np.append(
                                                  dx.samples + np.array([self.x[dx] for dx in dx_dirs]).reshape(
                                                      [len(dx_dirs), 1]),
                                                  m.reshape([1, npts ** 2]), 0).reshape(
                                                  [len(dx_dirs) + 1, npts, npts])))

        return matches

    def generate_posterior_grid(self, npts=10, projected=False, scale=None):
        """
        Generate a grid of points with extent governed by requested mismatch

        :param npts: number of points in each dimension of the grid
        :param projected: use the projected metric if True, else use metric
        :param scale: the scale to apply to the greatest extent of the grid
        :return ellipse_dict: SimplePESamples with grid of points and match at each point
        """
        if not self.snr:
            print("need an SNR to turn matches into probabilities")
            return -1

        matches = self.generate_match_grid(npts, projected, scale)
        post = np.exp(-self.snr ** 2 / 2 * (1 - matches['match'] ** 2))

        grid_probs = SimplePESamples(SamplesDict(matches.keys() + ['posterior'],
                                                 np.append(matches.samples, post).reshape(
                                                     [len(matches.keys()) + 1, npts, npts])))

        return grid_probs

    def generate_samples(self, npts=int(1e5)):
        """
        Generate a given number of samples

        :param npts: number of points to generate
        :return phys_samples: SimplePESamples with samples
        """
        sample_pts = self._get_samples(2 * npts)

        while sample_pts.number_of_samples < npts:
            extra_pts = self._get_samples(npts)
            sample_pts = SimplePESamples(SamplesDict(sample_pts.keys(),
                np.concatenate((sample_pts.samples.T, extra_pts.samples.T)).T))

        if sample_pts.number_of_samples > npts:
            sample_pts = sample_pts.downsample(npts)
        return sample_pts

    def _get_samples(self, npts=int(1e5)):
        """
        Generate an ellipse of points of constant mismatch

        :param npts: number of points to generate
        :return phys_samples: SimplePESamples with samples
        """
        pts = np.random.normal(0, 1, [npts, self.ndim])

        sample_pts = SimplePESamples(SamplesDict(self.dx_directions,
            (np.array([self.x[dx] for dx in self.normalized_evecs().keys()
                       ]).reshape([len(self.dx_directions), 1])
             + np.matmul(pts, self.normalized_evecs().samples.T /
                         self.n_sigma).T)))
        sample_pts.trim_unphysical()

        return sample_pts


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
    h0 = waveform.make_waveform(x, psd.delta_f, f_low, len(psd), approximant)
    for s in [1., -1.]:
        a[s] = waveform.check_physical(x, dx, s * scaling)
        try:
            h = waveform.make_offset_waveform(x, dx, s * a[s] * scaling,
                                              psd.delta_f, f_low, len(psd),
                                              approximant)
            m[s] = match(h0, h, psd, low_frequency_cutoff=f_low,
                         subsample_interpolation=True)[0]
        except RuntimeError:
            m[s] = 1e-5

    if verbose:
        print("Had to scale steps to %.2f, %.2f" % (a[-1], a[1]))
        print("Mismatches %.3g, %.3g" % (1 - m[-1], 1 - m[1]))
    if min(a.values()) < 1e-2:
        if verbose:
            print("we're really close to the boundary," 
                  "so down-weight match contribution")
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


def find_metric_and_eigendirections(x, dx_directions, snr, f_low, psd,
                                    approximant="IMRPhenomD",
                                    tolerance=0.05, max_iter=20):
    """
    Calculate the eigendirections in parameter space, normalized to enclose a
    90% confidence region at the requested SNR

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
    g = Metric(x, dx_directions, 0, f_low, psd, approximant, tolerance,
               prob=0.9, snr=snr)
    g.iteratively_update_metric(max_iter)
    return g

import numpy as np
from simple_pe import fstat
from scipy import special
from pesummary.core.reweight import rejection_sampling


def evec_sigma(m):
    """
    Calculate the eigenvalues and vectors of the localization matrix M.
    sigma is defined as the reciprocal of the square-root eigenvalue.
    Definitions from "Triangulation of gravitational wave sources with a
    network of detectors", New J. Phys. 11 123006.

    Parameters
    ----------
    m: np.array
        square matrix for which we calculate the eigen-vectors and sigmas

    Returns
    -------
    evec: np.array
        An array of eigenvectors of M, giving principal directions for
        localization
    sigma: np.array
        localization accuracy along eigen-directions.
    """
    ev, evec = np.linalg.eig(m)
    epsilon = 1e-10
    sigma = 1 / np.sqrt(ev + epsilon)
    evec = evec[:, sigma.argsort()]
    sigma.sort()
    return evec, sigma


def project_to_sky(x, y, event_xyz, gmst, evec, ellipse=False,
                   sky_weight=False):
    """
    Project a set of points onto the sky.

    Parameters
    ----------
    x: np.array
        x coordinates of points (relative to sky location)
    y: np.array
        y coordinate of points (relative to sky location)
    event_xyz: np.array
        xyz location of event
    gmst: float
        gmst of event
    evec: np.array
        localization eigenvectors
    ellipse: bool
        is this an ellipse
    sky_weight: bool
        re-weight to uniform on sky

    Returns
    -------
    phi: np.array
        phi coordinate of points
    theta: np.array
        theta coordinate of points
     """
    # check that we're not going outside the unit circle
    # if we are, first roll this to the beginning then truncate
    bad = x ** 2 + y ** 2 > 1
    if ellipse and sum(bad.flatten()) > 0:
        # ensure we have a continuous shape
        x = np.roll(x, -bad.argmax())
        y = np.roll(y, -bad.argmax())
        bad = np.roll(bad, -bad.argmax())

    x = x[~bad]
    y = y[~bad]

    z = np.sqrt(1 - x ** 2 - y ** 2) * np.sign(np.inner(event_xyz, evec[:, 2]))

    if sky_weight:
        weights = abs(1./z)
        weights /= weights.max()
        x, y, z = rejection_sampling(np.array([x, y, z]).T, weights).T

    # if we hit the edge then the localization region is 2 parts,
    # above and below z=0 surface
    if sum(bad.flatten()) > 0:
        x = np.append(np.concatenate((x, x[::-1])), x[0])
        y = np.append(np.concatenate((y, y[::-1])), y[0])
        z = np.append(np.concatenate((z, -z[::-1])), z[0])

    x_net = evec[:, 0]
    y_net = evec[:, 1]
    z_net = evec[:, 2]

    r = {}
    for i in range(3):
        r[i] = x * x_net[i] + y * y_net[i] + z * z_net[i]
    theta = np.arcsin(r[2] / np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2))
    phi = (np.arctan2(r[1], r[0]) + gmst) % (2 * np.pi)

    return phi, theta


class Localization(object):
    """
    class to hold the details and results of localization based on a
    given method
    """

    def __init__(self, method, event, mirror=False, p=0.9, d_max=1000, area=0):
        """
        Initialization

        Parameters
        ----------
        method: str
            how we do localization, one of "time", "coh",
            "left, "right", "marg"
        event: event.Event
            details of event
        mirror: bool
            are we looking in the mirror location
        p: float
            probability
        d_max: float
            maximum distance to consider
        area: float
            localization area
        """
        self.method = method
        self.mirror = mirror
        self.event = event
        self.p = p
        self.snr = 0
        self.z = None
        self.area = area
        self.M = None
        self.sigma = None
        self.evec = None
        self.PMP = None
        self.dt_i = None
        if method != "marg":
            self.calculate_m()
            self.calculate_max_snr()
            if self.M is not None:
                self.sky_project()
                self.calc_area()
            else:
                self.area = 1e6

            A = fstat.snr_f_to_a(self.z, self.event.get_fsig(mirror))
            D, cosi, _, _ = fstat.a_to_params(A)
            self.D = float(D)
            self.cosi = float(cosi)

        self.like = 0.
        if method != "time" and method != "marg":
            self.approx_like(d_max)

    def calculate_m(self):
        """
        Calculate the localization matrix as given in
        "Localization of transient gravitational wave sources:
        beyond triangulation", Class. Quantum Grav. 35 (2018) 105002
        this is a generalization of the timing based localization
        """
        a_i, c_ij, c_i, c = self.event.localization_factors(self.method,
                                                            self.mirror)
        cc = 1. / 2 * (np.outer(c_i, c_i) / c - c_ij)
        # Calculate the Matrix M (given in the coherent localization paper)
        m = np.zeros([3, 3])
        locations = self.event.get_data("location")

        for i1 in range(len(self.event.ifos)):
            for i2 in range(len(self.event.ifos)):
                m += np.outer(locations[i1] - locations[i2],
                              locations[i1] - locations[i2]) / 3e8 ** 2 \
                     * cc[i1, i2]
        self.M = m

    def calculate_max_snr(self):
        """
        Calculate the maximum SNR nearby the given point.
        For this, calculate the localization factors and the SNR projection
        and see whether there is a higher network SNR at a nearby
        point.  For timing, the peak will be at the initial point, but for
        coherent or left/right circular polarizations, the peak can be offset.
        """
        self.z = self.event.projected_snr(self.method, self.mirror)
        a_i, c_ij, c_i, c = self.event.localization_factors(self.method,
                                                            self.mirror)
        try:
            self.dt_i = 1. / 2 * np.inner(np.linalg.inv(c_ij), a_i)
        except:
            print("for method %s: Unable to invert C, setting dt=0" %
                  self.method)
            self.dt_i = np.zeros_like(a_i)
        z = self.event.projected_snr(self.method, self.mirror)
        f_band = self.event.get_data("f_band")
        if max(abs(self.dt_i * (2 * np.pi * f_band))) < 1. / np.sqrt(2):
            extra_snr = np.inner(self.dt_i, np.inner(c_ij, self.dt_i))
            if extra_snr > 0:
                # location of second peak is reasonable -- use it
                z = self.event.projected_snr(self.method, self.mirror,
                                             self.dt_i)
        self.z = z
        self.snr = np.linalg.norm(z)

    def calculate_dt_0(self, dt_i):
        """
        Calculate the overall time offset at a point offset from the source
        point by dt_i that maximizes the overall SNR

        Parameters
        ----------
        dt_i: np.array
            The time offsets at which to evaluate the SNR

        Returns
        -------
        dt_0: float
            The overall time offset that maximizes the SNR
        """
        a_i, c_ij, c_i, c = self.event.localization_factors(self.method,
                                                            self.mirror)
        a = sum(a_i)
        dt_0 = a / (2 * c) - np.inner(c_i, dt_i) / c
        return dt_0

    def calculate_snr(self, dt_i):
        """
        Calculate the SNR at a point offset from the source point
        dt_i is an array of time offsets for the detectors
        Note: we maximize over the overall time offset dt_0.
        If the time offsets are too large to trust the leading order
        approximation, then return the original SNR

        Parameters
        ----------
        dt_i: np.array
            The time offsets at which to evaluate the SNR

        Returns
        -------
        z: np.array
            The individual SNRs at the offset time, consistent with a signal
            (either coherent or left/right circular)
        snr: float
            the network SNR consistent with a signal
        """
        # See if the calculation is valid:
        f_band = self.event.get_data("f_band")
        if max(abs(dt_i * (2 * np.pi * f_band))) > 1. / np.sqrt(2):
            # location of second peak is outside linear regime, return zero
            z = np.zeros_like(dt_i)
        else:
            z = self.event.projected_snr(self.method, self.mirror, dt_i)
        snr = np.linalg.norm(z)
        return z, snr

    def approx_like(self, d_max=1000):
        """
        Calculate the approximate likelihood, based on equations A.18 and
        A.29 from "Localization of transient gravitational wave sources:
        beyond triangulation", Class. Quantum Grav. 35 (2018) 105002.

        Parameters
        ----------
        d_max: float
            maximum distance, used for normalization
        """
        if self.snr == 0:
            self.like = 0
            return
        Fp, Fc = self.event.get_f(self.mirror)
        self.like = self.snr ** 2 / 2
        if (self.method == "left") or (self.method == "right"):
            if Fc == 0:
                cosf = 0.5
            else:
                cos_fac = np.sqrt((Fp ** 2 + Fc ** 2) / (Fp * Fc))
                cosf = min(cos_fac / np.sqrt(self.snr), 0.5)
            self.like += np.log((self.D / d_max) ** 3 / self.snr ** 2 * cosf)
        elif Fc != 0:
            # not sensitive to second polarization, so can't do integral
            self.like += np.log(32. * (self.D / d_max) ** 3 * self.D ** 4 /
                                (Fp ** 2 * Fc ** 2) / (1 - self.cosi ** 2) ** 3)

    def sky_project(self):
        """
        Project localization matrix to zero out components in direction of
        source. This is implementing equations 10 and 11 from "Source
        localization with an advanced gravitational wave detector network",
        DOI 10.1088/0264-9381/28/10/10502
        """
        if self.mirror:
            source = self.event.mirror_xyz
        else:
            source = self.event.xyz
        p = np.identity(3) - np.outer(source, source)
        self.PMP = np.inner(np.inner(p, self.M), p)
        self.evec, self.sigma = evec_sigma(self.PMP)

    def calc_area(self):
        """
        Calculate the localization area using equations (12) and (15) of
        "Source localization with an advanced gravitational wave detector
        network", DOI 10.1088/0264-9381/28/10/10502
        """
        # calculate the area of the ellipse
        ellipse = - np.log(1. - self.p) * 2 * np.pi * (180 / np.pi) ** 2 * \
                  self.sigma[0] * self.sigma[1]
        # calculate the area of the band
        band = 4 * np.pi * (180 / np.pi) ** 2 * np.sqrt(2) * \
               special.erfinv(self.p) * self.sigma[0]
        # use the minimum (that's not nan)
        self.area = np.nanmin((ellipse, band))

    def make_ellipse(self, npts=101, scale=1.0):
        """
        Generate points that lie on the localization ellipse

        Parameters
        ----------
        npts: int
            number of points
        scale: float
            factor by which to scale the ellipse.  Default (scaling of 1) is to
            use the probability stored in the localization.

        Returns
        -------
        points: np.array
            a set of points marking the boundary of the localization ellipse
            the points are projected onto the sky (so in theta/phi coordinates)
        """
        scale *= np.sqrt(- 2 * np.log(1 - self.p))

        # check if the event is localized (in xyz),
        # if so, use the projected matrix, if not, don't project
        evec, sigma = evec_sigma(self.M)
        if sigma[2] < 1:
            evec = self.evec
            sigma = self.sigma

        # set up co-ordinates
        ang = np.linspace(0, 2 * np.pi, npts)

        # normalization to get area for given p-value:
        if self.mirror:
            xyz = self.event.mirror_xyz
        else:
            xyz = self.event.xyz

        x = np.inner(xyz, evec[:, 0]) + scale * sigma[0] * np.cos(ang)
        y = np.inner(xyz, evec[:, 1]) + scale * sigma[1] * np.sin(ang)

        return project_to_sky(x, y, xyz, self.event.gmst, evec, ellipse=True)

    def generate_loc_grid(self, npts=10, scale=1.):
        """
        Generate a grid of points with extent governed by the localization

        Parameters
        ----------
        npts: int
            number of points in each dimension of the grid
        scale: float
            factor by which to scale grid, relative to default given by the
            localization eigenvectors
        Returns
        -------
        grid_points: np.array
            containing a grid of points (theta and phi)
        """
        scale *= np.sqrt(- 2 * np.log(1 - self.p))

        evec, sigma = evec_sigma(self.M)
        if sigma[2] < 1:
            evec = self.evec
            sigma = self.sigma

        # set up co-ordinates
        grid = np.mgrid[-scale:scale:npts * 1j, -scale:scale:npts * 1j]

        # normalization to get area for given p-value:
        if self.mirror:
            xyz = self.event.mirror_xyz
        else:
            xyz = self.event.xyz

        x = np.inner(xyz, evec[:, 0]) + scale * sigma[0] * grid[0]
        y = np.inner(xyz, evec[:, 1]) + scale * sigma[1] * grid[1]

        return project_to_sky(x, y, xyz, self.event.gmst,evec)

    def generate_samples(self, npts=int(1e5), sky_weight=True):
        """
        Generate a set of samples based on Gaussian distribution in
        localization eigendirections

        Parameters
        ----------
        npts: int
            number of points to generate
        sky_weight: bool
            weight points to be uniform on the sky, rather than uniform over
            the localization ellipse

        Returns
        -------
         samples: phi, theta samples
        """
        if sky_weight:
            safety_fac = 10
        else:
            safety_fac = 2

        pts = np.random.normal(0, 1, [int(safety_fac * npts), 2])

        evec, sigma = evec_sigma(self.M)
        if sigma[2] < 1:
            evec = self.evec
            sigma = self.sigma

        # normalization to get area for given p-value:
        if self.mirror:
            xyz = self.event.mirror_xyz
        else:
            xyz = self.event.xyz

        x = np.inner(xyz, evec[:, 0]) + sigma[0] * pts[:, 0]
        y = np.inner(xyz, evec[:, 1]) + sigma[1] * pts[:, 1]

        phi, theta = project_to_sky(x, y, xyz, self.event.gmst, evec,
                                    sky_weight)

        if len(theta) > npts:
            keep = np.random.choice(len(theta), npts)
            theta = theta[keep]
            phi = phi[keep]

        else:
            print("Re-weighting resulted in fewer than requested trials")

        return phi, theta


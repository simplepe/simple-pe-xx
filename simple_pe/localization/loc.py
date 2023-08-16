import numpy as np
from simple_pe import fstat
from scipy import special
from pesummary.core.reweight import rejection_sampling


def evec_sigma(M):
    """
    Calculate the eigenvalues and vectors of M.
    sigma is defined as the reciprocal of the eigenvalue

    :param M: square matrix for which we calculate the eigen-vectors
    and sigmas
    """
    ev, evec = np.linalg.eig(M)
    epsilon = 1e-10
    sigma = 1 / np.sqrt(ev + epsilon)
    evec = evec[:, sigma.argsort()]
    sigma.sort()
    return evec, sigma


def project_to_sky(x, y, event_xyz, gmst, evec, ellipse=False,
                   sky_weight=False):
    """
     Project a set of points onto the sky.

     :param x: x coordinates of points (relative to sky location)
     :param y: y coordinate of points (relative to sky location)
     :param event_xyz: xyz location of event
     :param gmst: gmst of event
     :param evec: localization eigenvectors
     :param ellipse: is this an ellipse
     :param sky_weight: re-weight to uniform on sky
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
    class to hold the details and results of localization based on a g
    iven method
    """

    def __init__(self, method, event, mirror=False, p=0.9, d_max=1000, area=0):
        """
        Initialization

        param method: how we do localization, one of "time", "coh",
            "left, "right", "marg"
        :param event: details of event
        :param mirror: are we looking in the mirror location
        :param p: probability
        :param d_max: maximum distance to consider
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
        A_i, C_ij, c_i, c = self.event.localization_factors(self.method,
                                                            self.mirror)
        CC = 1. / 2 * (np.outer(c_i, c_i) / c - C_ij)
        # Calculate the Matrix M (given in the coherent localization paper)
        M = np.zeros([3, 3])
        locations = self.event.get_data("location")

        for i1 in range(len(self.event.ifos)):
            for i2 in range(len(self.event.ifos)):
                M += np.outer(locations[i1] - locations[i2],
                              locations[i1] - locations[i2]) / 3e8 ** 2 \
                     * CC[i1, i2]
        self.M = M

    def calculate_max_snr(self):
        self.z = self.event.projected_snr(self.method, self.mirror)
        A_i, C_ij, c_i, c = self.event.localization_factors(self.method,
                                                            self.mirror)
        try:
            self.dt_i = 1. / 2 * np.inner(np.linalg.inv(C_ij), A_i)
        except:
            print("for method %s: Unable to invert C, setting dt=0" %
                  self.method)
            self.dt_i = np.zeros_like(A_i)
        z = self.event.projected_snr(self.method, self.mirror)
        f_band = self.event.get_data("f_band")
        if max(abs(self.dt_i * (2 * np.pi * f_band))) < 1. / np.sqrt(2):
            extra_snr = np.inner(self.dt_i, np.inner(C_ij, self.dt_i))
            if extra_snr > 0:
                # location of second peak is reasonable -- use it
                z = self.event.projected_snr(self.method, self.mirror,
                                             self.dt_i)
        self.z = z
        self.snr = np.linalg.norm(z)

    def calculate_dt0(self, dt_i):
        """
        Calculate the SNR at a point offset from the source point
        dt_i is an array of time offsets for the detectors
        Note: we maximize over the overall time offset dt_o
        """
        A_i, C_ij, c_i, c = self.event.localization_factors(self.method,
                                                            self.mirror)
        a = sum(A_i)
        dt0 = a / (2 * c) - np.inner(c_i, dt_i) / c
        return dt0

    def calculate_snr(self, dt):
        # See if the calculation is valid:
        f_band = self.event.get_data("f_band")
        if max(abs(dt * (2 * np.pi * f_band))) > 1. / np.sqrt(2):
            # location of second peak is outside linear regime, return zero
            z = np.zeros_like(dt)
        else:
            z = self.event.projected_snr(self.method, self.mirror, dt)
        snr = np.linalg.norm(z)
        return z, snr

    def approx_like(self, d_max=1000):
        """
        Calculate the approximate likelihood, based on equations XXX

        :param d_max: maximum distance, used for normalization
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
        source. This is implementing equations 10 and 11 from the advanced
        localization paper
        """
        if self.mirror:
            source = self.event.mirror_xyz
        else:
            source = self.event.xyz
        P = np.identity(3) - np.outer(source, source)
        self.PMP = np.inner(np.inner(P, self.M), P)
        self.evec, self.sigma = evec_sigma(self.PMP)

    def calc_area(self):
        """
        Calculate the localization area
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
        Calculate the localization ellipse

        :param npts: number of points
        :param scale: factor by which to scale the ellipse
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

        :param npts: number of points in each dimension of the grid
        :param scale: factor by which to scale grid
        :return grid_dict: SimplePESamples with grid of points and
            match at each point
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

        :param npts: number of points to generate
        :param sky_weight: weight points to be uniform on the sky
        :return samples: phi, theta samples
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


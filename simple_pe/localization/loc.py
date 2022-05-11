import numpy as np
from simple_pe.fstat import fstat
from scipy import special


##################################################################
# Helper functions
##################################################################
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


##################################################################
# Class to store localization information
##################################################################
class Localization(object):
    """
    class to hold the details and results of localization based on a given method
    """

    def __init__(self, method, event, mirror=False, p=0.9, Dmax=1000, area=0):
        """
        Initialization

        param method: how we do localization, one of "time", "coh", "left, "right", "marg"
        :param event: details of event
        :param mirror: are we looking in the mirror location
        :param p: probability
        :param Dmax: maximum distance to consider
        """
        self.method = method
        self.mirror = mirror
        self.event = event
        self.p = p
        self.snr = 0
        self.area = area
        if method != "marg":
            self.calculate_m()
            self.calculate_max_snr()
            if self.M is not None:
                self.sky_project()
                self.calc_area()
            else:
                self.area = 1e6

            A = fstat.snr_f_to_a(self.z, self.event.get_fsig(mirror))
            self.D, self.cosi, _, _ = fstat.a_to_params(A)

        self.like = 0.
        if method != "time" and method != "marg":
            self.approx_like(Dmax)

    def calculate_m(self):
        """
        Calculate the localization matrix as given in
        "Localization of transient gravitational wave sources: beyond triangulation",
        Class. Quantum Grav. 35 (2018) 105002
        this is a generalization of the timing based localization
        """
        A_i, C_ij, c_i, c = self.event.localization_factors(self.method, self.mirror)
        CC = 1. / 2 * (np.outer(c_i, c_i) / c - C_ij)
        # Calculate the Matrix M (given in the coherent localization paper)
        M = np.zeros([3, 3])
        locations = self.event.get_data("location")

        for i1 in range(len(self.event.ifos)):
            for i2 in range(len(self.event.ifos)):
                M += np.outer(locations[i1] - locations[i2],
                           locations[i1] - locations[i2]) / (3e8) ** 2 \
                     * CC[i1, i2]
        self.M = M

    def calculate_max_snr(self):
        self.z = self.event.projected_snr(self.method, self.mirror)
        A_i, C_ij, c_i, c = self.event.localization_factors(self.method, self.mirror)
        try:
            self.dt_i = 1. / 2 * np.inner(np.linalg.inv(C_ij), A_i)
        except:
            print("for method %s: Unable to invert C, setting dt=0" % self.method)
            self.dt_i = np.zeros_like(A_i)
        z = self.event.projected_snr(self.method, self.mirror)
        f_band = self.event.get_data("f_band")
        if max(abs(self.dt_i * (2 * np.pi * f_band))) < 1. / np.sqrt(2):
            extra_snr = np.inner(self.dt_i, np.inner(C_ij, self.dt_i))
            if extra_snr > 0:
                # location of second peak is reasonable -- use it
                z = self.event.projected_snr(self.method, self.mirror, self.dt_i)
        self.z = z
        self.snr = np.linalg.norm(z)

    def calculate_dt0(self, dt_i):
        """
        Calculate the SNR at a point offset from the source point
        dt_i is an array of time offsets for the detectors
        Note: we maximize over the overall time offset dt_o
        """
        A_i, C_ij, c_i, c = self.event.localization_factors(self.method, self.mirror)
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

    def approx_like(self, Dmax=1000):
        """
        Calculate the approximate likelihood, based on equations XXX

        :param Dmax: maximum distance, used for normalization
        """
        if self.snr == 0:
            self.like = 0
            return
        Fp, Fc = self.event.get_f(self.mirror)
        self.like = self.snr ** 2 / 2
        if (self.method == "left") or (self.method == "right"):
            cos_fac = np.sqrt((Fp ** 2 + Fc ** 2) / (Fp * Fc))
            cosf = min(cos_fac / np.sqrt(self.snr), 0.5)
            self.like += np.log((self.D / Dmax) ** 3 / self.snr ** 2 * cosf)
        else:
            self.like += np.log(32. * (self.D / Dmax) ** 3 * self.D ** 4 / (Fp ** 2 * Fc ** 2)
                             / (1 - self.cosi ** 2) ** 3)

    def sky_project(self):
        """
        Project localization matrix to zero out components in direction of source
        This is implementing equations 10 and 11 from the advanced localization paper
        """
        if self.mirror == True:
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
        band = 4 * np.pi * (180 / np.pi) ** 2 * np.sqrt(2) * special.erfinv(self.p) * self.sigma[0]
        # use the minimum (that's not nan)
        self.area = np.nanmin((ellipse, band))

    def make_ellipse(self):
        """
        Calculate the localization ellipse
        """
        # check if the event is localized (in xyz), if so, use the projected matrix
        # if not, don't project
        evec, sigma = evec_sigma(self.M)
        if sigma[2] < 1:
            evec = self.evec
            sigma = self.sigma

        x_net = evec[:, 0]
        y_net = evec[:, 1]
        z_net = evec[:, 2]

        # set up co-ordinates
        r = {}
        ang = np.linspace(0, 2 * np.pi, 101)

        # normalization to get area for given p-value:
        x = np.inner(self.event.xyz, x_net) + \
            np.sqrt(- 2 * np.log(1 - self.p)) * sigma[0] * np.cos(ang)
        y = np.inner(self.event.xyz, y_net) + \
            np.sqrt(- 2 * np.log(1 - self.p)) * sigma[1] * np.sin(ang)

        # check that we're not going outside of unit circle
        # if we are, first roll this to the beginning then truncate
        bad = x ** 2 + y ** 2 > 1
        if sum(bad) > 0:
            x = np.roll(x, -bad.argmax())
            y = np.roll(y, -bad.argmax())
            bad = np.roll(bad, -bad.argmax())

            x = x[~bad]
            y = y[~bad]

        z = np.sqrt(1 - x ** 2 - y ** 2) * np.sign(np.inner(self.event.xyz, z_net))

        # if we hit the edge then the localization region is 2 parts, above and below z=0 surface
        if sum(bad) > 0:
            x = np.append(np.concatenate((x, x[::-1])), x[0])
            y = np.append(np.concatenate((y, y[::-1])), y[0])
            z = np.append(np.concatenate((z, -z[::-1])), z[0])

        for i in range(3):
            r[i] = x * x_net[i] + y * y_net[i] + z * z_net[i]
        theta = np.arcsin(r[2] / np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2))
        phi = np.arctan2(r[1], r[0])

        return phi, theta

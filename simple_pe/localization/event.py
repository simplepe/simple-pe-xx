import numpy as np
import random as rnd
import copy
from simple_pe.detectors import detectors
from simple_pe.localization import loc
import lal
from astropy.time import Time
from scipy.optimize import brentq
from scipy.special import logsumexp

##################################################################
# Helper functions
##################################################################
def snr_projection(f_sig, method):
    """
    Function to calculate the SNR projection matrix P for a given set
    of detector responses, f_sig

    :param f_sig: an Nx2 array of detector responses
    :param method: the way we project (one of "time", "coh", "left", "right")
    """
    if method == "time":
        P = np.identity(len(f_sig))
    elif method == "coh":
        M = np.zeros((2, 2))
        for f in f_sig:
            M += np.outer(f, f)
        P = np.inner(np.inner(f_sig, np.linalg.inv(M)), f_sig)
    elif method == "right":
        cf = np.array([complex(f[0], f[1]) for f in f_sig])
        P = np.outer(cf.conjugate(), cf) / np.inner(cf.conjugate(), cf)
    elif method == "left":
        cf = np.array([complex(f[0], f[1]) for f in f_sig])
        P = np.outer(cf, cf.conjugate()) / np.inner(cf, cf.conjugate())
    else:
        raise NameError("Invalid projection method: %s" % method)
    return P


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
# Class to store event information
##################################################################
class Event(object):
    """
    class to hold the events that we want to localize
    """

    def __init__(self, Dmax=0, gps=1000000000, params=None):
        """
        Initialize event

        :param Dmax: maximum distance to consider
        :param gmst: greenwich mean sidereal time
        :param params: parameters in form used by first 2 years paper
        """
        if params is not None:
            self.D = params["distance"]
            self.ra = np.radians(params["RAdeg"])
            self.dec = np.radians(params["DEdeg"])
            try:
                self.gps = params['gps']
            except:
                t = Time(params["MJD"], format='mjd')
                self.gps = lal.LIGOTimeGPS(int(t.gps), int(1e9 * (t.gps % 1)))
            self.gmst = lal.GreenwichMeanSiderealTime(self.gps)
            # self.gmst = float(t.sidereal_time("mean", "greenwich")/units.hourangle)
            self.phi = np.radians(params["coa-phase"])
            self.psi = np.radians(params["polarization"])
            self.cosi = np.cos(np.radians(params["inclination"]))
            self.mchirp = params["mass1"] ** (3. / 5) * params["mass2"] ** (3. / 5) * (
                    params["mass1"] + params["mass2"]) ** (-1. / 5)
        elif Dmax:
            self.D = rnd.uniform(0, 1) ** (1. / 3) * Dmax
            self.ra = rnd.uniform(0, 2 * np.pi)
            self.dec = np.arcsin(rnd.uniform(-1, 1))
            self.gps = lal.LIGOTimeGPS(gps, 0)
            self.gmst = lal.GreenwichMeanSiderealTime(self.gps)
            self.psi = rnd.uniform(0, 2 * np.pi)
            self.phi = rnd.uniform(0, 2 * np.pi)
            self.cosi = rnd.uniform(-1, 1)
            self.mchirp = 1.4 * 2 ** (-1. / 5)
        else:
            raise ValueError("Must provide either list of params or maximum distance")
        # general content:
        self.xyz = detectors.xyz(self.ra - self.gmst, self.dec)
        self.ifos = []
        self.mirror = False
        self.detected = False
        self.localization = {}
        self.area = {}
        self.patches = {}

    def add_network(self, network):
        """
        calculate the sensitivities and SNRs for the various detectors in network

        :param network: structure containing details of the network
        """
        self.threshold = network.threshold
        self.found = 0
        self.localized = 0
        self.snrsq = 0
        for ifo in network.ifos:
            i = getattr(network, ifo)
            if rnd.random() < i.duty_cycle:  # don't use numpy's RNG here as messes up seeding for networks
                det = copy.deepcopy(i)
                det.calculate_snr(self)
                # calculate SNR and see if the signal was found/useful for loc
                s = abs(det.snr)
                setattr(self, ifo, det)
                if s > det.found_thresh:
                    self.found += 1
                if s > det.loc_thresh:
                    if ifo != "H2" and ifo != "ETdet2":
                        self.localized += 1
                    self.snrsq += s ** 2
                    # add the details to the event
                    self.ifos.append(ifo)
        if self.found >= 2 and self.snrsq > self.threshold ** 2: self.detected = True

    def get_data(self, data):
        """
        get the relevant data for each detector and return it as an array

        :param data: string describing required data
        :returns: array with the data (for all ifos)
        """
        return np.array([getattr(getattr(self, i), data) for i in self.ifos])

    def get_fsig(self, mirror=False):
        """
        get the F_plus/cross times sigma for each detector

        :param mirror: boolean indicating whether we are considering the mirror location
        :return array with the sensitivities of the detectors
        """
        return np.array([getattr(self, i).get_fsig(mirror) for i in self.ifos])

    def get_f(self, mirror=False):
        """
        get the network sensitivity to plus and cross in the dominant polarization

        :param mirror: boolean indicating whether we are considering the mirror location
        :return length 2 array containing F_+, F_x response
        """
        M = np.zeros((2, 2))
        for f in self.get_fsig(mirror):
            M += np.outer(f, f)
        F = np.sqrt(np.linalg.eig(M)[0])
        F.sort()
        return F[::-1]

    def get_snr(self, dt_i=None):
        """
        get the relevant data for each detector and return it as an array
        if dt_i is given then shift the time in each detector by that amount
        """
        z = np.array([getattr(self, i).snr for i in self.ifos])
        if dt_i is not None:
            f_mean = self.get_data("f_mean")
            f_band = self.get_data("f_band")
            z *= (1 - 2 * np.pi ** 2 * (f_mean ** 2 + f_band ** 2) * dt_i ** 2
                  + 2.j * np.pi * dt_i * f_mean)
        return z

    def projected_snr(self, method, mirror=False, dt_i=None):
        """
        Calculate the projected SNR for a given method at either original or mirror
        sky location
        """
        f_sig = self.get_fsig(mirror)
        P = snr_projection(f_sig, method)
        zp = np.inner(self.get_snr(dt_i), P)
        return (zp)

    def calculate_mirror(self):
        """
        calculate the mirror location and detector sensitivity there
        """
        if len(self.ifos) == 3:
            self.mirror_loc = {}
            l = self.get_data("location")
            x = l[1] - l[0]
            y = l[2] - l[0]
            normal = np.cross(x, y)
            normal /= np.linalg.norm(normal)
            self.mirror_xyz = self.xyz - 2 * np.inner(self.xyz, normal) * normal
            mra, mdec = detectors.phitheta(self.mirror_xyz)
            mra += self.gmst
            self.mirror_ra = mra
            self.mirror_dec = mdec
            self.mirror = True
            for i in self.ifos:
                getattr(self, i).calculate_mirror_sensitivity(self)

    def calculate_sensitivity(self):
        """
        calculate the network sensitivity
        """
        self.sensitivity = np.linalg.norm(self.get_fsig())
        if self.mirror:
            self.mirror_sensitivity = np.linalg.norm(self.get_fsig(mirror=True))

    def localization_factors(self, method, mirror=False):
        """
        Calculate all of the localization factors for a given source
        and network of detectors, given the
        complex snr, sensitivity, bandwidth, mean frequency, location
        of the detectors.
        Here, we keep all the projection operators -- required if Z is not
        compatible with being a signal from the given sky location
        """
        f_mean = self.get_data("f_mean")
        f_band = self.get_data("f_band")
        # Calculate bar(f_sq)
        f_sq = (f_mean ** 2 + f_band ** 2)

        z = self.get_snr()
        locations = self.get_data("location")

        # calculate projection:
        f_sig = self.get_fsig(mirror)
        P = snr_projection(f_sig, method)

        # work out the localization factors
        B_i = 4 * np.pi ** 2 * np.real(sum(np.outer(f_sq * z.conjugate(), z) * P, axis=1))
        c_ij = 4 * np.pi ** 2 * np.real(np.outer(f_mean * z.conjugate(), f_mean * z) * P)
        C_ij = B_i * np.eye(len(B_i)) - c_ij
        c_i = sum(C_ij, axis=1)
        c = sum(c_i)

        A_i = 4 * np.pi * np.imag(sum(np.outer(f_mean * z.conjugate(), z) * P, axis=1))

        return A_i, C_ij, c_i, c

    def localize(self, method, mirror=False, p=0.9):
        """
        Localization of a source by a network of detectors, given the
        complex snr, sensitivity, bandwidth, mean frequency, location
        of the detectors.
        Here, we keep all the projection operators -- required if Z is not
        compatible with being a signal from the given sky location
        """
        if mirror:
            self.mirror_loc[method] = loc.Localization(method, self, mirror, p)
        else:
            self.localization[method] = loc.Localization(method, self, mirror, p)

    def combined_loc(self, method):
        """
        Calculate the area from original and mirror locations for the given method
        p = the confidence region (assume it's what was used to get a1, a2)
        """
        patches = 1
        p = self.localization[method].p
        a0 = - self.localization[method].area / np.log(1. - p)
        if self.mirror:
            if method == "marg":
                drho2 = 2 * (self.localization[method].like -
                             self.mirror_loc[method].like)
            else:
                drho2 = (self.localization[method].snr ** 2 -
                         self.mirror_loc[method].snr ** 2)
            prob_ratio = self.mirror_loc[method].p / p
            a_ratio = self.mirror_loc[method].area / self.localization[method].area
            if drho2 > 2 * (np.log(prob_ratio) + np.log(1 + p * a_ratio) - np.log(1 - p)):
                a = - np.log(1 - p * (1 + a_ratio * prob_ratio * np.exp(-drho2 / 2))) * a0
            else:
                patches = 2
                a = a0 * ((1 + a_ratio) * (-np.log(1 - p) + np.log(1 + a_ratio)
                                           - np.log(1 + a_ratio * prob_ratio * np.exp(-drho2 / 2)))
                          - a_ratio * (drho2 / 2 - np.log(prob_ratio)))
            if np.isnan(a):
                print("for method %s: we got a nan for the area" % method)
        if not self.mirror or np.isnan(a):
            a = - np.log(1. - p) * a0
            patches = 1

        self.patches[method] = patches
        self.area[method] = a

    def marg_loc(self, mirror=False, p=0.9):
        """
        Calculate the marginalized localization.
        """
        if mirror:
            localize = "mirror_loc"
        else:
            localize = "localization"
        l = getattr(self, localize)
        # set coherent to zero if we don't trust it:
        if (l["coh"].snr ** 2 - l["right"].snr ** 2 < 2) or \
                (l["coh"].snr ** 2 - l["left"].snr ** 2 < 2):
            l["coh"].like = 0
        keys = ["left", "right", "coh"]
        r_max = 1.1 * np.sqrt(np.nanmax([l[k].area for k in keys]) / np.pi)
        r_min = 0.9 * np.sqrt(np.nanmin([l[k].area for k in keys]) / np.pi)
        r = brentq(f, r_min, r_max, (keys, l, p))
        l["marg"] = loc.Localization("marg", self, mirror, p, area=np.pi * r ** 2)
        l["marg"].like = logsumexp([l[k].like + np.log(l[k].area) - np.log(-2 * np.pi * np.log(1 - l[k].p))
                                    for k in keys])
        l["marg"].like -= np.log(l["marg"].area) - np.log(-2 * np.pi * np.log(1 - p))

    def localize_all(self, p=0.9):
        """
        Calculate all localizations
        """
        self.calculate_mirror()
        self.calculate_sensitivity()
        for method in ["time", "coh", "left", "right"]:
            self.localize(method, mirror=False, p=p)
            if self.mirror: self.localize(method, mirror=True, p=p)
        self.marg_loc(p=p)
        if self.mirror: self.marg_loc(mirror=True, p=p)
        for method in ["time", "coh", "marg"]:
            self.combined_loc(method)


def f(r, keys, l, p):
    f = 0
    lmax = max([l[k].like for k in keys])
    for k in keys:
        s2 = l[k].area / (-2 * np.pi * np.log(1 - l[k].p))
        f += np.exp(l[k].like - lmax) * s2 * (1 - p - np.exp(-r ** 2 / (2 * s2)))
    return f
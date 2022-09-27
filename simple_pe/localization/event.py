import numpy as np
import random as rnd
import copy
from simple_pe.detectors import detectors
from simple_pe.localization import loc
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

    :param f_sig: a Nx2 array of detector responses [F+, Fx] x sigma
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


##################################################################
# Class to store event information
##################################################################
class Event(object):
    """
    class to hold the details of the event
    """

    def __init__(self, dist, ra, dec, phi, psi, cosi, mchirp, t_gps):
        """
        Initialize event.
        """

        self.D = float(dist)
        self.ra = float(ra)
        self.dec = float(dec)
        self.psi = float(psi)
        self.phi = float(phi)
        self.cosi = float(cosi)
        self.mchirp = float(mchirp)
        t = Time(t_gps, format='gps')
        self.gps = float(t.gps)
        self.gmst = float(t.sidereal_time('mean', 'greenwich').rad)
        self.xyz = detectors.xyz(self.ra - self.gmst, self.dec)
        self.ifos = []
        self.mirror = False
        self.mirror_xyz = None
        self.detected = False
        self.sensitivity = None
        self.mirror_sensitivity = None
        self.mirror_dec = None
        self.mirror_ra = None
        self.snrsq = None
        self.localized = None
        self.found = None
        self.threshold = None
        self.localization = {}
        self.mirror_loc = {}
        self.area = {}
        self.patches = {}

    @classmethod
    def from_params(cls, params):
        """
        Give a set of parameters, as used in the first 2 years paper,
        and use these to initialize an event
        :param params: parameters in form used by first 2 years paper
        """
        try:
            t = Time(params['gps'], format='gps')
        except:
            t = Time(params["MJD"], format='mjd')

        return cls(dist=params["distance"],
                   ra=np.radians(params["RAdeg"]),
                   dec=np.radians(params["DEdeg"]),
                   phi=np.radians(params["coa-phase"]),
                   psi=np.radians(params["polarization"]),
                   cosi=np.cos(np.radians(params["inclination"])),
                   mchirp=params["mass1"] ** (3. / 5) * params["mass2"] ** (3. / 5) * (
                           params["mass1"] + params["mass2"]) ** (-1. / 5),
                   t_gps=t.gps,

                   )

    @classmethod
    def random_values(cls, d_max=1000, mass=1.4, t_gps=1000000000):
        """
        Generate an event with random distance, orientation at given time and mass
        :param d_max: maximum distance
        :param mass: component mass (assumed equal mass)
        :param t_gps: GPS time of event
        """
        return cls(dist=rnd.uniform(0, 1) ** (1. / 3) * d_max,
                   ra=rnd.uniform(0, 2 * np.pi),
                   dec=np.arcsin(rnd.uniform(-1, 1)),
                   psi=rnd.uniform(0, 2 * np.pi),
                   phi=rnd.uniform(0, 2 * np.pi),
                   cosi=rnd.uniform(-1, 1),
                   mchirp=mass * 2 ** (-1. / 5),
                   t_gps=t_gps
                   )

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
        f_pc = np.sqrt(np.linalg.eig(M)[0])
        f_pc.sort()
        return f_pc[::-1]

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
        return zp

    def calculate_mirror(self):
        """
        calculate the mirror location and detector sensitivity there
        """
        if len(self.ifos) == 3:
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
        Calculate all the localization factors for a given source
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

        # calculate projection:
        f_sig = self.get_fsig(mirror)
        P = snr_projection(f_sig, method)

        # work out the localization factors
        B_i = 4 * np.pi ** 2 * np.real(np.sum(np.outer(f_sq * z.conjugate(), z) * P, axis=1))
        c_ij = 4 * np.pi ** 2 * np.real(np.outer(f_mean * z.conjugate(), f_mean * z) * P)
        C_ij = B_i * np.eye(len(B_i)) - c_ij
        c_i = np.sum(C_ij, axis=1)
        c = np.sum(c_i)

        A_i = 4 * np.pi * np.imag(np.sum(np.outer(f_mean * z.conjugate(), z) * P, axis=1))

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

import numpy as np
import random as rnd
import copy
from simple_pe import detectors
from simple_pe.localization import loc, sky_loc
from pesummary.gw.conversions.mass import mchirp_from_m1_m2
from astropy.time import Time
from scipy.optimize import brentq
from scipy.special import logsumexp


##################################################################
# Helper functions
##################################################################
def snr_projection(f_sig, method):
    """
    Function to calculate the SNR projection matrix p for a given set
    of detector responses, f_sig

    Parameters
    ----------
     f_sig: array
        a Nx2 array of detector responses [F+, Fx] x sigma
     method: str
        the way we project (one of "time", "coh", "left", "right")

    Returns
    -------
    p: array
        SNR projection matrix
    """
    if len(f_sig) == 1:
        # single detector, so projection is identity
        p = np.identity(len(f_sig))
    elif method == "time":
        p = np.identity(len(f_sig))
    elif method == "coh":
        M = np.zeros((2, 2))
        for f in f_sig:
            M += np.outer(f, f)
        p = np.inner(np.inner(f_sig, np.linalg.inv(M)), f_sig)
    elif method == "right":
        cf = np.array([complex(f[0], f[1]) for f in f_sig])
        p = np.outer(cf.conjugate(), cf) / np.inner(cf.conjugate(), cf)
    elif method == "left":
        cf = np.array([complex(f[0], f[1]) for f in f_sig])
        p = np.outer(cf, cf.conjugate()) / np.inner(cf, cf.conjugate())
    else:
        raise NameError("Invalid projection method: %s" % method)
    return p


##################################################################
# Class to store event information
##################################################################
class Event(object):
    """
    Class to hold the details of the event.  This contains the sky location,
    orientation, mass of the event.  The Event class can also have details of
    the active GW network, including the detectors, sensivitities and SNRs
    added.  These can be used to calculate the localization of the event.
    """
    def __init__(self, dist, ra, dec, phi, psi, cosi, mchirp, t_gps):
        """
        Initialize event.

    Parameters
    ----------
     dist: float
        distance to event
     ra: float
        right ascension
     dec: float
        declination
     phi: float
        coalescence phase
     psi: float
        polarization
     cosi: float
        cos of inclination angle
     mchirp: float
        chirp mass
     t_gps: float
        GPS time of event (coalescence time)
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

        Parameters
        ----------
        params: dict
            parameters in form used by first 2 years paper
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
                   mchirp=mchirp_from_m1_m2(params["mass1"], params["mass2"]),
                   t_gps=t.gps,
                   )

    @classmethod
    def random_values(cls, d_max=1000, mass=1.4, t_gps=1000000000):
        """
        Generate an event with random distance, orientation at given time and
        mass

        Parameters
        ----------
         d_max: float
            maximum distance
         mass: float
            component mass (assumed equal mass)
         t_gps: float
            GPS time of event
        """
        return cls(dist=rnd.uniform(0, 1) ** (1. / 3) * d_max,
                   ra=rnd.uniform(0, 2 * np.pi),
                   dec=np.arcsin(rnd.uniform(-1, 1)),
                   psi=rnd.uniform(0, 2 * np.pi),
                   phi=rnd.uniform(0, 2 * np.pi),
                   cosi=rnd.uniform(-1, 1),
                   mchirp=mchirp_from_m1_m2(mass, mass),
                   t_gps=t_gps
                   )

    @classmethod
    def from_snrs(cls, net, snrs, times, mchirp, ra=None, dec=None):
        """
        Give a network with SNR and time in each detector and use this
        to populate the event information.  If ra and dec are provided,
        they are used.  If not, then sky location is inferred from the time
        of arrival in different detectors.  For fewer than 3 detectors,
        sky location is chosen arbitrarily among possible points.

        Parameters
        ----------
        net: network.Network
            a network with SNR and time for each detector
        snrs: dict
            the complex snr in each detector
        times: dict
            the time in each detector
        mchirp: float
            the chirp mass of the event
        ra: float
            the right ascenscion of the source (optional)
        dec: float
            the declination of the source (optional)
        """
        for i in net.ifos:
            getattr(net, i).snr = snrs[i]
            getattr(net, i).time = times[i]
        f_band = {i: getattr(net, i).f_band for i in net.ifos}

        if (ra is None) and (dec is None):
            ra, dec = sky_loc.localization_from_timing(net.ifos, times, f_band)
        elif ra and dec:
            pass
        else:
            raise ValueError(
                "Please either provide an estimate for both 'ra' and  'dec', "
                "or neither."
            )
        ev = cls(dist=0.,
                 ra=ra,
                 dec=dec,
                 phi=0.,
                 psi=0.,
                 cosi=0.,
                 mchirp=mchirp,
                 t_gps=np.mean(list(times.values())),
                 )

        ev.add_network(net)

        return ev

    def add_network(self, network):
        """
        Calculate the sensitivities and SNRs for the various detectors in
        network

        Parameters
        ----------
        network: network.Network
            an object containing details of the network
        """
        self.threshold = network.threshold
        self.found = 0
        self.localized = 0
        self.snrsq = 0
        for ifo in network.ifos:
            i = getattr(network, ifo)
            # don't use Numpy's RNG here as messes up seeding for networks
            if rnd.random() < i.duty_cycle:
                det = copy.deepcopy(i)
                setattr(self, ifo, det)
                # calculate SNR (if not already given)
                if det.snr is None:
                    det.calculate_snr(self)
                else:
                    det.calculate_sensitivity(self)

                s = abs(det.snr)
                if s > det.found_thresh:
                    self.found += 1
                if s > det.loc_thresh:
                    self.snrsq += s ** 2
                    # add the details to the event
                    self.ifos.append(ifo)
            # only count one ET and one Hanford detector in localization
            hs = sum([i in self.ifos for i in ['H1', 'H2']])
            ets = sum([i in self.ifos for i in ['E1', 'E2', 'E3',
                                                'ETdet1', 'ETdet2']])
            self.localized = len(self.ifos) - max(hs - 1, 0) - max(ets - 1, 0)

        if self.found >= 2 and self.snrsq > self.threshold ** 2:
            self.detected = True

    def get_data(self, data):
        """
        Get the relevant data for each detector and return it as an array

        Parameters
        ----------
        data: str
            string giving name of data field

        Returns
        -------
        np.array
            with the data (for all ifos)
        """
        return np.array([getattr(getattr(self, i), data) for i in self.ifos])

    def get_fsig(self, mirror=False):
        """
        Get F_plus/F_cross multiplied by sigma (sensitivity) for each detector

        Parameters
        ----------
        mirror: boolean
            indicating whether we are considering the mirror location

        Returns
        -------
        np.array
            with the sensitivities of the detectors
        """
        return np.array([getattr(self, i).get_fsig(mirror) for i in self.ifos])

    def get_f(self, mirror=False):
        """
        Get the network sensitivity to plus and cross in dominant polarization

        Parameters
        ----------
        mirror: boolean
            indicating whether we are considering the mirror location

        Returns
        -------
        np.array
            length 2 array containing F_+, F_x response
        """
        m = np.zeros((2, 2))
        fsig = self.get_fsig(mirror)
        for f in fsig:
            m += np.outer(f, f)
        eigs = np.linalg.eig(m)[0]
        if len(fsig) == 1:
            # with 1-detector have no sensitivity to x
            # numerically can get negative
            eigs[eigs < 0] = 0
        f_pc = np.sqrt(eigs)
        f_pc.sort()
        return f_pc[::-1]

    def alpha_net(self, mirror=False):
        """
        Get the relative network sensitivity to the second polarization

        Parameters
        ----------
         mirror: boolean
            indicating whether we are considering the mirror location

        Returns
        -------
        float
            value of alpha_network
        """
        fp, fc = self.get_f(mirror)

        return fc/fp

    def get_snr(self, dt_i=None):
        """
        Calculate the snr for each detector at the requested time offset

        Parameters
        ----------
        dt_i: np.array
            time shift to be applied in each detector
            
        Returns
        -------           
        np.array
            the complex snr for each detector
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
        Calculate the projected SNR for a given method at either original or
        mirror sky location

        Parameters
        ----------
        method: str
            localization method to use
        mirror: boolean 
            indicating whether we are considering the mirror location
        dt_i: time shift to be applied in each detector
        
        Returns
        -------           
        np.array
            the complex snr for each detector
        """
        f_sig = self.get_fsig(mirror)
        p = snr_projection(f_sig, method)
        zp = np.inner(self.get_snr(dt_i), p)
        return zp

    def calculate_mirror(self):
        """
        Calculate the mirror location and detector sensitivity there
        """
        if len(self.ifos) == 3:
            location = self.get_data("location")
            x = location[1] - location[0]
            y = location[2] - location[0]
            normal = np.cross(x, y)
            normal /= np.linalg.norm(normal)
            self.mirror_xyz = self.xyz - 2 * np.inner(self.xyz, normal) * normal
            mra, mdec = detectors.phitheta(self.mirror_xyz)
            mra += self.gmst
            self.mirror_ra = mra % (2 * np.pi)
            self.mirror_dec = mdec
            self.mirror = True
            for i in self.ifos:
                getattr(self, i).calculate_mirror_sensitivity(self)

    def calculate_sensitivity(self):
        """
        Calculate the network sensitivity to the event, given the sky
        location and masses.
        """
        self.sensitivity = np.linalg.norm(self.get_fsig())
        if self.mirror:
            self.mirror_sensitivity = np.linalg.norm(self.get_fsig(mirror=True))

    def localization_factors(self, method, mirror=False):
        """
        Calculate all the localization factors for a given source
        and network of detectors, given the complex snr, sensitivity, 
        bandwidth, mean frequency, location of the detectors.
        Definition of terms given in "Localization of transient gravitational 
        wave sources: beyond triangulation", Class. Quantum Grav. 35 (2018) 
        105002.  

        Parameters
        ----------
        method: string
            localization method to use
        mirror: boolean 
            indicating whether we are considering the mirror location
            
        Returns
        ------- 
        a_i: np.array
            the localization factor A_i
        c_ij: np.array
            the localization matrix C_ij
        c_i: np.array
            the localization factor C_i
        c: float
            the localization factor C
        """
        f_mean = self.get_data("f_mean")
        f_band = self.get_data("f_band")
        # Calculate bar(f_sq)
        f_sq = (f_mean ** 2 + f_band ** 2)

        z = self.get_snr()

        # calculate projection:
        f_sig = self.get_fsig(mirror)
        p = snr_projection(f_sig, method)

        # work out the localization factors
        b_i = 4 * np.pi ** 2 * np.real(np.sum(np.outer(f_sq * z.conjugate(), z)
                                              * p, axis=1))
        c_ij = 4 * np.pi ** 2 * np.real(np.outer(f_mean * z.conjugate(),
                                                 f_mean * z) * p)
        c_ij = b_i * np.eye(len(b_i)) - c_ij
        c_i = np.sum(c_ij, axis=1)
        c = np.sum(c_i)

        a_i = 4 * np.pi * np.imag(np.sum(np.outer(f_mean * z.conjugate(), z)
                                         * p, axis=1))

        return a_i, c_ij, c_i, c

    def localize(self, method, mirror=False, p=0.9):
        """
        Calculate localization of source at given probability with
        chosen method

        Parameters
        ----------
        method: str
            localization method to use
        mirror: boolean
            indicating whether we are considering the mirror location
        p: float
        probability region to calculate (default 0.9)
        """
        if mirror:
            self.mirror_loc[method] = loc.Localization(method, self,
                                                       mirror, p)
        else:
            self.localization[method] = loc.Localization(method, self,
                                                         mirror, p)

    def combined_loc(self, method):
        """
        Calculate the area from original and mirror locations for the given
        method

        Parameters
        ----------
        method: str
            localization method to use
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
            a_ratio = self.mirror_loc[method].area / \
                      self.localization[method].area

            if drho2 > 2 * (np.log(prob_ratio) +
                            np.log(1 + p * a_ratio) - np.log(1 - p)):
                a = - np.log(1 - p * (1 + a_ratio * prob_ratio *
                                      np.exp(-drho2 / 2))) * a0
            else:
                patches = 2
                a = a0 * ((1 + a_ratio) * (-np.log(1 - p) + np.log(1 + a_ratio)
                                           - np.log(1 + a_ratio * prob_ratio *
                                                    np.exp(-drho2 / 2)))
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
        Calculate the marginalized localization.  Method described
        in "Localization of transient gravitational
        wave sources: beyond triangulation", Class. Quantum Grav. 35 (2018).

        Parameters
        ----------
        mirror: boolean
            indicating whether we are considering the mirror location
        p: float
            probability region to calculate (default=0.9)
        """
        if mirror:
            localize = "mirror_loc"
        else:
            localize = "localization"

        localization = getattr(self, localize)
        if (localization["coh"].snr ** 2 - localization["right"].snr ** 2 < 2) \
                or (localization["coh"].snr ** 2 - localization["left"].snr **
                    2 < 2):
            # set coherent to zero if we don't trust it:
            localization["coh"].like = 0

        keys = ["left", "right", "coh"]
        r_max = 1.1 * np.sqrt(np.nanmax([localization[k].area for k in keys])
                              / np.pi)
        r_min = 0.9 * np.sqrt(np.nanmin([localization[k].area for k in keys])
                              / np.pi)
        r = brentq(f, r_min, r_max, (keys, localization, p))
        localization["marg"] = loc.Localization("marg", self, mirror, p,
                                                area=np.pi * r ** 2)
        localization["marg"].like = logsumexp([localization[k].like +
                                               np.log(localization[k].area) -
                                               np.log(-2 * np.pi *
                                                      np.log(1 -
                                                             localization[
                                                                 k].p)) for k in keys])
        localization["marg"].like -= np.log(localization["marg"].area) - \
                                     np.log(-2 * np.pi * np.log(1 - p))

    def localize_all(self, p=0.9, methods=None):
        """
        Calculate localization with given set of methods.

        Parameters
        ----------
        p: float
            probability region to calculate
        methods: list
            A list of localization methods to use from 'time', 'coh', 'left',
            'right', 'marg'.  Default is to calculate all.
        """
        if methods is None:
            methods = ["time", "coh", "left", "right", "marg"]

        if 'marg' in methods:
            marg = True
            methods.remove('marg')
        else:
            marg = False

        self.calculate_mirror()
        self.calculate_sensitivity()
        for method in methods:
            self.localize(method, mirror=False, p=p)
            if self.mirror:
                self.localize(method, mirror=True, p=p)

        if marg:
            methods.append('marg')
            self.marg_loc(p=p)
            if self.mirror:
                self.marg_loc(mirror=True, p=p)

        for method in methods:
            self.combined_loc(method)


def f(r, keys, localization, p):
    """
    Function used in marginalization of localizations
    TODO: Fix documentation, and check if this function is used

    Parameters
    ----------
     r: float
        radius
     keys: list
        the different localization methods to consider
     localization: loc.Localization
        object storing localization information
     p: float
        probability

    Returns
    -------
    float
        f
    """
    f = 0
    lmax = max([localization[k].like for k in keys])
    for k in keys:
        s2 = localization[k].area / (-2 * np.pi * np.log(1 - localization[k].p))
        f += np.exp(localization[k].like - lmax) * s2 * \
             (1 - p - np.exp(-r ** 2 / (2 * s2)))
    return f

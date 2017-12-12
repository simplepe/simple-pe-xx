from numpy import *
import copy
import detectors
import fstat
import lal
from scipy import special
from astropy.time import Time
from astropy import units
from scipy.optimize import brentq
from scipy.misc import logsumexp

# a list of ifos that we can consider
ifos = ("H1", "H2", "I1", "K1", "L1", "V1", "ETdet1", "ETdet1", "ETdet1")


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
        P = identity(len(f_sig))
    elif method == "coh":
        M = zeros((2, 2))
        for f in f_sig:
            M += outer(f, f)
        P = inner(inner(f_sig, linalg.inv(M)), f_sig)
    elif method == "right":
        cf = array([complex(f[0], f[1]) for f in f_sig])
        P = outer(cf.conjugate(), cf) / inner(cf.conjugate(), cf)
    elif method == "left":
        cf = array([complex(f[0], f[1]) for f in f_sig])
        P = outer(cf, cf.conjugate()) / inner(cf, cf.conjugate())
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
    ev, evec = linalg.eig(M)
    epsilon = 1e-10
    sigma = 1 / sqrt(ev + epsilon)
    evec = evec[:, sigma.argsort()]
    sigma.sort()
    return evec, sigma

##################################################################
# Class to store detector information
##################################################################
class Det(object):
    """
    class to hold the details of a detector
    """

    def __init__(self, location, response, det_range, f_mean, f_band,
                 found_thresh=5.0, loc_thresh=4.0, duty_cycle=1.0):
        """
        Initialize
        :param location: array with detector location
        :param response: matrix with detector response
        :param det_range: the BNS range of the detector
        :param f_mean: float, mean frequency
        :param f_band: float, frequency bandwidth
        :param found_thresh: threshold for declaring an event found
        :param loc_thresh: threshold for declaring an event localized
        :param duty_cycle: fraction of time the detector is operational
        """
        self.location = location
        self.response = response
        self.det_range = det_range
        self.sigma = 2.26 * det_range * 8  # this gives the SNR at 1 Mpc
        self.f_mean = f_mean
        self.f_band = f_band
        self.found_thresh = found_thresh
        self.loc_thresh = loc_thresh
        self.duty_cycle = duty_cycle

    def calculate_sensitivity(self, event):
        """
        Calculate the sensitivity of the detector to an event
        :param event: object, containing ra, dec, psi, gmst
        """
        self.f_plus, self.f_cross = lal.ComputeDetAMResponse(self.response,
                                                             event.ra, event.dec,
                                                             event.psi, event.gmst)

    def calculate_mirror_sensitivity(self, event):
        """
        Calculate the sensitivity of the detector to an event, in its mirror sky location
        :param event: object, containing mirror_ra, mirror_dec, psi, gmst
        """
        self.mirror_f_plus, self.mirror_f_cross = \
            lal.ComputeDetAMResponse(self.response,
                                     event.mirror_ra, event.mirror_dec,
                                     event.psi, event.gmst)

    def calculate_snr(self, event):
        """
        Calculate the expected SNR of the event in the detector
        :param event: object, containing ra, dec, psi, gmst, phi, cosi
        :return the complex SNR for the signal
        """
        self.calculate_sensitivity(event)
        self.snr = (event.mchirp/ (1.4 * 2**(-1./5)) )**(5./6) * self.sigma / event.D * \
                   complex(cos(2 * event.phi), -sin(2 * event.phi)) * \
                   complex(self.f_plus * (1 + event.cosi ** 2) / 2, self.f_cross * event.cosi)

    def get_fsig(self, mirror=False):
        """
        Method to return the sensitivity of the detector
        :param mirror: boolean, is this the mirror position
        :return length 2 array: sigma * (F_plus, F_cross)
        """
        if mirror:
            return self.sigma * array([self.mirror_f_plus, self.mirror_f_cross])
        else:
            return self.sigma * array([self.f_plus, self.f_cross])


##################################################################
# Class to store network information 
##################################################################
class Network(object):
    """
    class to hold the details of the network.
    """

    def __init__(self, threshold=12.0):
        """

        :param threshold: detection threshold for the network
        """
        self.threshold = threshold
        self.ifos = []

    def add_ifo(self, ifo, location, response, det_range, f_mean, f_band,
                found_thresh=5.0, loc_thresh=4.0, duty_cycle=1.0):
        """
        :param ifo: name of ifo
        :param location: ifo location
        :param response: matrix with detector response
        :param det_range: the BNS range of the detector
        :param f_mean: float, mean frequency
        :param f_band: float, frequency bandwidth
        :param found_thresh: threshold for declaring an event found
        :param loc_thresh: threshold for declaring an event localized
        :param duty_cycle: fraction of time the detector is operational
        """
        d = Det(location, response, det_range, f_mean, f_band,
                found_thresh, loc_thresh, duty_cycle)
        setattr(self, ifo, d)
        self.ifos.append(ifo)

    def set_configuration(self, configuration, found_thresh=5.0, loc_thresh=4.0,
                          duty_cycle=1.0):
        """
        set the details of the detectors based on the given configuration.
        data is stored in the detectors module
        :param configuration: name of configuration
        :param found_thresh: threshold for single ifo detection
        :param loc_thresh: threshold for single ifo localization
        :param duty_cycle: fraction of time detectors are operational
        """
        ranges = detectors.range_8(configuration)
        ifos = ranges.keys()
        location, response = detectors.detectors(ifos)
        fmeans = detectors.fmean(configuration)
        fbands = detectors.bandwidth(configuration)
        for ifo in ifos:
            self.add_ifo(ifo, location[ifo], response[ifo], ranges[ifo], fmeans[ifo],
                         fbands[ifo], found_thresh, loc_thresh, duty_cycle)

    def get_data(self, data):
        """
        get the relevant data for each detector and return it as an array
        :param data: name of data to return from a detector
        :return array containing requested data
        """
        return array([getattr(getattr(self, i), data) for i in self.ifos])

################################
# plot the localization detector ellipse.  
# Note that this assumes the ellipse is centred on the event 
################################
def projection_ellipse(loc, event):
  # set up co-ordinates
  r = {}
  dangle=  pi/50.0
  angle= arange(0,2*pi+dangle*0.5,dangle)
  # calculate the basis
  x_net = loc.evec[:,0]
  y_net = loc.evec[:,1]
  z_net = loc.evec[:,2]
  # normalization to get 90% area:
  source = event.xyz
  x = inner(source,x_net) + \
      sqrt(2 * log(10)) *  loc.sigma[0] * cos(angle) 
  y = inner(source,y_net) + \
      sqrt(2 * log(10)) *  loc.sigma[1] * sin(angle) 
  z = sqrt(1 - x**2 - y**2) * sign(inner(source,z_net))
  # check that we're not going outside of unit circle:
  bad = x**2 + y**2 > 1
  if sum(bad) > 0:
    x = concatenate((x[bad.argmax():], x[0:bad.argmax()]))
    y = concatenate((y[bad.argmax():], y[0:bad.argmax()]))
    z = concatenate((z[bad.argmax():], z[0:bad.argmax()]))
    bad = concatenate((bad[bad.argmax():], bad[0:bad.argmax()]))
    x = x[~bad]
    y = y[~bad]
    z = z[~bad]
    x = append(concatenate((x, x[::-1])), x[0])
    y = append(concatenate((y, y[::-1])), y[0])
    z = append(concatenate((z, -z[::-1])),z[0])
  for i in xrange(3):
    r[i] = x * x_net[i] + y * y_net[i] + z * z_net[i]
  theta = arcsin(r[2] / sqrt(r[0]**2 + r[1]**2 + r[2]**2) )
  phi = arctan2(r[1],r[0])
  return phi, theta

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
            self.ra = radians(params["RAdeg"])
            self.dec = radians(params["DEdeg"])
            try: self.gps = params['gps']
            except: 
                t = Time(params["MJD"], format='mjd')
                self.gps = lal.LIGOTimeGPS(int(t.gps), int(1e9 * (t.gps % 1)))
            self.gmst = lal.GreenwichMeanSiderealTime(self.gps)
            # self.gmst = float(t.sidereal_time("mean", "greenwich")/units.hourangle)
            self.phi = radians(params["coa-phase"])
            self.psi = radians(params["polarization"])
            self.cosi = cos(radians(params["inclination"]))
            self.mchirp = params["mass1"]**(3./5) * params["mass2"]**(3./5) * (params["mass1"] + params["mass2"])**(-1./5)
        elif Dmax:
            self.D = random.uniform(0, 1) ** (1. / 3) * Dmax
            self.ra = random.uniform(0, 2 * math.pi)
            self.dec = arcsin(random.uniform(-1, 1))
            self.gps = lal.LIGOTimeGPS(gps,0)
            self.gmst = lal.GreenwichMeanSiderealTime(self.gps)
            self.psi = random.uniform(0, 2 * math.pi)
            self.phi = random.uniform(0, 2 * math.pi)
            self.cosi = random.uniform(-1, 1)
            self.mchirp = 1.4 * 2**(-1./5)
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
            if random.uniform(0, 1) < i.duty_cycle:
                det = copy.deepcopy(i)
                det.calculate_snr(self)
                # calculate SNR and see if the signal was found/useful for loc
                s = abs(det.snr)
                setattr(self, ifo, det)
                if s > det.found_thresh:
                    self.found += 1
                if s > det.loc_thresh:
                    if ifo != "H2" and ifo !="ETdet2":
                        self.localized += 1
                    self.snrsq += s ** 2
                    # add the details to the event
                    self.ifos.append(ifo)
        if self.found >= 2 and self.snrsq > self.threshold ** 2: self.detected = True

    def get_data(self, data):
        """
        get the relevant data for each detector and return it as an array
        :param data: string describing required data
        :return array with the data (for all ifos)
        """
        return array([getattr(getattr(self, i), data) for i in self.ifos])

    def get_fsig(self, mirror=False):
        """
        get the F_plus/cross times sigma for each detetor
        :param mirror: boolean indicating whether we are considering the mirror location
        :return array with the sensitivities of the detectors
        """
        return array([getattr(self, i).get_fsig(mirror) for i in self.ifos])

    def get_f(self, mirror=False):
        """
        get the network sensitivity to plus and cross in the dominant polarization
        :param mirror: boolean indicating whether we are considering the mirror location
        :return length 2 array containing F_+, F_x response
        """
        M = zeros((2, 2))
        for f in self.get_fsig(mirror):
            M += outer(f, f)
        F = sqrt(linalg.eig(M)[0])
        F.sort()
        return F[::-1]

    def get_snr(self, dt_i=None):
        """
        get the relevant data for each detector and return it as an array
        if dt_i is given then shift the time in each detector by that amount
        """
        z = array([getattr(self, i).snr for i in self.ifos])
        if dt_i is not None:
            f_mean = self.get_data("f_mean")
            f_band = self.get_data("f_band")
            z *= (1 - 2 * pi ** 2 * (f_mean ** 2 + f_band ** 2) * dt_i ** 2
                            + 2.j * pi * dt_i * f_mean)
        return z

    def projected_snr(self, method, mirror=False, dt_i = None):
        """
        Calculate the projected SNR for a given method at either original or mirror
        sky location
        """
        f_sig = self.get_fsig(mirror)
        P = snr_projection(f_sig, method)
        zp = inner(self.get_snr(dt_i), P)
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
            normal = cross(x, y)
            normal /= linalg.norm(normal)
            self.mirror_xyz = self.xyz - 2 * inner(self.xyz, normal) * normal
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
        self.sensitivity = linalg.norm(self.get_fsig())
        if self.mirror:
            self.mirror_sensitivity = linalg.norm(self.get_fsig(mirror=True))

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
        B_i = 4 * pi ** 2 * real(sum(outer(f_sq * z.conjugate(), z) * P, axis=1))
        c_ij = 4 * pi ** 2 * real(outer(f_mean * z.conjugate(), f_mean * z) * P)
        C_ij = B_i * eye(len(B_i)) - c_ij
        c_i = sum(C_ij, axis=1)
        c = sum(c_i)

        A_i = 4 * pi * imag(sum(outer(f_mean * z.conjugate(), z) * P, axis=1))

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
            self.mirror_loc[method] = Localization(method, self, mirror, p)
        else:
            self.localization[method] = Localization(method, self, mirror, p)

    def combined_loc(self, method):
        """
        Calculate the area from original and mirror locations for the given method
        p = the confidence region (assume it's what was used to get a1, a2)
        """
        patches = 1
        p = self.localization[method].p
        a0 = - self.localization[method].area / log(1. - p)
        if self.mirror:
            if method == "marg":
                drho2 = 2 * (self.localization[method].like -
                             self.mirror_loc[method].like)
            else:
                drho2 = (self.localization[method].snr ** 2 -
                         self.mirror_loc[method].snr ** 2)
            prob_ratio = self.mirror_loc[method].p / p
            a_ratio = self.mirror_loc[method].area / self.localization[method].area
            if drho2 > 2 * (log(prob_ratio) + log(1 + p * a_ratio) - log(1 - p)):
                a = - log(1 - p * (1 + a_ratio * prob_ratio * exp(-drho2 / 2))) * a0
            else:
                patches = 2
                a = a0 * ((1 + a_ratio) * (-log(1 - p) + log(1 + a_ratio)
                                           - log(1 + a_ratio * prob_ratio * exp(-drho2 / 2)))
                          - a_ratio * (drho2 / 2 - log(prob_ratio)))
            if isnan(a): 
                print("for method %s: we got a nan for the area" % method) 
        if not self.mirror or isnan(a):
            a = - log(1. - p) * a0
            patches = 1

        self.patches[method] = patches
        self.area[method] = a

        
    def marg_loc(self, mirror=False, p=0.9):
        """
        Calculate the marginalized localization.
        """
        if mirror:
            loc = "mirror_loc"
        else:
            loc = "localization"
        l = getattr(self, loc)
        # set coherent to zero if we don't trust it:
        if (l["coh"].snr ** 2 - l["right"].snr ** 2 < 2) or \
                (l["coh"].snr ** 2 - l["left"].snr ** 2 < 2):
            l["coh"].like = 0
        keys = ["left", "right", "coh"]
        r_max = 1.1 * sqrt(nanmax([l[k].area for k in keys])/pi)
        r_min = 0.9 * sqrt(nanmin([l[k].area for k in keys])/pi)
        r = brentq(f, r_min, r_max, (keys, l, p))
        l["marg"] = Localization("marg", self, mirror, p, area = pi*r**2)
        l["marg"].like = logsumexp([l[k].like + log(l[k].area) - log(-2*pi*log(1-l[k].p)) 
                                    for k in keys])
        l["marg"].like -= log(l["marg"].area) - log(-2*pi*log(1-p))

                
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
        s2 = l[k].area/(-2*pi*log(1-l[k].p))
        f += exp(l[k].like - lmax) * s2 * (1 - p - exp(-r**2/(2*s2)))
    return f



##################################################################
# Class to store localization information
##################################################################
class Localization(object):
    """
    class to hold the details of a localization method
    """

    def __init__(self, method, event, mirror=False, p=0.9, Dmax=1000, area = 0):
        """
        Initialization

        :param method: how we do localization, one of "time", "coh", "left, "right", "marg"
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
        if method is not "marg":
            self.calculate_m()
            self.calculate_max_snr()
            if self.M is not None:
                self.sky_project()
                self.calc_area()
            else:
                self.area = 1e6

            A  = fstat.snr_f_to_a(self.z, self.event.get_fsig(mirror))
            self.D, self.cosi, _, _ = fstat.a_to_params(A)

        self.like = 0.
        if method is not "time" and method is not "marg": 
            self.approx_like(Dmax)

    def calculate_m(self):
        A_i, C_ij, c_i, c = self.event.localization_factors(self.method, self.mirror)
        CC = 1. / 2 * (outer(c_i, c_i) / c - C_ij)
        # Calculate the Matrix M (given in the coherent localization paper)
        M = zeros([3, 3])
        locations = self.event.get_data("location")

        for i1 in xrange(len(self.event.ifos)):
            for i2 in xrange(len(self.event.ifos)):
                M += outer(locations[i1] - locations[i2],
                    locations[i1] - locations[i2]) / (3e8) ** 2 \
                    * CC[i1, i2]
        self.M = M

    def calculate_max_snr(self):
        self.z = self.event.projected_snr(self.method, self.mirror)
        A_i, C_ij, c_i, c = self.event.localization_factors(self.method, self.mirror)
        try:
            self.dt_i = 1. / 2 * inner(linalg.inv(C_ij), A_i)
        except:
            print("for method %s: Unable to invert C, setting dt=0" % self.method)
            self.dt_i = zeros_like(A_i)
        z = self.event.projected_snr(self.method, self.mirror)
        f_band = self.event.get_data("f_band")
        if max(abs(self.dt_i * (2 * pi * f_band))) < 1. / sqrt(2):
            extra_snr = inner(self.dt_i, inner(C_ij, self.dt_i))
            if extra_snr > 0:
                # location of second peak is reasonable -- use it
                z = self.event.projected_snr(self.method, self.mirror, self.dt_i)
        self.z = z
        self.snr = linalg.norm(z)

    def calculate_dt0(self, dt_i):
        """
        Calculate the SNR at a point offset from the source point
        dt_i is an array of time offsets for the detectors
        Note: we maximize over the overall time offset dt_o
        """
        A_i, C_ij, c_i, c = self.event.localization_factors(self.method, self.mirror)
        a = sum(A_i)
        dt0 = a/(2*c) - inner(c_i, dt_i) / c
        return dt0

    def calculate_snr(self, dt):
        A_i, C_ij, c_i, c = self.event.localization_factors(self.method, self.mirror)

        # See if the calculation is valid:
        f_band = self.event.get_data("f_band")
        if max(abs(dt * (2 * pi * f_band))) > 1. / sqrt(2):
            # location of second peak is outside linear regime, return zero
            z = zeros_like(dt)
        else:    
            z = self.event.projected_snr(self.method, self.mirror, dt)
        snr = linalg.norm(z)
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
            cos_fac = sqrt((Fp ** 2 + Fc ** 2) / (Fp * Fc))
            cosf = min(cos_fac / sqrt(self.snr), 0.5)
            self.like += log((self.D / Dmax) ** 3 / self.snr ** 2 * cosf)
        else:
            self.like += log(32. * (self.D / Dmax) ** 3 * self.D ** 4 / (Fp ** 2 * Fc ** 2)
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
        P = identity(3) - outer(source, source)
        self.PMP = inner(inner(P, self.M), P)
        self.evec, self.sigma = evec_sigma(self.PMP)

    def calc_area(self):
        """
        Calculate the localization area
        :param p: probability for event to be contained in the given area
        """
        # calculate the area of the ellipse
        ellipse = - log(1. - self.p) * 2 * math.pi * (180 / math.pi) ** 2 * \
                self.sigma[0] * self.sigma[1]
        # calculate the area of the band 
        band = 4 * math.pi * (180 / math.pi) ** 2 * sqrt(2) * special.erfinv(self.p) * self.sigma[0]
        # use the minimum (that's not nan)
        self.area = nanmin((ellipse, band))



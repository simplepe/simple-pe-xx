from numpy import *
import numpy as np
import random as rnd
import copy
from simple_pe.detectors import detectors
from simple_pe.fstat import fstat
import lal
from scipy import special
from astropy.time import Time
from scipy.optimize import brentq
from scipy.special import logsumexp


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
        :returns: the complex SNR for the signal
        """
        self.calculate_sensitivity(event)
        self.snr = (event.mchirp / (1.4 * 2 ** (-1. / 5))) ** (5. / 6) * self.sigma / event.D * \
                   complex(cos(2 * event.phi), -sin(2 * event.phi)) * \
                   complex(self.f_plus * (1 + event.cosi ** 2) / 2, self.f_cross * event.cosi)

    def get_fsig(self, mirror=False):
        """
        Method to return the sensitivity of the detector

        :param mirror: boolean, is this the mirror position
        :returns: length 2 array: sigma * (F_plus, F_cross)
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
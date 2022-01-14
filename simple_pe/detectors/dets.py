from numpy import *
from simple_pe.detectors import detectors
import lal
from pycbc import detector


##################################################################
# Class to store detector information
##################################################################
class Det(detector.Detector):
    """
    class to hold the details of a detector
    """

    def __init__(self, detector_name, det_range, f_mean, f_band,
                 found_thresh=5.0, loc_thresh=4.0, duty_cycle=1.0):
        """
        Initialize
        :param detector_name: 2 character string for detector
        :param det_range: the BNS range of the detector
        :param f_mean: float, mean frequency
        :param f_band: float, frequency bandwidth
        :param found_thresh: threshold for declaring an event found
        :param loc_thresh: threshold for declaring an event localized
        :param duty_cycle: fraction of time the detector is operational
        """
        super().__init__(detector_name)
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

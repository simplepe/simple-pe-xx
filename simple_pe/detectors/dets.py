import numpy as np
from pycbc import detector
from pesummary.gw.conversions.mass import mchirp_from_m1_m2


##################################################################
# Class to store detector information
##################################################################
class Det(detector.Detector):
    """
    class to hold the details of a detector

    Parameters
    ----------
    detector_name: 2 character string for detector
    det_range: the BNS range of the detector
    f_mean: float, mean frequency
    f_band: float, frequency bandwidth
    found_thresh: threshold for declaring an event found
    loc_thresh: threshold for declaring an event localized
    duty_cycle: fraction of time the detector is operational
    bns_range: is the given range for BNS (if yes, then rescale SNR with mchirp^5/6)
    """

    def __init__(self, detector_name, det_range, f_mean, f_band,
                 found_thresh=5.0, loc_thresh=4.0, duty_cycle=1.0, bns_range=True):

        super().__init__(detector_name)
        self.det_range = det_range
        self.sigma = 2.26 * det_range * 8  # this gives the SNR at 1 Mpc
        self.f_mean = f_mean
        self.f_band = f_band
        self.found_thresh = found_thresh
        self.loc_thresh = loc_thresh
        self.duty_cycle = duty_cycle
        self.bns_range = bns_range
        self.f_cross = None
        self.f_plus = None
        self.mirror_f_cross = None
        self.mirror_f_plus = None
        self.snr = None
        self.time = None

    def calculate_sensitivity(self, event):
        """
        Calculate the sensitivity of the detector to an event

        Parameters
        ----------
        event: object, containing ra, dec, psi, gmst
        """
        self.f_plus, self.f_cross = self.antenna_pattern(event.ra, event.dec, event.psi,
                                                         event.gps)

    def calculate_mirror_sensitivity(self, event):
        """
        Calculate the sensitivity of the detector to an event, in its mirror sky location

        Parameters
        ----------
        event: object, containing mirror_ra, mirror_dec, psi, gmst
        """
        self.mirror_f_plus, self.mirror_f_cross = self.antenna_pattern(event.mirror_ra,
                                                                       event.mirror_dec,
                                                                       event.psi, event.gps)

    def calculate_snr(self, event):
        """
        Calculate the expected SNR of the event in the detector

        Parameters
        ----------
        event: object, containing ra, dec, psi, gmst, phi, cosi

        Returns
        -------
        the complex SNR for the signal
        """
        self.calculate_sensitivity(event)
        if self.bns_range:
            mass_scale = (event.mchirp / mchirp_from_m1_m2(1.4, 1.4)) ** (5. / 6)
        else:
            mass_scale = 1.
        self.snr = mass_scale * self.sigma / event.D * complex(np.cos(2 * event.phi), -np.sin(2 * event.phi)) * \
                   complex(self.f_plus * (1 + event.cosi ** 2) / 2, self.f_cross * event.cosi)

    def get_fsig(self, mirror=False):
        """
        Method to return the sensitivity of the detector

        Parameters
        ---------
        mirror: boolean, is this the mirror position

        Returns
        -------
        :returns: length 2 array: sigma * (F_plus, F_cross)
        """
        if mirror:
            return self.sigma * np.array([self.mirror_f_plus, self.mirror_f_cross])
        else:
            return self.sigma * np.array([self.f_plus, self.f_cross])

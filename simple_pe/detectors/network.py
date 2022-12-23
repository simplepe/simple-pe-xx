import numpy as np
from simple_pe.detectors import detectors
from simple_pe.detectors import dets

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

    def add_ifo(self, ifo, det_range, f_mean, f_band,
                found_thresh=5.0, loc_thresh=4.0, duty_cycle=1.0, bns_range=True):
        """
        :param ifo: name of ifo
        :param det_range: the BNS range of the detector
        :param f_mean: float, mean frequency
        :param f_band: float, frequency bandwidth
        :param found_thresh: threshold for declaring an event found
        :param loc_thresh: threshold for declaring an event localized
        :param duty_cycle: fraction of time the detector is operational
        :param bns_range: is the given range for BNS (if yes, then rescale SNR with mchirp^5/6)
        """
        d = dets.Det(ifo, det_range, f_mean, f_band,
                     found_thresh, loc_thresh, duty_cycle, bns_range)
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
        fmeans = detectors.fmean(configuration)
        fbands = detectors.bandwidth(configuration)
        for ifo in ifos:
            self.add_ifo(ifo, ranges[ifo], fmeans[ifo], fbands[ifo], found_thresh, loc_thresh, duty_cycle)

    def get_data(self, data):
        """
        get the relevant data for each detector and return it as an array

        :param data: name of data to return from a detector
        :return array containing requested data
        """
        return np.array([getattr(getattr(self, i), data) for i in self.ifos])

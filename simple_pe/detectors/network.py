import numpy as np
from simple_pe import detectors
from simple_pe.detectors import Det

##################################################################
# Class to store network information
##################################################################


class Network(object):
    """
    class to hold the details of the network.

    Parameters
    ----------
    threshold: detection threshold for the network
    """

    def __init__(self, threshold=12.0):
        self.threshold = threshold
        self.ifos = []

    def add_ifo(self, ifo, horizon, f_mean, f_band,
                found_thresh=5.0, loc_thresh=4.0, duty_cycle=1.0,
                bns_range=True):
        """
        Add an ifo to a network

        Parameters
        ----------
        ifo: name of ifo
        det_range: the BNS range of the detector
        f_mean: float, mean frequency
        f_band: float, frequency bandwidth
        found_thresh: threshold for declaring an event found
        loc_thresh: threshold for declaring an event localized
        duty_cycle: fraction of time the detector is operational
        bns_range: is the given range for BNS (if yes, then rescale SNR with chirp_mass^5/6)
        """
        d = Det(ifo, horizon, f_mean, f_band,
                found_thresh, loc_thresh, duty_cycle, bns_range)
        setattr(self, ifo, d)
        self.ifos.append(ifo)

    def set_configuration(self, configuration, found_thresh=5.0, loc_thresh=4.0,
                          duty_cycle=1.0):
        """
        set the details of the detectors based on the given configuration.
        data is stored in the detectors module

        Parameters
        ----------
        configuration: name of configuration
        found_thresh: threshold for single ifo detection
        loc_thresh: threshold for single ifo localization
        duty_cycle: fraction of time detectors are operational
        """
        ranges = detectors.range_8(configuration)
        ifos = ranges.keys()
        fmeans = detectors.fmean(configuration)
        fbands = detectors.bandwidth(configuration)
        for ifo in ifos:
            self.add_ifo(ifo, 2.26 * ranges[ifo], fmeans[ifo], fbands[ifo],
                         found_thresh, loc_thresh, duty_cycle)

    def generate_network_from_psds(self, ifos, psds, f_lows,
                                   approximant='IMRPhenomD',
                                   found_thresh=5.0, loc_thresh=4.0,
                                   duty_cycle=1.0, bns_range=True):
        """
        Generate a network from a list of ifos, with associated PSDs
         and f_lows

        Parameters
        ----------
        ifos: list of ifos
        psds: dictionary of psds associated with the ifos
        f_lows: dictionary of low frequency cutoffs for each ifo
        approximant: approximant to use in waveform generation
        found_thresh: threshold for declaring an event found
        loc_thresh: threshold for declaring an event localized
        duty_cycle: fraction of time the detector is operational
        bns_range: is the given range for BNS
            (if yes, then rescale SNR with chirp_mass^5/6)
        """
        for ifo in ifos:
            horizon, f_mean, f_band = \
                detectors.calc_reach_bandwidth(mass1=1.4, mass2=1.4, spin=0,
                                               approx=approximant,
                                               power_spec=psds[ifo],
                                               fmin=f_lows[ifo],
                                               thresh=8)
            self.add_ifo(ifo, horizon, f_mean, f_band,
                         found_thresh, loc_thresh, duty_cycle,
                         bns_range)

    def get_data(self, data):
        """
        get the relevant data for each detector and return it as an array

        Parameters
        ----------
        data: name of data to return from a detector

        Returns
        -------
        data_array: array containing requested data
        """
        return np.array([getattr(getattr(self, i), data) for i in self.ifos])

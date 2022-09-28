import numpy as np
from scipy import interpolate

from pycbc.filter import sigma, sigmasq
from pycbc.waveform import get_fd_waveform
from simple_pe.cosmology import cosmology


def calc_reach_bandwidth(mass1, mass2, spin, approx, power_spec, fmin, thresh=8.):
    """
    Calculate the horizon, mean frequency and bandwidth for a given PSD in the detector frame
    :param mass1: the mass of the first component
    :param mass2: the mass of the second component
    :param spin: the aligned spin for both compoenents
    :param approx: the waveform used to calculate the horizon
    :param power_spec: the power spectrum to use
    :param fmin: the minimum frequency
    :param thresh: the SNR at which to calculate the horizon
    :return max_dist: the horizon for the signal
    :return meanf: the mean frequency
    :return sigf: the frequency bandwidth
    """
    fmax = power_spec.sample_frequencies[-1]
    df = power_spec.delta_f
    hpf, hcf = get_fd_waveform(approximant=approx,
                               mass1=mass1,
                               mass2=mass2,
                               spin1z=spin,
                               spin2z=spin,
                               f_lower=fmin,
                               f_final=fmax,
                               delta_f=df)
    ss = float(sigmasq(hpf, power_spec,
                       low_frequency_cutoff=fmin,
                       high_frequency_cutoff=hpf.sample_frequencies[-1]))
    hpf *= hpf.sample_frequencies ** 0.5
    ssf = float(sigmasq(hpf, power_spec,
                        low_frequency_cutoff=fmin,
                        high_frequency_cutoff=hpf.sample_frequencies[-1]))
    hpf *= hpf.sample_frequencies ** 0.5
    ssf2 = float(sigmasq(hpf, power_spec,
                         low_frequency_cutoff=fmin,
                         high_frequency_cutoff=hpf.sample_frequencies[-1]))
    max_dist = np.sqrt(ss) / thresh
    meanf = ssf / ss
    sigf = (ssf2 / ss - meanf ** 2) ** 0.5
    return max_dist, meanf, sigf


def calc_detector_horizon(mass1, mass2, spin, power_spec, fmin, snr=8, waveform='IMRPhenomD', triangle=False):
    """
    Calculate the horizon for a given PSD [in the detector frame]
    :param mass1: the mass of the first component
    :param mass2: the mass ratio of second component
    :param spin: the z-component of spin for both components
    :param power_spec: the power spectrum to use
    :param fmin: the minimum frequency
    :param snr: the SNR at which to calculate the horizon
    :param waveform: the waveform used to calculate the horizon
    :param triangle: scale horizon for a triangular detector (True/False)
    :return horizon: the horizon distance for the given system
    """
    fmax = power_spec.sample_frequencies[-1]
    df = power_spec.delta_f
    try:
        hp, _ = get_fd_waveform(approximant=waveform,
                                mass1=mass1,
                                mass2=mass2,
                                spin1z=spin,
                                spin2z=spin,
                                distance=1,
                                f_lower=fmin,
                                f_final=fmax,
                                delta_f=df)
        sig = sigma(hp,
                    power_spec,
                    low_frequency_cutoff=fmin,
                    high_frequency_cutoff=hp.sample_frequencies[-1])
    except:
        sig = 0.

    if triangle:
        return 1.5 * sig / snr
    else:
        return sig / snr


def interpolate_horizon(min_mass, max_mass, q, spin, power_spec, fmin, snr, waveform='IMRPhenomD', triangle=False):
    """
    Generate an interpolation function for the horizon [in the detector frame] for a binary
    with total mass between min_mass and max_mass, with given mass ratio
    and spin from a frequency fmin in a detector with given power_spec

    :param min_mass: the minimum total mass
    :param max_mass: the maximum total mass
    :param q: the mass ratio
    :param spin: the z-component of spin for both components
    :param fmin: the minimum frequency
    :param snr: the SNR at which to calculate the horizon
    :param waveform: the waveform used to calculate the horizon
    :param triangle: scale horizon for a triangular detector (True/False)
    :return horizon_interp: horizon interpolation function
    """
    # add a safety margin so interpolation definetely covers range
    masses = np.logspace(np.log10(0.5 * min_mass), np.log10(1.5 * max_mass), 100)
    horizon = np.array([calc_detector_horizon(mass * q / (1. + q), mass * 1 / (1. + q), spin,
                                              power_spec, fmin, snr, waveform, triangle)
                        for mass in masses])
    horizon_interp = interpolate.interp1d(masses, horizon)
    return horizon_interp


def calc_hm_horizon(mass1, mass2, spin, power_spec, fmin, snr=8, mode='22', waveform='IMRPhenomXHM', triangle=False):
    """
    Calculate the horizon for a given PSD [in the detector frame]

    :param mass1: the mass of the first component
    :param mass2: the mass ratio of second component
    :param spin: the z-component of spin for both components
    :param power_spec: the power spectrum to use
    :param fmin: the minimum frequency
    :param snr: the SNR at which to calculate the horizon
    :param mode: the mode for which to calculate horizon
    :param waveform: the waveform used to calculate the horizon
    :param triangle: scale horizon for a triangular detector (True/False)
    :return horizon: the higher mode horizon in detector frame for given masses and spin
    """
    fmax = power_spec.sample_frequencies[-1]
    df = power_spec.delta_f

    mode_array_dict = {
        '22': [[2, 2], [2, -2]], '44': [[4, 4], [4, -4]],
        '33': [[3, 3], [3, -3]]
    }

    if mode not in mode_array_dict.keys():
        print("Not implemented for this mode")
        return 0

    try:
        hp, _ = get_fd_waveform(approximant=waveform,
                                mass1=mass1,
                                mass2=mass2,
                                spin1z=spin,
                                spin2z=spin,
                                mode_array=mode_array_dict[mode],
                                distance=1,
                                f_lower=fmin,
                                f_final=fmax,
                                delta_f=df,
                                inclination=np.pi / 2)
        sig = sigma(hp,
                    power_spec,
                    low_frequency_cutoff=fmin,
                    high_frequency_cutoff=hp.sample_frequencies[-1])
        if mode == '22':
            sig *= 2
    except:
        sig = 0.
    if triangle:
        return 1.5 * sig / snr
    else:
        return sig / snr


def interpolate_hm_horizon(min_mass, max_mass, q, spin, power_spec, fmin, snr=8, mode='22',
                           waveform='IMRPhenomXHM', triangle=False):
    """
    Generate an interpolation function for the horizon [in the detector frame] for a binary
    with total mass between min_mass and max_mass, with given mass ratio
    and spin from a frequency fmin in a detector with given power_spec

    :param min_mass: the minimum total mass
    :param max_mass: the maximum total mass
    :param q: the mass ratio
    :param spin: the z-component of spin for both components
    :param fmin: the minimum frequency
    :param snr: the SNR at which to calculate the horizon
    :param mode: the mode to calculate the horizon for
    :param waveform: the waveform used to calculate the horizon
    :param triangle: scale horizon for a triangular detector (True/False)
    :return horizon_interp: horizon interpolation in detector frame for higher modes
    """
    # add a safety margin so interpolation definitely covers range
    masses = np.logspace(np.log10(0.5 * min_mass), np.log10(1.5 * max_mass), 100)
    horizon = np.array([calc_hm_horizon(mass * q / (1. + q), mass * 1 / (1. + q), spin,
                                        power_spec, fmin, snr, mode, waveform, triangle)
                        for mass in masses])
    horizon_interp = interpolate.interp1d(masses, horizon)
    return horizon_interp


def interpolate_source_horizon(min_mass, max_mass, hor_interp, snr_factor=1.):
    """
    Generate an interpolation function for the reach of the detector for
    a binary with total mass between min_mass and max_mass in the source frame,
    given the detector frame horizon.  The SNR factor takes into account the difference
    in threshold between the horizon and requested contour (either through
    wanting a different SNR limit or through sky averaging)

    :param min_mass: the minimum total mass
    :param max_mass: the maximum total mass
    :param hor_interp: detector frame interpolator
    :param snr_factor: ratio between SNR for horizon and requested contour
    :return h_interp: horizon interpolation in source frame
    """
    # add a safety margin so interpolation definetely covers range
    masses = np.logspace(np.log10(0.5 * min_mass), np.log10(1.5 * max_mass), 1000)
    d_horizon = hor_interp(masses) * snr_factor
    z_horizon = cosmology.redshift_at_lum_dist(d_horizon)
    m_horizon = masses / (1 + z_horizon)

    h_interp = interpolate.interp1d(m_horizon, z_horizon)

    return h_interp

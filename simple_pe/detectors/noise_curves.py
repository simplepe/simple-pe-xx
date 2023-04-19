import numpy as np
from scipy import interpolate

from pycbc.filter import sigma, sigmasq
from pycbc.waveform import get_fd_waveform
from simple_pe.cosmology import redshift_at_lum_dist


def calc_reach_bandwidth(mass1, mass2, spin, approx, power_spec, fmin, thresh=8.):
    """
    Calculate the horizon, mean frequency and bandwidth for a given PSD in the
    detector frame

    Parameters
    ----------
    mass1: float
        the mass of the first component
    mass2: float
        the mass of the second component
    spin: float
        the aligned spin for both compoenents
    approx: str
        the waveform used to calculate the horizon
    power_spec: pycbc.psd
        the power spectrum to use
    fmin: float
        the minimum frequency
    thresh: float
        the SNR at which to calculate the horizon

    Returns
    -------
    max_dist: float
        the horizon for the signal
    meanf: float
        the mean frequency
    sigf: float
        the frequency bandwidth
    """
    from simple_pe.waveforms.waveform import make_waveform
    fmax = power_spec.sample_frequencies[-1]
    df = power_spec.delta_f
    params = {
        "mass_1": mass1, "mass_2": mass2, "spin_1z": spin,
        "spin_2z": spin, "distance": 1.
    }
    hpf, hcf = make_waveform(
        params, df, fmin, int(fmax / df) + 1, approximant=approx,
        return_hc=True
    )
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


def calc_detector_horizon(mass1, mass2, spin, power_spec, fmin, snr=8, waveform='IMRPhenomD',
                          triangle=False):
    """
    Calculate the horizon for a given PSD [in the detector frame]

    Parameters
    ----------
    mass1: float
        the mass of the first component
    mass2: float
        the mass of second component
    spin: float
        the z-component of spin for both components
    power_spec: pycbc.psd
        the power spectrum to use
    fmin: float
        the minimum frequency
    snr: float
        the SNR at which to calculate the horizon
    waveform: str
        the waveform used to calculate the horizon
    triangle: bool
        scale horizon for a triangular detector (True/False)
    
    Returns
    -------
    horizon: float
        the horizon distance for the given system
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


def interpolate_horizon(min_mass, max_mass, q, spin, power_spec, fmin, snr,
                        waveform='IMRPhenomD', triangle=False):
    """
    Generate an interpolation function for the horizon [in the detector frame] for a binary
    with total mass between min_mass and max_mass, with given mass ratio
    and spin from a frequency fmin in a detector with given power_spec

    Parameters
    ----------
    min_mass: float
        the minimum total mass
    max_mass: float
        the maximum total mass
    q: float
        the mass ratio
    spin: float
        the z-component of spin for both components
    fmin: float
        the minimum frequency
    snr: float
        the SNR at which to calculate the horizon
    waveform: str
        the waveform used to calculate the horizon
    triangle: bool
        scale horizon for a triangular detector (True/False)
    
    Returns
    -------
    horizon_interp:
        horizon interpolation function
    """
    # add a safety margin so interpolation definetely covers range
    masses = np.logspace(np.log10(0.5 * min_mass), np.log10(1.5 * max_mass), 100)
    horizon = np.array([calc_detector_horizon(mass * q / (1. + q), mass * 1 / (1. + q), spin,
                                              power_spec, fmin, snr, waveform, triangle)
                        for mass in masses])
    horizon_interp = interpolate.interp1d(masses, horizon)
    return horizon_interp


def calc_hm_horizon(mass1, mass2, spin, power_spec, fmin, snr=8, mode='22',
                    waveform='IMRPhenomXHM', triangle=False):
    """
    Calculate the horizon for a given PSD [in the detector frame]

    Parameters
    ----------
    mass1: float
        the mass of the first component
    mass2: float
        the mass ratio of second component
    spin: float
        the z-component of spin for both components
    power_spec: pycbc.psd
        the power spectrum to use
    fmin: float
        the minimum frequency
    snr: float
        the SNR at which to calculate the horizon
    mode: str
        the mode for which to calculate horizon
    waveform: str
        the waveform used to calculate the horizon
    triangle: bool
        scale horizon for a triangular detector (True/False)

    Returns
    -------
    horizon: the higher mode horizon in detector frame for given masses and spin
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

    Parameters
    ----------
    min_mass: float
        the minimum total mass
    max_mass: float
        the maximum total mass
    q: float
        the mass ratio
    spin: float
        the z-component of spin for both components
    fmin: float
        the minimum frequency
    snr: float
        the SNR at which to calculate the horizon
    mode: str
        the mode to calculate the horizon for
    waveform: str
        the waveform used to calculate the horizon
    triangle: bool
        scale horizon for a triangular detector (True/False)

    Returns
    -------
    horizon_interp:
        horizon interpolation in detector frame for higher modes
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

    Parameters
    ----------
    min_mass: float
        the minimum total mass
    max_mass: float
        the maximum total mass
    hor_interp:
        detector frame interpolator
    snr_factor: float
        ratio between SNR for horizon and requested contour

    Returns
    -------
    h_interp:
        horizon interpolation in source frame
    """
    # add a safety margin so interpolation definetely covers range
    masses = np.logspace(np.log10(0.5 * min_mass), np.log10(1.5 * max_mass), 1000)
    d_horizon = hor_interp(masses) * snr_factor
    z_horizon = redshift_at_lum_dist(d_horizon)
    m_horizon = masses / (1 + z_horizon)

    h_interp = interpolate.interp1d(m_horizon, z_horizon)

    return h_interp

import numpy as np
from scipy import interpolate

from pycbc.filter import sigma, sigmasq
from pycbc.waveform import get_fd_waveform
from simple_pe.cosmology import redshift_at_lum_dist
from pycbc.detector import Detector
from pycbc.conversions import mass1_from_mtotal_q, mass2_from_mtotal_q
from simple_pe.fstat import fstat, fstat_hm
from simple_pe.waveforms.waveform_modes import mode_array_dict


def calc_reach_bandwidth(mass1, mass2, spin, approx, power_spec, fmin,
                         thresh=8.):
    """
    Calculate the horizon, mean frequency and bandwidth for a given PSD in
    the detector frame

    Parameters
    ----------
    mass1: float
        the mass of the first component
    mass2: float
        the mass of the second component
    spin: float
        the aligned spin for both components
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


def calc_detector_horizon(mass1, mass2, spin, power_spec, fmin, snr=8,
                          waveform='IMRPhenomD', triangle=False):
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
    return calc_mode_horizon(mass1, mass2, spin, power_spec, fmin, snr,
                             '22', waveform, triangle)


def interpolate_horizon(min_mass, max_mass, q, spin, power_spec, fmin, snr,
                        waveform='IMRPhenomD', triangle=False):
    """
    Generate an interpolation function for the horizon [in the detector frame]
    for a binary with total mass between min_mass and max_mass, with given
    mass rati and spin from a frequency fmin in a detector with given power_spec

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
    horizon_interp:
        horizon interpolation function
    """
    return interpolate_mode_horizon(min_mass, max_mass, q, spin, power_spec,
                                    fmin, snr, '22', waveform, triangle)


def calc_mode_horizon(mass1, mass2, spin, power_spec, fmin, snr=8, mode='22',
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


def interpolate_mode_horizon(min_mass, max_mass, q, spin, power_spec, fmin,
                             snr=8, mode='22', waveform='IMRPhenomXHM',
                             triangle=False):
    """
    Generate an interpolation function for the horizon [in the detector frame]
    for a binary with total mass between min_mass and max_mass, with given mass
    ratio and spin from a frequency fmin in a detector with given power_spec

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
    power_spec: pycbc.psd
        the power spectrum to use
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
    masses = np.logspace(np.log10(0.5 * min_mass), np.log10(1.5 * max_mass),
                         100)
    horizon = np.array([calc_mode_horizon(mass1_from_mtotal_q(mass, q),
                                          mass2_from_mtotal_q(mass, q),
                                          spin, power_spec, fmin, snr, mode,
                                          waveform, triangle)
                        for mass in masses])
    horizon_interp = interpolate.interp1d(masses, horizon)
    return horizon_interp


def interpolate_source_horizon(min_mass, max_mass, hor_interp, snr_factor=1.):
    """
    Generate an interpolation function for the reach of the detector for
    a binary with total mass between min_mass and max_mass in the source frame,
    given the detector frame horizon.  The SNR factor takes into account the
    difference in threshold between the horizon and requested contour
    (either through wanting a different SNR limit or through sky averaging)

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
    # add a safety margin so interpolation definitely covers range
    masses = np.logspace(np.log10(0.5 * min_mass), np.log10(1.5 * max_mass),
                         1000)
    d_horizon = hor_interp(masses) * snr_factor
    z_horizon = redshift_at_lum_dist(d_horizon)
    m_horizon = masses / (1 + z_horizon)

    h_interp = interpolate.interp1d(m_horizon, z_horizon)

    return h_interp


def generate_fplus_fcross_samples(ntrials=int(1e6), triangle=False):
    """
    generate values of F+ and Fx for a set of ntrials samples located at
    random sky positions (uniformly over the sky)

    Parameters
    ----------
    ntrials: int
        the number of trials
    triangle: bool
        L or triangle detector

    Returns
    -------
    f_plus: numpy.array
        array of f_plus values
    f_cross: numpy.array
        array of f_plus values
    """
    ra = np.random.uniform(0, 2 * np.pi, ntrials)
    dec = np.arcsin(np.random.uniform(-1, 1, ntrials))
    psi = np.random.uniform(0, np.pi, ntrials)
    t_gps = 1187008882.4434

    if triangle:
        ifos = ["ET1", "ET2", "ET3"]
    else:
        ifos = ['H1']

    dets = {ifo: Detector(ifo) for ifo in ifos}

    f_plus = {}
    f_cross = {}
    f = {}

    for ifo in ifos:
        f_plus[ifo] = np.zeros_like(ra)
        f_cross[ifo] = np.zeros_like(ra)
        f[ifo] = np.zeros_like(ra)
        for i, r in enumerate(ra):
            f_plus[ifo][i], f_cross[ifo][i] = \
                dets[ifo].antenna_pattern(r, dec[i], psi[i], t_gps)
            f[ifo] = np.sqrt(f_plus[ifo] ** 2 + f_cross[ifo] ** 2)

    f_plus['net'] = np.linalg.norm(np.array([f_plus[ifo] for ifo in ifos]),
                                   axis=0)
    f_cross['net'] = np.linalg.norm(np.array([f_cross[ifo] for ifo in ifos]),
                                    axis=0)

    return f_plus['net'], f_cross['net']


def calc_orientation_factors(ntrials=int(1e6), modes=None, triangle=False):
    """
     generate orientation factors for a set of samples located at random sky
     locations and random orientations

     Parameters
     ----------
     ntrials: int
         the number of trials
     modes: list
         a list of modes
     triangle: bool
         L or triangle detector

     Returns
     -------
     f_plus: numpy.array
         array of f_plus values
     f_cross: numpy.array
         array of f_plus values
     """
    if modes is None:
        modes = ['22']

    # the (source) polarization relative to the radiation frame is randomly
    # chosen
    chi = np.random.uniform(0, np.pi, ntrials)
    cosi = np.random.uniform(-1, 1, ntrials)
    phi0 = np.zeros_like(cosi)  # set the phase to zero (doesn't affect results)
    dist = np.ones_like(cosi)  # arbitrarily fix distance to unity
    f_plus, f_cross = generate_fplus_fcross_samples(ntrials, triangle)

    a_vals = {}
    for mode in modes:
        a_vals[mode] = fstat_hm.params_to_mode_a(mode, dist, cosi, chi, phi0,
                                                 alpha=1.)

    amp = fstat.expected_snr(a_vals, f_plus, f_cross)
    return amp


def calc_amp_info(amp, probs=None):
    """
    Calculate the maximum amplitude and the ratios at the given probs

    Parameters
    ----------
    ntrials: int
        the number of trials
    modes: list
        a list of modes
    triangle: bool
        L or triangle detector

    Returns
    -------
    """
    if probs is None:
        probs = [0.1, 0.5, 0.9]

    amp.sort()
    amax = np.max(amp)
    p_amp = np.zeros(len(probs))
    for i, p in enumerate(probs):
        p_amp[i] = amp[int(-p * len(amp))] / amax

    return amax, p_amp

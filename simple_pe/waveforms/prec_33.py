import numpy as np
from pesummary.gw.conversions.angles import _dphi, _dpsi
from simple_pe.waveforms import waveform_modes


def _make_waveform(
        approx, theta_jn, phi_jl, phase, psi_J, mass_1, mass_2, tilt_1, tilt_2,
        phi_12, a_1, a_2, beta, distance, apply_detector_response=True, **kwargs
):
    """Generate a frequency domain waveform

    Parameters
    ----------
    approx: str
        Name of the approximant you wish to use when generating the waveform
    theta_jn: float
        Angle between the total angular momentum and the line of sight
    phi_jl: float
        Azimuthal angle of the total orbital angular momentum around the
        total angular momentum
    phase: float
        The phase of the binary at coaelescence
    psi_J: float
        The polarization in the J-frame
    mass_1: float
        Primary mass of the binary
    mass_2: float
        Secondary mass of the binary
    tilt_1: float
        The angle between the total orbital angular momentum and the primary
        spin
    tilt_2: float
        The angle between the total orbital angular momentum and the primary
        spin
    phi_12: float
        The angle between the primary spin and the secondary spin
    a_1: float
        The spin magnitude on the larger object
    a_2: float
        The spin magnitude on the secondary object
    beta: float
        The opening angle of the system. Defined as the angle between the
        orbital angular momentum, L, and the total angular momentum J.
    apply_detector_response: Bool, optional
        if True apply an effective detector response and return
        fp * hp + fc * hc else return hp, hc. Default True
    **kwargs: dict
        All additional kwargs are passed to the
        pesummary.gw.waveform.fd_waveform function
    """
    from pesummary.gw.waveform import fd_waveform
    _samples = {
        "theta_jn": [theta_jn], "phi_jl": [phi_jl], "phase": [phase],
        "mass_1": [mass_1], "mass_2": [mass_2], "tilt_1": [tilt_1],
        "tilt_2": [tilt_2], "phi_12": [phi_12], "a_1": [a_1],
        "a_2": [a_2], "luminosity_distance": [distance]
    }
    waveforms = fd_waveform(
        _samples, approx, kwargs.get("df", 1. / 256),
        kwargs.get("f_low", 20.), kwargs.get("f_final", 1024.),
        f_ref=kwargs.get("f_ref", 20.), ind=0, pycbc=True,
        mode_array=waveform_modes.mode_array_dict['33']
    )
    hp, hc = waveforms["h_plus"], waveforms["h_cross"]
    if kwargs.get("flen", None) is not None:
        flen = kwargs.get("flen")
        hp.resize(flen)
        hc.resize(flen)
    if not apply_detector_response:
        return hp, hc
    dpsi = _dpsi(theta_jn, phi_jl, beta)
    fp = np.cos(2 * (psi_J - dpsi))
    fc = -1. * np.sin(2 * (psi_J - dpsi))
    h = (fp * hp + fc * hc)
    h *= np.exp(3j * _dphi(theta_jn, phi_jl, beta))
    return h


def calculate_precessing_harmonics(
        mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, phi_12, beta, distance,
        harmonics=[0, 1], approx="IMRPhenomXPHM", **kwargs
):
    """Decompose a precessing waveform into a series of harmonics as defined
    the SimplePE paper
    We return harmonics up to k=2, with accuracy to O(b^4)

    Parameters
    ----------
    mass_1: float
        Primary mass of the bianry
    mass_2: float
        Secondary mass of the binary
    a_1: float
        The spin magnitude on the larger object
    a_2: float
        The spin magnitude on the secondary object
    tilt_1: float
        The angle between the total orbital angular momentum and the primary
        spin
    tilt_2: float
        The angle between the total orbital angular momentum and the secondary
        spin
    phi_12: float
        The angle between the primary spin and the secondary spin
    beta: float
        The angle between the total angular momentum and the total orbital
        angular momentum
    distance: float
        The distance to the source
    harmonics: list, optional
        List of harmonics which you wish to calculate. Default [0, 1]
    approx: str, optional
        Approximant to use for the decomposition. Default IMRPhenomPv2
    """
    harm = {}

    if (0 in harmonics) or (2 in harmonics):
        h_eo_0 = _make_waveform(
            approx, np.pi / 2, 0, 0, 0,
            mass_1, mass_2, tilt_1, tilt_2,
            phi_12, a_1, a_2, beta, distance, **kwargs
        )
        h_eo_pi2 = _make_waveform(
            approx, np.pi/2, np.pi/2, np.pi/6, 0,
            mass_1, mass_2, tilt_1, tilt_2,
            phi_12, a_1, a_2, beta, distance, **kwargs
        )

        if 0 in harmonics:
            harm[0] = 0.5 * (h_eo_0 + h_eo_pi2)
        if 2 in harmonics:
            harm[2] = 0.1 * (h_eo_0 - h_eo_pi2)

    if 1 in harmonics:
        h_fo_0 = _make_waveform(
            approx, 0, 0, 0, 0,
            mass_1, mass_2, tilt_1, tilt_2,
            phi_12, a_1, a_2, beta, distance,
            **kwargs
        )
        harm[1] = 0.25 * h_fo_0

    return harm


def make_waveform_from_precessing_harmonics(
        harmonic_dict, theta_jn, phi_jl, phase, f_plus_j, f_cross_j
):
    """Generate waveform for a binary merger with given precessing harmonics and
    orientation

    Parameters
    ----------
    harmonic_dict: dict
        harmonics to include
    theta_jn: np.ndarray
        the angle between total angular momentum and line of sight
    phi_jl: np.ndarray
        the initial precession phase angle
    phase: np.ndarray
        the initial orbital phase
    f_plus_j: np.ndarray
        The Detector plus response function as defined using the J-aligned frame
    f_cross_j: np.ndarray
        The Detector cross response function as defined using the J-aligned
        frame
    """
    amps = harmonic_amplitudes(
        theta_jn, phi_jl, f_plus_j, f_cross_j, harmonic_dict
    )
    h_app = 0
    for k, harm in harmonic_dict.items():
        if h_app:
            h_app += amps[k] * harm
        else:
            h_app = amps[k] * harm
    h_app *= np.exp(3j * phase + 3j * phi_jl)
    return h_app


def harmonic_amplitudes(
        theta_jn, phi_jl, f_plus_j, f_cross_j, harmonics=[0, 1]
):
    """Calculate the amplitudes of the precessing harmonics as a function of
    orientation

    Parameters
    ----------
    theta_jn: np.ndarray
        the angle between J and line of sight
    phi_jl: np.ndarray
        the precession phase
    f_plus_j: np.ndarray
        The Detector plus response function as defined using the J-aligned frame
    f_cross_j: np.ndarray
        The Detector cross response function as defined using the J-aligned
        frame
    harmonics: list, optional
        The list of harmonics you wish to return. Default is [0, 1]
    """
    tau = np.tan(theta_jn / 2)

    amp = {}
    if 0 in harmonics:
        amp[0] = - 4 * (tau / (1 + tau ** 2) ** 3 * (f_plus_j - 1j * f_cross_j) +
                        tau ** 5 / (1 + tau ** 2) ** 3 * (f_plus_j + 1j * f_cross_j)
                        )
    if 1 in harmonics:
        amp[1] = -4 * np.exp(-1j * phi_jl) * (
                (1 - 5 * tau ** 2) / (1 + tau ** 2) ** 3 * (f_plus_j - 1j * f_cross_j) -
                (tau ** 6 - 5 * tau ** 4) / (1 + tau ** 2) ** 3 * (f_plus_j + 1j * f_cross_j)
        )
    if 2 in harmonics:
        amp[2] = - 20 * np.exp(-2j * phi_jl) * (
                - tau * (1 - 2 * tau**2) / (1 + tau ** 2) ** 3 * (f_plus_j - 1j * f_cross_j)
                - tau**3 * (tau**2 - 1) / (1 + tau ** 2) ** 3 * (f_plus_j + 1j * f_cross_j)
        )
    return amp

import copy
import numpy as np

spin_max = 0.98
ecc_max = 0.5
prec_min = 1e-3

param_mins = {'chirp_mass': 1.,
              'total_mass': 2.,
              'mass_1': 1.,
              'mass_2': 1.,
              'symmetric_mass_ratio': 0.04,
              'chi_eff': -1 * spin_max,
              'chi_align': -1 * spin_max,
              'chi_p': prec_min,
              'chi_p2': prec_min,
              'prec': prec_min,
              'chi': prec_min,
              'spin_1z': -1 * spin_max,
              'spin_2z': -1 * spin_max,
              'a_1': 0.,
              'a_2': 0.,
              'tilt_1': 0.,
              'tilt_2': 0.,
              'tilt': prec_min,
              'eccentricity': 0.,
              'ecc2': 0.,
              }

param_maxs = {'chirp_mass': 1e4,
              'total_mass': 1e4,
              'mass_1': 1e4,
              'mass_2': 1e4,
              'symmetric_mass_ratio': 0.2499,
              'chi_eff': spin_max,
              'chi_align': spin_max,
              'chi_p': spin_max,
              'chi_p2': spin_max**2,
              'chi': spin_max,
              'spin_1z': spin_max,
              'spin_2z': spin_max,
              'a_1': spin_max,
              'a_2': spin_max,
              'tilt_1': np.pi,
              'tilt_2': np.pi,
              'tilt': np.pi - prec_min,
              'eccentricity': ecc_max,
              'ecc2': ecc_max**2,
              }


def param_bounds(params, dx_directions, harm2=False):
    """
    calculate appropriate bounds on the dx_directions given the value of params

    :param params: dictionary with parameter values for initial point
    :param dx_directions: list of parameters for which to calculate waveform variations
    :param harm2: flag to indicate filtering 2-harmonic waveform
    """
    mins = copy.deepcopy(param_mins)
    maxs = copy.deepcopy(param_maxs)

    # generate bounds on spins:
    chia = "chi_eff" if "chi_eff" in params.keys() else "chi_align"
    chip = "chi_p2" if "chi_p2" in params.keys() else "chi_p"
    if chip == "chi_p2":
        n = 1
    else:
        n = 2

    if (chia in params) and (chip in params) and ((chia in dx_directions) or (chip in dx_directions)):
        if chia in dx_directions:
            mins[chia] = - np.sqrt(mins[chia] ** 2 - params[chip] ** n)
            maxs[chia] = np.sqrt(maxs[chia] ** 2 - params[chip] ** n)
        if chip in dx_directions:
            maxs[chip] = (maxs[chip] ** n - params[chia] ** 2) ** (1 / n)
            if harm2:
                # need to have nonzero chi_p to generate 2 harmonics
                mins[chip] = mins['prec'] ** n

    bounds = [(mins[k], maxs[k]) for k in dx_directions]

    return bounds

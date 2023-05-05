from pycbc.types import TimeSeries
from pycbc.waveform import taper_timeseries


def generate_eccentric_waveform(params, df, f_low, f_len):
    """
    params: simplePE params describing event
    df: frequency step
    f_low: low frequency
    f_len: length in frequency domain
    modes: modes to caluclate
    """
    try:
        import EOBRun_module
    except:
        print("Unable to import EOBRun_module, please check it is installed")
        return -1

    if "ecc2" in params.keys():
        ecc = params["ecc2"][0] ** 0.5
    else:
        ecc = params["eccentricity"][0]

    s_rate = 2 * int(f_len * df)
    t_len = int(1 / df)

    pars = {
        'M': params['total_mass'][0],
        'q': params['inverted_mass_ratio'][0],
        'chi1': params['spin_1z'][0],
        'chi2': params['spin_2z'][0],
        'LambdaAl2': 0.,
        'LambdaBl2': 0.,
        'ecc': ecc,
        'ecc_freq': 1,
        'domain': 0,
        'srate_interp': s_rate,
        'use_geometric_units': "no",
        'initial_frequency': f_low,
        'interp_uniform_grid': "yes",
        'use_mode_lm': [1],
        'arg_out': "yes",
    }
    
    t, hp, hc, _, _ = EOBRun_module.EOBRunPy(pars)

    hp_t = TimeSeries(hp, t[1] - t[0])
    hp_t = taper_timeseries(hp_t, tapermethod="TAPER_STARTEND")
    hp_t.resize(t_len * s_rate)
    hp = hp_t.to_frequencyseries()

    hc_t = TimeSeries(hc, t[1] - t[0])
    hc_t = taper_timeseries(hc_t, tapermethod="TAPER_STARTEND")
    hc_t.resize(t_len * s_rate)
    hc = hc_t.to_frequencyseries()

    return hp, hc

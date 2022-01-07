import numpy
from scipy import interpolate
from pycbc import psd
from pycbc import waveform

def interpolate_psd(psd_data, freq_data, fmin, fmax, df):
    npts = int(fmax/df) + 1
    freq = numpy.arange(0, fmax+df, df)
    p_int = interpolate.interp1d(freq_data, psd_data)
    p = numpy.zeros_like(freq)
    p[freq>=fmin] = p_int(freq[freq>=fmin])
    psd_int = psd.FrequencySeries(p, delta_f = df)
    return psd_int
   
def calc_reach_bandwidth(mass1, mass2, approx, power_spec, fmin, thresh=8.):
    fmax = power_spec.sample_frequencies[-1]
    df = power_spec.delta_f
    hpf, hcf = waveform.get_fd_waveform(approximant=approx, mass1=mass1, mass2=mass2,
                                        f_lower=fmin, f_final = fmax, delta_f = df)
    ss = float(waveform.sigmasq(hpf, power_spec, 
                                low_frequency_cutoff = fmin, 
                                high_frequency_cutoff = hpf.sample_frequencies[-1]))
    hpf *= hpf.sample_frequencies**(0.5)
    ssf = float(waveform.sigmasq(hpf, power_spec, 
                                low_frequency_cutoff = fmin, 
                                high_frequency_cutoff = hpf.sample_frequencies[-1]))
    hpf *= hpf.sample_frequencies**(0.5)
    ssf2 = float(waveform.sigmasq(hpf, power_spec, 
                                low_frequency_cutoff = fmin, 
                                high_frequency_cutoff = hpf.sample_frequencies[-1]))
    max_dist = numpy.sqrt(ss)/thresh
    meanf = ssf / ss
    sigf = (ssf2 / ss - meanf**2)**(0.5)
    return max_dist, meanf, sigf


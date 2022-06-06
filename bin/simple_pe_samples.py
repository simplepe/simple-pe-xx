# import useful python packages
import numpy as np
from scipy.stats import ncx2
from pycbc import psd
import h5py

from simple_pe.param_est import metric, pe

## Input data -- need to change to reading from a file:

npts = int(1e5)
n_sig = 3.1

ifos = ['H1', 'L1', 'V1']

snrs = {'22': 17.98,
        'H1': 9.06,
        'L1': 15.12,
        'V1': 3.55,
        '33': 3.5,
        'prec': 2.25,
        'right': 17.92,
        'left': 13.30}

params = {'mchirp': 15.288,
          'eta': 0.180,
          'chi_eff': 0.232}

param_array = np.array([params['mchirp'], params['eta'], params['chi_eff']])

approximants = {'aligned': 'IMRPhenomD',
                'hm': 'IMRPhenomPv3HM',
                'precessing': 'IMRPhenomPv3HM'}

psds = {'H1': 'aLIGOMidHighSensitivityP1200087',
        'L1': 'aLIGOMidHighSensitivityP1200087',
        'V1': 'AdVMidHighSensitivityP1200087',
        'f_low': 20.,
        'f_high': 8192,
        'length': 32
        }

alphas = {'network': 0.2}

psds['delta_f'] = 1. / psds['length']

# generate the PSDs
pycbc_psd = {}
for ifo in ifos:
    pycbc_psd[ifo] = psd.analytical.from_string(psds[ifo], psds['length'] * psds['f_high'] + 1, psds['delta_f'],
                                                psds['f_low'])

pycbc_psd['harm'] = 3. / sum([1. / pycbc_psd[ifo] for ifo in ifos])

# Approximate parameter estimation: chirp mass, mass ratio and effective spin

# Identify principal directions and parameter ranges
v_phys = metric.find_eigendirections(params['mchirp'], params['eta'], params['chi_eff'], snrs['22'], n_sig,
                                     psds['f_low'], pycbc_psd['harm'], approximants['aligned'])

# generate aligned spin samples
xx = np.random.normal(0, 1 / n_sig, npts)
yy = np.random.normal(0, 1 / n_sig, npts)
zz = np.random.normal(0, 1 / n_sig, npts)
pts = np.array([xx, yy, zz])

samples = np.inner(pts.T, v_phys.T) + param_array

samples = samples[(samples[:, 1] < 0.25) & (samples[:, 1] > 0)]
npts = len(samples)

mchirp = samples[:, 0]
eta = samples[:, 1]
chi_eff = samples[:, 2]

# generate starting distributions for theta and chi_p
cos_theta = (15 * np.random.uniform(0, 1, npts) + 1) ** 0.25 - 1
theta = np.arccos(cos_theta)
chi_p = np.random.uniform(0, 1, npts)
# make sure that spins are less than 1
chi_p *= np.sqrt(0.98 - chi_eff ** 2)

weighted_samples = {'mchirp': mchirp,
                    'eta': eta,
                    'chi_eff': chi_eff,
                    'theta': theta,
                    'chi_p': chi_p}

weights = np.ones_like(mchirp)
# calculate the SNR in higher modes, precession and 2nd polarization
if '33' in snrs.keys():
    rho_33 = pe.calculate_rho_33(mchirp, eta, chi_eff, theta, snrs['22'],
                                 pycbc_psd['harm'], psds['f_low'], approximant="IMRPhenomXPHM")
    rv_33 = ncx2(2, snrs['33'] ** 2)
    p_33 = rv_33.pdf(rho_33 ** 2)
    p_33 /= p_33.max()
    weighted_samples['p_33'] = p_33
    weights *= p_33

if 'prec' in snrs.keys():
    rho_p = pe.calculate_rho_p(mchirp, eta, chi_eff, theta, chi_p, snrs['22'],
                               pycbc_psd['harm'], psds['f_low'], approximant="IMRPhenomXPHM")
    rv_prec = ncx2(2, snrs['prec'] ** 2)
    p_prec = rv_prec.pdf(rho_p ** 2)
    p_prec /= p_prec.max()
    weighted_samples['p_prec'] = p_prec
    weights *= p_prec

if ('left' in snrs.keys()) and ('right' in snrs.keys()):
    rho_2pol = pe.calculate_rho_2nd_pol(theta, alphas['network'], snrs['22'])
    snrs['2pol'] = np.sqrt(snrs['22'] ** 2 - max(snrs['left'], snrs['right']) ** 2)
    rv_2pol = ncx2(2, snrs['2pol'] ** 2)
    p_2pol = rv_2pol.pdf(rho_2pol ** 2)
    p_2pol /= p_2pol.max()
    weighted_samples['p_2pol'] = p_2pol
    weights *= p_2pol / p_2pol.max()

weighted_samples['weights'] = weights

# cut samples based on weights
p = np.random.uniform(0, 1, len(weights))
keep = weights / weights.max() > p

final_samples = {'mchirp': mchirp[keep],
                 'eta': eta[keep],
                 'chi_eff': chi_eff[keep],
                 'theta': theta[keep],
                 'chi_p': chi_p[keep]}

f = h5py.File('weighted_samples.hdf5', 'w')
for p in weighted_samples.keys():
    f.create_dataset(p, data=weighted_samples[p])
f.close()

f = h5py.File('final_samples.hdf5', 'w')
for p in final_samples.keys():
    f.create_dataset(p, data=final_samples[p])
f.close()

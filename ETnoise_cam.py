import numpy as np
from numpy import *
import math
from pycbc import waveform
from pycbc.psd import aLIGOZeroDetHighPower as advdet
from sympy import Symbol, nsolve
import sympy, mpmath
import lal
#from matplotlib import pyplot
import h5py
from simple_pe import noise_curves_cam
from scipy import interpolate


#mpmath.mp.dps = 15
df, fl, fh = 1/8., 10., 4096.
# f = np.arange(df/2., fh + df, df) # np.linspace(1.,8178.2003999999997, 3000) #
f = np.linspace(5., fh + df,32769)
curve_data = open('simple_pe/curve_data.txt')

vals=curve_data.read()
frequencies=array(vals.split()[::6], dtype=float)
# aligo=vals.split()[1::6]
ETD=array(vals.split()[2::6], dtype=float)**2
CE=array(vals.split()[3::6], dtype=float)**2

## read frequencies from file instead of above
#curve_data = open('simple_pe/curve_data.txt')   # from mathew evans data: https://dcc.ligo.org/cgi-bin/private/DocDB/ShowDocument?docid=P1600143&version=
#vals=curve_data.read()
#f = np.array(vals.split()[::6],dtype=float)
#df, fl, fh = float(f[1])-float(f[0]), min(f), max(f)

def interpolate_psd(psd_data, freq_data, fmin, fmax, df):
    npts = int(fmax/df) + 1
    freq = np.linspace(fmin, fh + df,npts)
    p_int = interpolate.interp1d(freq_data, psd_data)
    p = np.zeros_like(freq)
    p[freq>=fmin] = p_int(freq[freq>=fmin])
    return p

def getnoise(tech = 'aligo'):
    
    if tech == 'ETB':
        xt = f/100.
        fit = 2.39/(10**27*xt**15.64) + 0.349/xt**2.145 + 1.76/xt**0.12 + 0.409*xt**1.1
        noise = fit**2/10.**50
    #noise *= 2. # handput to reduce the range by a factor of .7
    
    if tech == 'CosmicExplorer':
        noise = 10**(-50)*((f/10.5)**(-50) + (f/25.)**(-10) + 1.26*(f/50.)**(-4) + 2*(f/80.)**(-2) + 5 + 2*(f/100.)**2)
    
    if tech == 'LIGOBlueBird':
        noise = 8.*10**(-48)*((f/15.)**(-20.) + (f/20.)**(-8) + (f/22.)**(-4.)  + (f/3.)**(-1.) + 0.063 + 0.7*(f/2000.)**2.)
    
    if tech == 'aligo':
        f_len = int(fh/df) + 1
        noise = advdet(f_len, df, df)
        noise = noise.numpy()
        noise[0] = noise[1] #to remove zero from the first element

    if tech == 'CE':
        noise=interpolate_psd(CE,frequencies,5.,4096.,1/8.)
    
    if tech == 'ETD':
        noise=interpolate_psd(ETD,frequencies,5.,4096.,1/8.)
    return f, noise

def save_horizons(tech='aligo', mchmin = 1., mchmax = 650., etamin = 0.05, etamax = .25, dmch = 0.2, deta = .01):
    
    fname = tech+"_mchmin-"+str(mchmin)+"_mchmax-"+str(mchmax)+"_etamin-"+str(etamin)+"_etamax-"+str(etamax)+".h5py"
    print "Saving to file:", fname
    nmch, neta = int((mchmax - mchmin)/dmch)+1, int((etamax - etamin)/deta)+1
    #print nmch, neta
    infile = h5py.File(fname, "w")
    dset1 = infile.create_dataset("mchirp", (nmch*neta,), dtype='f', maxshape=(nmch*neta,))
    dset2 = infile.create_dataset("eta", (nmch*neta,), dtype='f', maxshape=(nmch*neta,))
    dset3 = infile.create_dataset("d_horizon", (nmch*neta,), dtype='f', maxshape=(nmch*neta,))
    dset4 = infile.create_dataset("fmean", (nmch*neta,), dtype='f', maxshape=(nmch*neta,))
    dset5 = infile.create_dataset("fband", (nmch*neta,), dtype='f', maxshape=(nmch*neta,))
    
    dset1.attrs['mchmin'] = mchmin
    dset1.attrs['mchmax'] = mchmax
    dset2.attrs['etamin'] = etamin
    dset2.attrs['etamax'] = etamax
    dset1.attrs['dmch'] = dmch
    dset2.attrs['deta'] = deta
    
    noise, count = getnoise(tech), 0
    for mch in np.arange(mchmin, mchmax, dmch):
        for eta in np.arange(etamin, etamax, deta):
            m1, m2 = get_m1m2(mch, eta)
            wfrm = get_waveform(m1, m2)
            dhr = d_horizon(wfrm, noise)
            fmean, fsq = get_fsq_fmean(wfrm, noise)
            
            dset1[count] = mch
            dset2[count] = eta
            dset3[count] = dhr
            snr = dhr * 8
            dset4[count] = 4*fmean/snr/snr
            dset5[count] = np.sqrt(4*fsq/snr/snr - (4*fmean/snr/snr)**2)
            
            count += 1
            print mch

    for dataset in infile:
        infile[dataset].resize((count,))
        infile.close()

def d_horizon(wfrm, noise):
    sq = (wfrm*np.conjugate(wfrm)/noise)*df
    return np.real(2*np.sqrt(sq.sum()))/8 # at SNR = 8

def get_fsq_fmean(wfrm, noise):
    
    fmean = (wfrm*np.conjugate(wfrm)/noise)*f*df
    fsq = (wfrm*np.conjugate(wfrm)/noise)*f**2*df
    
    return np.real(fmean.sum()), np.real(fsq.sum())


def get_waveform(m1, m2, d = 1):
    sptilde, sctilde = waveform.get_fd_waveform(approximant="IMRPhenomD",
                                                mass1=m1, mass2=m2, delta_f = df, f_lower=fl, f_final=fh, distance = d)
    return sptilde

def get_m1m2(mch, eta):
    x1, x2 = Symbol('x1'), Symbol('x2')
    f1 = (x1*x2)/(x1+x2)**2 - eta
    f2 = (x1*x2)**.6/(x1+x2)**.2 - mch
    return nsolve((f1, f2), (x1, x2), (mch*(eta/.25), mch*(.25/eta)), verify = False)


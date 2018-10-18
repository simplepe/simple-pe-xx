import numpy as np
import math
from pycbc import waveform
from pycbc.psd import aLIGOZeroDetHighPower as advdet
from sympy import Symbol, nsolve
import sympy, mpmath
import lal
from matplotlib import pyplot
import h5py

mpmath.mp.dps = 15
df, fl, fh = 1/8., 10., 4096.
f = np.arange(df/2, fh + df, df)

def getnoise(tech = 'aligo'):

    if tech == 'ETB':
        xt = f/100.
        fit = 2.39/(10**27*xt**15.64) + 0.349/xt**2.145 + 1.76/xt**0.12 + 0.409*xt**1.1
        noise = fit**2/10.**50
        noise *= 2. # handput to reduce the range by a factor of .7

    if tech == 'CosmicExplorer':
        noise = 10**(-50)*((f/10.5)**(-50) + (f/25.)**(-10) + 1.26*(f/50.)**(-4) + 2*(f/80.)**(-2) + 5 + 2*(f/100.)**2)

    if tech == 'LIGOBlueBird':
        noise = 8.*10**(-48)*((f/15.)**(-20.) + (f/20.)**(-8) + (f/22.)**(-4.)  + (f/3.)**(-1.) + 0.063 + 0.7*(f/2000.)**2.)

    if tech == 'aligo':
        f_len = int(fh/df) + 1
        noise = advdet(f_len, df, df)
        noise = noise.numpy()
        noise[0] = noise[1] #to remove zero from the first element
        
    return noise

def d_horizon(wfrm, noise):

    inp = 2*wfrm*np.conjugate(wfrm)/noise
    dhr = 4*np.sqrt(inp.sum())*df/8.
    return np.real(dhr)

def save_horizons(tech='aligo', mchmin = 1., mchmax = 650., etamin = 0.05, etamax = .25, dmch = 1., deta = .02):

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
    for ii in range(nmch*neta):
        mch = np.random.uniform(mchmin, mchmax)
        eta = np.random.uniform(etamin, etamax)
        m1, m2 = get_m1m2(mch, eta)
        wfrm = get_waveform(m1, m2)
        dhr = d_horizon(wfrm, noise)
        fmean, fband = get_fband_fmean(wfrm, noise)
            
        dset1[ii] = mch
        dset2[ii] = eta
        dset3[ii] = dhr
        dset4[ii] = fmean
        dset5[ii] = fband

        if ii % 100 == 0:
            print ii
           
    infile.close()
            
def get_fband_fmean(wfrm, noise):
    
    fmean = (2*wfrm*np.conjugate(wfrm)/noise)*f*df
    fband = (2*wfrm*np.conjugate(wfrm)/noise)*f**2*df
    
    return np.real(fmean.sum()), np.real(np.sqrt(fband.sum()))
        

def get_waveform(m1, m2, d = 1):
    sptilde, sctilde = waveform.get_fd_waveform(approximant="IMRPhenomD",
                             mass1=m1, mass2=m2, delta_f = df, f_lower=fl, f_final=fh, distance = d)
    return sptilde

def get_m1m2(mch, eta):
    x1, x2 = Symbol('x1'), Symbol('x2')
    f1 = (x1*x2)/(x1+x2)**2 - eta
    f2 = (x1*x2)**.6/(x1+x2)**.2 - mch
    return nsolve((f1, f2), (x1, x2), (mch*(eta/.25), mch*(.25/eta)), verify = False)

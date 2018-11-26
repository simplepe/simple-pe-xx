
import numpy as np
from pycbc.waveform import get_fd_waveform
from pycbc.filter import match
from pycbc import conversions
import copy

def make_waveform(x, dx, df, dist, f_low, flen):
    mc = x[0] + dx[0]
    eta = x[1] + dx[1]
    h_plus, h_cross = get_fd_waveform(approximant='IMRPhenomD', \
                                  mass1 = conversions.mass1_from_mchirp_eta(mc, eta), \
                                  mass2 = conversions.mass2_from_mchirp_eta(mc, eta), \
                                  spin1z = x[2] + dx[2],\
                                  spin2z = x[3] + dx[3],\
                                  delta_f=df,\
                                  distance = dist,
                                  f_lower=f_low)
    h_plus.resize(flen)
    return(h_plus)
    
def scale_vectors(vec, min_match, x, df, dist, f_low, flen, psd, tol=1e-8):
    h_plus = make_waveform(x, np.zeros_like(x), df, dist, f_low, flen)
    ndim = len(vec)
    v = copy.deepcopy(vec)
    for i in xrange(ndim):
        m = 0
        while (not m > (min_match * (1 - tol))) or (not m < (min_match * (1 + tol)) ):
            dx = v[i] 
            if (x - dx)[0] < 0: dx *= 0.9 * x[0]/dx[0]
            if (x - dx)[1] < 0: dx *= 0.9 * x[1]/dx[1]
            if (x - dx)[1] > 0.25: 
                print("Hit equal mass boundary")
                dx = 0.249 - x[1]
            diff_h_plus = make_waveform(x, dx, df, dist, f_low, flen)
            m, _ = match(h_plus, diff_h_plus, psd, low_frequency_cutoff=f_low)
            scale = np.sqrt((1. - min_match) / (1.0 - m))
            v[i] *= scale
    return v

def calculate_metric(vec, x, df, dist, f_low, flen, psd):
    ndim = len(vec)
    gij = np.zeros([ndim, ndim])

    # make the original waveform
    h_plus = make_waveform(x, np.zeros_like(x), df, dist, f_low, flen)
    # calculate the metric components in the basis defined by vec
    for i in xrange(ndim):
        dx = vec[i]
        # diagonal components
        diff_h_plus = make_waveform(x, dx, df, dist, f_low, flen)
        mplus = match(h_plus, diff_h_plus, psd=psd, low_frequency_cutoff=f_low)[0]
        diff_h_plus = make_waveform(x, -dx, df, dist, f_low, flen)
        mminus = match(h_plus, diff_h_plus, psd=psd, low_frequency_cutoff=f_low)[0]
        gij[i,i] = 1 - 0.5 * (mplus + mminus)

    for i in xrange(ndim):    
        for j in xrange(i+1,ndim):
            for s in ([[1,1],[1,-1],[-1,1],[-1,-1]]):
                dx = (s[0] * vec[i] + s[1] * vec[j])/np.sqrt(2)
                diff_h_plus = make_waveform(x, dx, df, dist, f_low, flen)
                gij[i,j] += -0.25 * s[0]/s[1] *  match(h_plus, diff_h_plus, 
                                         psd=psd, low_frequency_cutoff=f_low)[0]
            gij[j,i] = gij[i,j]
    return gij

def physical_metric(gij, vec):
    # calculate the metric for the coordinates (x,y,z) rather than vectors, vec
    vnorm = np.linalg.norm(vec, axis=1)
    ghatij = gij /vnorm/ vnorm.reshape((-1,1))
    return ghatij


def rotate_basis(gij, min_match):
    evals, evec = np.linalg.eig(gij)
    v = (evec * np.sqrt((1 - min_match)/evals)).T
    return v

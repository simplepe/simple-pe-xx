
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
            # check that the new point is physical
            alpha = check_physical(x, dx)
            diff_h_plus = make_waveform(x, alpha * dx, df, dist, f_low, flen)
            m_alpha, _ = match(h_plus, diff_h_plus, psd, low_frequency_cutoff=f_low)
            m = ((1 - alpha**2) + m_alpha) / alpha**2
            print(m)
            scale = np.sqrt((1. - min_match) / (1.0 - m))
            v[i] *= scale
    return v

def check_physical(x, dx, maxs = [1e4, 0.25, 0.98, 0.98], mins = [0, 0, -0.98, -0.98]):
    """
    check whether ther point described by the positions x + dx is
    physically permitted.  If not, rescale and return the scaling factor
    """
    alpha = 1.
    for i in xrange(4):
        if (x + dx)[i] < mins[i]: alpha = min(alpha, (x[i] - mins[i])/dx[i])
        if (x + dx)[i] > maxs[i]: alpha = min(alpha, (maxs[i] - x[i])/dx[i])
    return alpha

def calculate_metric(vec, x, df, dist, f_low, flen, psd):
    ndim = len(vec)
    gij = np.zeros([ndim, ndim])

    # make the original waveform
    h_plus = make_waveform(x, np.zeros_like(x), df, dist, f_low, flen)
    # calculate the metric components in the basis defined by vec
    for i in xrange(ndim):
        dx = vec[i]
        # diagonal components
        alpha = check_physical(x, dx)
        diff_h_plus = make_waveform(x, alpha * dx, df, dist, f_low, flen)
        m_alpha, _ = match(h_plus, diff_h_plus, psd, low_frequency_cutoff=f_low)
        mplus = ((1 - alpha**2) + m_alpha) / alpha**2
 
        alpha = check_physical(x, -dx)
        diff_h_plus = make_waveform(x, alpha * dx, df, dist, f_low, flen)
        m_alpha, _ = match(h_plus, diff_h_plus, psd, low_frequency_cutoff=f_low)
        mminus = ((1 - alpha**2) + m_alpha) / alpha**2
        gij[i,i] = 1 - 0.5 * (mplus + mminus)

    for i in xrange(ndim):    
        for j in xrange(i+1,ndim):
            for s in ([[1,1],[1,-1],[-1,1],[-1,-1]]):
                dx = (s[0] * vec[i] + s[1] * vec[j])/np.sqrt(2)
                alpha = check_physical(x, dx)
                diff_h_plus = make_waveform(x, alpha * dx, df, dist, f_low, flen)
                m_alpha, _ = match(h_plus, diff_h_plus, psd, low_frequency_cutoff=f_low)
                m = ((1 - alpha**2) + m_alpha) / alpha**2
                gij[i,j] += -0.25 * s[0]/s[1] *  m
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


import numpy as np
from pycbc.waveform import get_fd_waveform
from pycbc.filter import match
from pycbc import conversions
import copy

def make_waveform(x, dx, dist, df, f_low, flen, waveform="IMRPhenomD"):
    """
    This function makes a waveform for the given parameters and 
    returns h_plus generated at value (x + dx).  Currently defaults to

    Parameters
    ----------
    x : np.array with four values assumed to be mchirp, eta, s1z, s2z
    dx: same as x. 
    dist: distance to the signal
    df: frequency spacing of points
    f_low: low frequency cutoff
    flen: length of the frequency domain array to generate
    waveform: the waveform generator to use
    
    Returns
    -------
    h_plus: waveform as a frequency series with the requested df, flen
    """

    mc = x[0] + dx[0]
    eta = x[1] + dx[1]
    h_plus, h_cross = get_fd_waveform(approximant=waveform, \
                                  mass1 = conversions.mass1_from_mchirp_eta(mc, eta), \
                                  mass2 = conversions.mass2_from_mchirp_eta(mc, eta), \
                                  spin1z = x[2] + dx[2],\
                                  spin2z = x[3] + dx[3],\
                                  delta_f=df,\
                                  distance = dist,\
                                  f_lower=f_low)
    h_plus.resize(flen)
    return(h_plus)
    
def scale_vectors(x, vec, dist, min_match, df, f_low, flen, psd, 
        waveform="IMRPhenomD", tol=1e-8): 
    """
    This function scales the input vectors so that the minimal match between
    a waveform at point x and one at x + v[i] is equal to the minimal match,
    up to the specified tolerance.

    Parameters
    ----------
    x : np.array with four values assumed to be mchirp, eta, s1z, s2z
    vec: an array of directions dx in which to vary the waveform parameters
    dist: distance to the signal
    min_match: the desired minimal match
    df: frequency spacing of points
    f_low: low frequency cutoff
    flen: length of the frequency domain array to generate
    psd: the power spectrum to use in calculating the minimal match
    waveform: the waveform generator to use
    tol: the maximum difference between the requested and actual min_match

    Returns
    -------
    v: A set of vectors in the directions given by v but normalized to give the
    desired minimal match
    """
    h = make_waveform(x, np.zeros_like(x), dist, df, f_low, flen, waveform)

    ndim = len(vec)
    v = copy.deepcopy(vec)
    for i in xrange(ndim):
        m = 0
        while (not m > (min_match * (1 - tol))) or \
                (not m < (min_match * (1 + tol)) ):
            dx = v[i]
            m = 0
            # calculate average match in +/- dx direction
            for sign in [1., -1.]:
                alpha = check_physical(x, sign * dx)
                h_prime = make_waveform(x, alpha * sign * dx, dist, 
                        df, f_low, flen, waveform)
                m_a, _ = match(h, h_prime, psd, low_frequency_cutoff=f_low)
                m += 0.5 * scale_match(m_a, alpha)
               
            # rescale vector
            scale = np.sqrt((1. - min_match) / (1.0 - m))
            v[i] *= scale
    return v

def check_physical(x, dx, maxs = [1e4, 0.25, 0.98, 0.98], 
        mins = [0, 0, -0.98, -0.98]):
    """
    A function to check whether ther point described by the positions x + dx is
    physically permitted.  If not, rescale and return the scaling factor

    Parameters
    ----------
    x : np.array with four values assumed to be mchirp, eta, s1z, s2z
    dx: same as x. 
    maxs: the maximum permitted values of the physical parameters
    mins: the minimum physical values of the physical parameters

    Returns
    -------
    alpha: the scaling factor required to make x + dx physically permissable
    """
    alpha = 1.
    for i in xrange(4):
        if (x + dx)[i] < mins[i]: alpha = min(alpha, (x[i] - mins[i])/dx[i])
        if (x + dx)[i] > maxs[i]: alpha = min(alpha, (maxs[i] - x[i])/dx[i])
    return alpha

def scale_match(m_alpha, alpha):
    """
    A function to scale the match calculated at an offset alpha to the
    match at unit offset

    Parameters
    ----------
    m_alpha: the match at an offset alpha
    alpha: the value of alpha

    Returns
    -------
    m : the match at unit offset
    """
    m = (alpha**2 - 1 + m_alpha) / alpha**2
    return m


def calculate_metric(x, vec, dist, df, f_low, flen, psd, waveform="IMRPhenomD"):
    """
    A function to calculate the metric at a point x, associated to a given set
    of variations in the directions given by vec.

    Parameters
    ----------
    x : np.array with four values assumed to be mchirp, eta, s1z, s2z
    vec: an array of directions dx in which to vary the waveform parameters
    dist: distance to the signal
    df: frequency spacing of points
    f_low: low frequency cutoff
    flen: length of the frequency domain array to generate
    psd: the power spectrum to use in calculating the minimal match
    waveform: the waveform generator to use

    Returns
    -------
    gij: a square matrix, with size given by the length of vec, that gives the
         metric at x along the directions given by vec
    """
    ndim = len(vec)
    gij = np.eye(ndim)

    # make the original waveform
    h = make_waveform(x, np.zeros_like(x), dist, df, f_low, flen, waveform)

    # diagonal components
    # g_ii = 1 - 0.5 [m(dx_i) + m(-dx_i)]
    for i in xrange(ndim):
        dx = vec[i]

        for sign in [1., -1.]:
            alpha = check_physical(x, sign * dx)
            h_prime = make_waveform(x, alpha * sign * dx, dist, df, f_low, 
                    flen, waveform)
            m_a, _ = match(h, h_prime, psd, low_frequency_cutoff=f_low)
            m = scale_match(m_a, alpha)
            gij[i,i] -= 0.5 * m

    # off diagonal
    # g_ij = 0.25 * [- m(1/sqrt(2) (dx_i + dx_j)) - m(-1/sqrt(2) (dx_i + dx_j))
    #               + m(1/sqrt(2) (dx_i - dx_j)) - m(-1/sqrt(2) (dx_i - dx_j))]
    for i in xrange(ndim):    
        for j in xrange(i+1,ndim):
            for s in ([[1,1],[1,-1],[-1,1],[-1,-1]]):
                dx = (s[0] * vec[i] + s[1] * vec[j])/np.sqrt(2)
                alpha = check_physical(x, dx)
                h_prime = make_waveform(x, alpha * dx, dist, df, f_low, flen, 
                        waveform)
                m_a, _ = match(h, h_prime, psd, low_frequency_cutoff=f_low)
                m = scale_match(m_a, alpha)
                gij[i,j] += -0.25 * s[0]/s[1] *  m
            gij[j,i] = gij[i,j]

    return gij

def physical_metric(gij, basis):
    """
    A function to calculate the metric in physical coordinates

    Parameters
    ----------
    gij: the metric calculated with respect to a set of basis vectors
    basis: the basis vectors expressed in terms of the physical coordinates

    Returns
    -------
    gphys: the metric in physical coordinates
    """
    vnorm = np.linalg.norm(basis, axis=1)
    ghatij = gij /vnorm/ vnorm.reshape((-1,1))
    return ghatij


def calculate_evecs(gij, min_match):
    """
    A function to calculate the eigenvectors of the metric gij normalized
    so that the minimal match along the eigendirection is given by min_match

    Parameters
    ----------
    gij: the metric
    min_match: the required minimal match

    Returns
    -------
    v: the appropriately scaled eigenvectors
    """
    evals, evec = np.linalg.eig(gij)
    v = (evec * np.sqrt((1 - min_match)/evals)).T
    return v

def update_metric(x, gij, basis, min_match, dist, df, f_low, flen, psd, 
        waveform="IMRPhenomD"):
    """
    A function to re-calculate the metric gij based on the matches obtained
    for the eigenvectors of the original metric

    Parameters
    ----------
    x: the point in parameter space used to calculate the metric
    gij: the original metric
    basis: the basis relating the directions of the metric to the physical space
    min_match: the desired minimum match
    dist: distance to the signal
    df: frequency spacing of points
    f_low: low frequency cutoff
    flen: length of the frequency domain array to generate
    psd: the power spectrum to use in calculating the minimal match
    waveform: the waveform generator to use

    Returns
    -------
    g_prime: the updated metric
    """
    evecs = calculate_evecs(gij, min_match)
    v_phys = np.inner(evecs, basis.T)
    g_prime = calculate_metric(x, v_phys, dist, df, f_low, flen, psd)
    evec_inv = np.linalg.inv(evecs)
    gij_prime = np.inner(np.inner(evec_inv, g_prime), evec_inv)
    return gij_prime, evecs

def metric_error(gij, evecs, min_match):
    """
    A function to calculate the inner products between the evecs and check
    they are orthogonal and correctly normalized

    Parameters
    ----------
    gij: the metric
    evecs: the eigenvectors
    min_match: the desired minimum match (equivalently, norm of evecs)

    Returns
    -------
    max_err: the maximum error in the inner products
    """
    vgv = np.inner(np.inner(evecs, gij), evecs)
    off_diag = np.max(abs(vgv[~np.eye(gij.shape[0],dtype=bool)]))
    diag = np.max(abs(np.diag(vgv)) + min_match - 1)
    max_err = max(off_diag, diag)
    return max_err

def find_nearby(data, x, gij, basis, min_match, dist, df, f_low, flen, psd,
        waveform="IMRPhenomD", verbose=False):
    """
    A function to find the point in the grid defined by the metric gij
    that gives the highest match with the data

    Parameters
    ----------
    data: the data containing the waveform of interest
    x: the point in parameter space used to calculate the metric
    gij: the parameter space metric
    basis: the basis relating the directions of the metric to the physical space
    min_match: the desired minimum match
    dist: distance to the signal
    df: frequency spacing of points
    f_low: low frequency cutoff
    flen: length of the frequency domain array to generate
    psd: the power spectrum to use in calculating the minimal match
    waveform: the waveform generator to use

    Returns
    -------
    x_prime: the point in the grid with the highest match
    """
    
    evecs = calculate_evecs(gij, min_match)
    ndim = len(evecs)
    v_phys = np.inner(evecs, basis.T)

    while True:
        h = make_waveform(x, np.zeros_like(x), dist, df, f_low, flen, waveform)
        m_0, _ = match(data, h, psd, low_frequency_cutoff=f_low)
        matches = np.zeros([ndim, 2])
        alphas = np.zeros([ndim, 2])
        
        for i in xrange(ndim):
            for j in xrange(2):
                dx = (-1)**j * v_phys[i]
                alphas[i,j] = check_physical(x, dx)
                h = make_waveform(x, alphas[i,j] * dx, dist, df, f_low, flen, 
                        waveform)
                matches[i,j], _ = match(data, h, psd, 
                        low_frequency_cutoff=f_low)

        if verbose: 
            print("Central match %.2f; maximum offset match %.2f" % 
                (m_0, matches.max()))

        if (matches.max() > m_0):
            # maximum isn't at the centre so update location
            i, j = np.unravel_index(np.argmax(matches), matches.shape)
            x += (-1)**j * v_phys[i]
            if verbose:
                print("Moving in the %d eigendirection, %d units" % 
                    (i, alphas[i,j] * (-1)**j) )
        else: 
            if verbose: print("Maximum at the centre, stopping")        
            break

    return x


def find_peak(data, x, gij, basis, min_match, dist, df, f_low, flen, psd, 
         waveform="IMRPhenomD"):
    """
    A function to find the maximum match 

    Parameters
    ----------
    data: the data containing the waveform of interest
    x: the point in parameter space used to calculate the metric
    gij: the parameter space metric
    basis: the basis relating the directions of the metric to the physical space
    min_match: the desired minimum match
    dist: distance to the signal
    df: frequency spacing of points
    f_low: low frequency cutoff
    flen: length of the frequency domain array to generate
    psd: the power spectrum to use in calculating the minimal match
    waveform: the waveform generator to use

    Returns
    -------
    x_prime: the point in the grid with the highest match
    """
    
    evecs = calculate_evecs(gij, min_match)
    ndim = len(evecs)
    v_phys = np.inner(evecs, basis.T)

    h = make_waveform(x, np.zeros_like(x), dist, df, f_low, flen, waveform)
    m_0, _ = match(data, h, psd, low_frequency_cutoff=f_low)
    matches = np.zeros([ndim, 2])
    alphas = np.zeros([ndim, 2])
    
    delta_x = np.zeros_like(x)

    for i in xrange(ndim):
        for j in xrange(2):
            dx = (-1)**j * v_phys[i]
            alphas[i,j] = check_physical(x, dx)
            h = make_waveform(x, alphas[i,j] * dx, dist, df, f_low, flen,
                    waveform)
            m, _  = match(data, h, psd, 
                    low_frequency_cutoff=f_low)
            matches[i,j] = scale_match(m, alphas[i,j])

    for i in xrange(ndim):
        delta_x += (matches[i,0] - matches[i,1]) * 0.5 * v_phys[i] / \
                (m_0 - 0.5 * (matches[i,0] + matches[i,1]))

    h = make_waveform(x, delta_x, dist, df, f_low, flen, waveform)
    m_peak = match(data, h, psd, low_frequency_cutoff=f_low)[0]
    x_peak = x + delta_x
    
    return x_peak, m_peak
    


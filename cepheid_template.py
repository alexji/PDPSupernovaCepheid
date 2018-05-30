"""
Code based on results from Yoachim et al. 2009
https://arxiv.org/abs/0903.4186
PCA Coefficients and code from
http://staff.washington.edu/yoachim/code.php
"""

import numpy as np
from astropy.table import Table

def cepheid_lightcurve(band, tarr, m0, period, phaseshift, shape1, shape2, shape3, shape4, datatable=None):
    """
    Generate a Cepheid light curve.
    band: one of "I", "V", "B"
    tarr: times at which you want the light curve evaluated
    m0: mean magnitude for the light curve
    period: same units as tarr
    phaseshift: same units as tarr
    shape1-4: parameters determining the shape of the light curve.
        These are the first four principle components from Yoachim et al. 2009
        They should generally be > 0.
    datatable: which set of templates to use.
        By default, it loads the long period templates.
        Long period: >10 days; Short period: <10 days
        Can also pass an integer.
        Even int -> long period, odd int -> short period.
        Note: for speed in fitting, read the table you want and pass it in.
    """
    allowed_bands = ["I","V","B"]
    assert band.upper() in allowed_bands
    
    if datatable is None: 
        datatable = load_longperiod_datatable()
    elif isinstance(datatable,(int,float)):
        datatable = int(datatable)
        if (datatable % 2) == 1:
            datatable = load_shortperiod_datatable()
        else:
            datatable = load_longperiod_datatable()
    
    Nt = len(tarr)
    tstack = np.ravel([tarr for x in range(3)])
    #p0 = m0i, m0v, m0b, period, phase shift, tbreak, tbreak2
    p0 = [m0,m0,m0,period,phaseshift,Nt,2*Nt, shape1, shape2, shape3, shape4]
    lcs = gen_lc(tstack, p0, datatable)
    lc = lcs[allowed_bands.index(band)]
    return lc

def load_longperiod_datatable():
    return Table.read("BVI_templates/long_struc.fits")    
def load_shortperiod_datatable():
    return Table.read("BVI_templates/short_struc.fits")    

def f_comp(t,p0):
    """ from f_comp.pro """
    # p0 = m0, period, phase, (alphas), (betas)
    newt = np.array(t-p0[2])
    n_extra = 3
    na = int((len(p0)-n_extra)/2.)
    alphas = np.array(p0[n_extra:(n_extra+na)])
    alphas = np.ones_like(newt)[:,np.newaxis].dot(alphas[np.newaxis,:])
    betas = np.array(p0[(n_extra+na):(n_extra+2*na)])
    betas = np.ones_like(newt)[:,np.newaxis].dot(betas[np.newaxis,:])
    tt = newt[:,np.newaxis].dot((np.ones(na))[np.newaxis,:])
    k = np.ones(len(t))[:,np.newaxis].dot((np.arange(na)+1.)[np.newaxis,:])
    arg = (2*np.pi*k*tt/p0[1])
    result = alphas * np.sin(arg) + betas * np.cos(arg)
    result = p0[0] + np.sum(result,axis=1)
    return result
def trip_ceph(t,p0):
    """ From trip_ceph.pro """
    #p0 = m0i, m0v, m0b, period, phase shift, tbreak, tbreak2,
    #    (alphai), (betai), (alphav), (betav), (alphab), (betab)
    n_extra = 7
    na = (len(p0)-n_extra)/6.
    twona = int(2*na)
    fourna = int(4*na)
    sixna = int(6*na)
    pi = list([p0[0]]) + list(p0[3:5]) + list(p0[n_extra:(n_extra+twona)])
    pv = list([p0[1]]) + list(p0[3:5]) + list(p0[(n_extra+twona):(n_extra+fourna)])
    pb = list([p0[2]]) + list(p0[3:5]) + list(p0[(n_extra+fourna):(n_extra+sixna)])
    
    ti = t[0:p0[5]]
    tv = t[p0[5]:p0[6]]
    tb = t[p0[6]:len(t)]
    
    lc_i = f_comp(ti,pi)
    lc_v = f_comp(tv,pv)
    lc_b = f_comp(tb,pb)
    return lc_i, lc_v, lc_b
def gen_lc(x,p0,struct,pmax=85.):
    """ From gen_lc.pro """
    #p0 = m0i, m0v, m0b, period, phase shift, tbreak1, tbreak2
    p = np.array(p0).copy()
    if len(p) < 11: # use the poly fits to fill in the pca1-pca4 coeffs
        c_need = 11 - len(p0)
        poly_fits = struct["POLY_FITS"][0]
        new_coeffs = []
        for i in range(4-c_need,3+1):
            coef = np.polyval(poly_fits[::-1,i],np.log10(min(p0[3],pmax)))
            new_coeffs.append(coef)
        p = np.array(list(p)+new_coeffs)
    AVE_VALS = struct["AVE_VALS"][0]
    eigvecs = (struct["EIGENVECTORS"][0])[0:4,:]
    albes = AVE_VALS + eigvecs.T.dot(p[7:11])
    p = list(p[0:7]) + list(albes)
    return trip_ceph(x,p)

"""
Implementation of Vector Spherical Harmonics to facilitate computations involving
VSH as basis functions. The immediate application is in Mie theory, and construction of
Mie T-matrix and solutions for plane wave incidence are also provided

A term in the expansion is given by

E = alm Mlm(kr) + blm Nlm(kr)

where alm and blm depend on the ratio of sphere to background refractive index, and size parameter in free space

So parameters are:

(ka, nsphere/nbackground, position vector, incidence direction)

The numerical paramter is maxL = maximum value of the order of spherical harmonics
"""
import sys
import os
import numpy as np
from numpy import exp, sqrt
from math import pi
import datetime
from dataclasses import dataclass
from scipy import special as sp
import pickle
from multiprocessing import Pool, cpu_count
import emops_simple as em
CTYPE = np.complex128
np.seterr(under='warn')
__VERSION__=1.0
def transform_coordinates_cartesian_to_spherical(points):
    """
    Input
    -----
    points = N x 3 array with each row specifying: [x,y,z]

    Output
    ------
    [r,theta,phi] : with theta = angle with the z-axis
    """
    r = np.sqrt( (points**2).sum(axis=-1) )
    u = np.sqrt( (points[:,:2]**2).sum(axis=-1) )
    tht = np.arctan2(u, points[:,-1])
    phi = np.arctan2(points[:,1],points[:,0])   
    return np.vstack((r, tht, phi)).T
def transform_coordinates_spherical_to_cartesian(points):
    """
    """
    r = points[:,0]
    t = points[:,1]
    p = points[:,2]    
    return r[:,None] * np.vstack((np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t))).T
#
def spherical_unit_vectors(theta,phi):
    """
    Unit vectors for spherical coordiante system, represented in a fixed Cartesian frame
    with a 3-tuple: (v1, v2, v3), where v3 is along the z-axis that defines theta

    Vectors satisfy:
    rhat x theta_hat = phi_hat

    Input
    -----
    theta : array of angles with respect to z-axis
    phi   : array of angles of rotation around z-axis (same size as theta)

    Returns
    -------
    rhat, theta_hat, phi_hat : unit vectors each N x 3 array

    where N = size of theta, phi arrays  
    """
    rhat    = np.vstack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))).T
    tht_hat = np.vstack((np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta))).T
    phi_hat = np.vstack((-np.sin(phi), np.cos(phi), np.zeros(len(phi)))).T
    return rhat, tht_hat, phi_hat
#
def riccati1_first_order(n, z,derivative=False):
    """
    Riccati function that works with complex argument z
    """
    f = z ** 2 / 3 
    if derivative:
        df = 2 * z / 3
        return df
    return f
#
def riccati1(n, z,derivative=False):
    """
    Riccati function that works with complex argument z
    """
    f = z * sp.spherical_jn(n,z).astype(CTYPE)
    if derivative:
        df = sp.spherical_jn(n,z).astype(CTYPE) + z * sp.spherical_jn(n,z,derivative=True).astype(CTYPE)
        return df
    return f
#
def riccati3(n, z,derivative=False):
    """
    Riccati function that works with complex argument z
    """
    j = sp.spherical_jn(n,z).astype(CTYPE)
    y = sp.spherical_yn(n,z).astype(CTYPE)
    if not derivative:
        psi = z * (j + 1j*y)
        return psi
    else:
        dj = sp.spherical_jn(n,z,derivative=True).astype(CTYPE)
        dy = sp.spherical_yn(n,z,derivative=True).astype(CTYPE)
        #
        dpsi = j+1j*y + z*(dj + 1j*dy)
        return dpsi
    return
#
@dataclass
class vsh(object):
    """
    Vectors are represented as: E[i] = ith cartesian component
    i.e. Cartesian component runs along rows

    M = (lm, cartesian, point)
    A = (lm, cartesian, point)
    """
    maxL : int = None
    kind : int = None
    size_parameter : CTYPE = None
    #shape = (l, m, point)
    def __init__(self, maxL, points, size_parameter=None, coord='Cartesian', kind=1, minL = 1,n=1,ncpu=None):
        assert kind in (1,3) , 'kwarg "kind" should be 1 or 3'
        if kind==3:
            assert size_parameter is not None, 'Specify "size_paramter" for normalizing vector basis'
        self.maxL = maxL
        self.n    = n
        self.minL = minL
        self.kind = kind
        self.size_parameter = size_parameter
        self.points = points
        self.ncpu = ncpu
        if coord.upper()=='CARTESIAN':
            self._points_sph = transform_coordinates_cartesian_to_spherical(points).T
        elif coord.upper()=='SPHERICAL':
            self._points_sph = points.T
        else:
            raise NameError(coord+' not recognized')
        self.__lm_table()
        self.__gen_vector_sph()
        pass
    def __lm_table(self):
        self._lm_table = []
        for l in range(self.minL, self.maxL+1):
            for m in range(-l,l+1):
                self._lm_table += [(l,m)]
        self._lm_table = np.asarray(self._lm_table)
        return
    @property
    def Lvalues(self):
        return self._lm_table[:,0]
    @property
    def angular_momentum_table(self):
        return self._lm_table
    @staticmethod
    def vector_sph(l,m,theta,phi,callerid = 0):
        #
        rhat = np.asarray([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)]).T
        # A1 = -i(Lx, Ly, Lz) Ylm / sqrt(l*(l+1))
        # Scipy uses: theta for the angle around rotation axis
        Ylm = sp.sph_harm(m,l,phi,theta)
        Lplus_Ylm = np.zeros(Ylm.shape)
        Lminus_Ylm = np.zeros(Ylm.shape)
        if m < l:
            Lplus_Ylm = np.sqrt((l-m)*(l+m+1) ) * sp.sph_harm(m+1,l,phi,theta)
        if m > -l:
            Lminus_Ylm = np.sqrt((l+m)*(l-m+1)) * sp.sph_harm(m-1,l,phi,theta)

        A1x = (Lplus_Ylm + Lminus_Ylm)/2
        A1y = (Lplus_Ylm - Lminus_Ylm)/(2j)
        A1z = m*Ylm

        denom = 1j*np.sqrt(l*(l+1))
        A1 = np.vstack((A1x, A1y, A1z)).T / denom
        assert np.all(~np.isnan(A1)), 'nan at l,m = {}, {}'.format(l,m)
        #
        # A2 = rhat x A1 = rhat x tht_hat A1_tht + rhat x phi_hat A1_phi
        A2 = np.cross(rhat, A1)
        #
        A3 = rhat * np.tile(Ylm[:,None],(1,3))
        #
        return A1.T, A2.T, A3.T, callerid

    def __gen_vector_sph(self):
        th = self._points_sph[1]
        phi = self._points_sph[2]

        self._rhat = np.vstack((
            np.cos(phi)*np.sin(th), np.sin(phi)*np.sin(th), np.cos(th)
        ))
        self._that = np.vstack((-np.sin(th), np.cos(th), 0*th))
        self._phat = np.vstack((
            np.cos(phi)*np.cos(th), np.sin(phi)*np.cos(th), -np.sin(th)
        ))

        r  = self._points_sph[0]
        th = th.copy() + 1e-16

        Lvals = np.sort(np.unique(self._lm_table[:,0]))
        PSI = np.zeros((2,len(Lvals), r.shape[0]),dtype=CTYPE)
        Ldict = dict([(l, i) for i,l in enumerate(Lvals)])
        # time0 = datetime.datetime.now()
        n = self.n
        if self.kind==1:
            PSI[0] = np.asarray([riccati1(L, n*r) for L in Lvals])
            PSI[1] = np.asarray([riccati1(L, n*r,derivative=True) for L in Lvals])
        else:
            PSI[0] = np.asarray([riccati3(L, n*r) for L in Lvals])
            PSI[1] = np.asarray([riccati3(L, n*r,derivative=True) for L in Lvals])
            normPSI = np.asarray([riccati3(L, self.size_parameter) for L in Lvals])
            PSI[0] *= 1./normPSI[:,None]
            PSI[1] *= 1./normPSI[:,None]
        # print('Riccati',datetime.datetime.now()-time0)
        self.M = np.zeros((len(self._lm_table), 3, r.shape[0] ), dtype=CTYPE)
        self.N = np.zeros_like(self.M)
        self.Ntrans = np.zeros_like(self.N)
        self.A1 = np.zeros_like(self.M)
        self.A2 = np.zeros_like(self.M)
        self.A3 = np.zeros_like(self.M)
        INDX = -1

        sel =  r < 1e-16
        rshift = n*r
        rshift[sel] = 1e-16
        ncpu = self.ncpu
        if ncpu==1:
            workers = None
        elif ncpu is None:
            ncpu = min(len(self._lm_table)//2, cpu_count()//2)
            workers = Pool(ncpu)
            ncpu = workers._processes
        chksum = np.zeros(len(self._lm_table),dtype=bool)
        args = []
        for INDX,(l1,m1) in enumerate(self._lm_table):
            args+=[(l1,m1,th,phi,(INDX,l1,m1))]
            if len(args)==ncpu or INDX==len(self._lm_table)-1:
                if workers is None:
                    res = [self.vector_sph(*arg) for arg in args]
                else:
                    res = workers.starmap(self.vector_sph, args)
                args = []
                for A1lm, A2lm, A3lm,(i,l,m) in res:
                    j = Ldict[l]
                    # A1lm, A2lm, A3lm = self.vector_sph(l,m,th,phi)
                    Mlm = A1lm * PSI[0][j]/rshift
                    Nlm = A2lm * PSI[1][j]/rshift + np.sqrt(l*(l+1.)) * PSI[0][j]/rshift**2 * A3lm
                    # PSI[0] ~ O(r^2)
                    # PSI[1] ~ 2/3 r + O(r^2)                    
                    Mlm[:,sel] = 0.
                    Nlm[:,sel] = 2/3 * A2lm[:,sel]

                    chksum[i] = True

                    self.M[i] = Mlm
                    self.N[i] = Nlm

                    self.A1[i] = A1lm
                    self.A2[i] = A2lm
                    self.A3[i] = A3lm

                    # Transverse component is useful in some analyses with far field
                    Nlm = A2lm * PSI[1][j]/rshift
                    Nlm[:,sel] = 2/3 * A2lm[:,sel]
                    self.Ntrans[i] = Nlm

        if workers is not None:
            workers.close()
        assert  chksum.sum() == len(self._lm_table), 'Missing Spherical Harmonics at some (l,m) values'
        return
    #
    @staticmethod
    def _safe_mult(a,b):
        _low = -100.
        a = np.log(a + 1.e-20)
        b = np.log(b + 1.e-20)
        c = a + b
        sel3 = c.real > _low
        c[sel3] = np.exp(c[sel3])
        c[~sel3] = 0.
        return c
    #
    def realspace_field(self,coeff_M, coeff_N ):
        """
        Returns
        -------
        E = 3 x N matrix
        """
        F1 = self._safe_mult(self.M, coeff_M[:,None,None])
        F2 = self._safe_mult(self.N, coeff_N[:,None,None])
        F = F1.sum(axis=0) + F2.sum(axis=0)
        return F
    #
    def gen_transverse_functions(self):
        self.Mtrans = self.M
        self.A1trans = self.A1
        self.A2trans = self.A2
        self.A3trans = 0
        self.Ntrans = np.zeros_like(self.N)
        for ik, vec in enumerate(self.points):
            vec = vec[:,None]
            P = Id - np.matmul(vec, vec.T)
            for i in range(len(self._lm_table)):
                self.Ntrans
#
def mie_alpha_beta(l, nratio, ka):
    """
    Returns: (alpha1,alpha2, beta1, beta2)
    """
    psi1 = riccati1(l, ka)
    Dpsi1 = riccati1(l, ka, derivative=True)

    n_psi1 = riccati1(l, nratio*ka)
    n_Dpsi1 = riccati1(l, nratio*ka, derivative=True)

    psi3 = riccati3(l, ka)
    Dpsi3 = riccati3(l, ka, derivative=True)

    alpha1 = ( psi1 * n_Dpsi1 - 1/nratio * Dpsi1 * n_psi1 )
    alpha2 = ( psi1 * n_Dpsi1 -   nratio * Dpsi1 * n_psi1 )
    beta1  = (psi3 * n_Dpsi1 - 1/nratio * Dpsi3 * n_psi1)
    beta2  = (psi3 * n_Dpsi1 -   nratio * Dpsi3 * n_psi1)
    return (alpha1,alpha2, beta1, beta2)

def mie_sca(l, nratio, ka, normalized=False):
    """
    Returns
    T_sca = scattered field = (elm, flm)
    T_int = Internal field = (clm,dlm)

    Esca = elm * alm , flm * blm
    """ 
    psi1 = riccati1(l, ka)
    Dpsi1 = riccati1(l, ka, derivative=True)

    n_psi1 = riccati1(l, nratio*ka)
    n_Dpsi1 = riccati1(l, nratio*ka, derivative=True)

    psi3 = riccati3(l, ka)
    Dpsi3 = riccati3(l, ka, derivative=True)

    # Scattered Field
    if normalized:
        elm = - ( psi1 * n_Dpsi1 - 1/nratio * Dpsi1 * n_psi1 ) / (n_Dpsi1 - 1/nratio * Dpsi3/psi3 * n_psi1)
        flm = - ( psi1 * n_Dpsi1 -   nratio * Dpsi1 * n_psi1 ) / (n_Dpsi1 -   nratio * Dpsi3/psi3 * n_psi1)
    else:
        elm = - ( psi1 * n_Dpsi1 - 1/nratio * Dpsi1 * n_psi1 ) / (psi3 * n_Dpsi1 - 1/nratio * Dpsi3 * n_psi1)
        flm = - ( psi1 * n_Dpsi1 -   nratio * Dpsi1 * n_psi1 ) / (psi3 * n_Dpsi1 -   nratio * Dpsi3 * n_psi1)

    return (elm, flm)
#
def mie_int(l, nratio, ka):
    """
    Returns
    T_sca = scattered field = (elm, flm)
    T_int = Internal field = (clm,dlm)

    Esca = elm * alm , flm * blm
    """ 
    n_psi1 = riccati1(l, nratio*ka)
    n_Dpsi1 = riccati1(l, nratio*ka, derivative=True)

    psi3 = riccati3(l, ka)
    Dpsi3 = riccati3(l, ka, derivative=True)

    # Scattered Field
    clm = -1j          / (psi3 * n_Dpsi1 - 1/nratio * Dpsi3 * n_psi1)
    dlm = -1j * nratio / (psi3 * n_Dpsi1 -   nratio * Dpsi3 * n_psi1)
    return (clm, dlm)
#
def apply_tmatrix(t, lvals_t, coeffs, lvals_coeffs):
    """
    t = tmatrix: t[0] = e_l, t[1] = f_l
    coeffs[0] = M coefficients
    coeffs[1] = N coefficients
    Returns
    -------
    A = array of length len( lvals_coeffs )
    """
    A = np.zeros(len(lvals_coeffs),dtype=CTYPE)
    B = np.zeros(len(lvals_coeffs),dtype=CTYPE)
    IND_t = dict([(l,i) for i,l in enumerate(lvals_t)])
    for i,l in enumerate(lvals_coeffs):
        l_t = IND_t[l]
        A[i] = t[0][l_t]*coeffs[0][i]
        B[i] = t[1][l_t]*coeffs[1][i]
    return A,B,lvals_coeffs
#
def pw_to_vsh(maxL, kvec, ehat,kind=1, return_vshobj=False, ncpu=1):
    """
    Inputs
    -------
    kvec = [Nx3] array with 3-vector per row
    ehat = [Nx3] array with 3-vector per row
    Returns
    -------
    aPW_ehat[i] =  4pi i^l     conj(A1lm(khat)).ehat (lm,kvec[i])
    bPW_ehat[i] = -4pi i^(l+1) conj(A2lm(khat)).ehat (lm,kvec[i])
    """
    kvec = np.array(kvec)
    ehat = np.array(ehat)
    assert kvec.ndim==2, 'kvec should be in the form of Nx3 matrix'
    assert ehat.ndim==2, 'ehat should be in the form Nx3, same as kvec shape'
    vobj  = vsh(maxL, np.atleast_2d(kvec), coord='Cartesian',kind=kind, ncpu=ncpu)
    # (lm, Cartesian, kvec-point)   
    ehatT = ehat.T
    aPW = (vobj.A1.conj() * ehatT[None,:,:]).sum(axis=1)
    bPW = (vobj.A2.conj() * ehatT[None,:,:]).sum(axis=1)

    L  = vobj.Lvalues[:,None]
    aPW *=  4 * pi * np.exp(1j * pi/2*L    )
    bPW *= -4 * pi * np.exp(1j * pi/2*(1+L))
    
    if return_vshobj:
        return aPW, bPW, vobj
    return aPW, bPW, vobj.Lvalues   
def vsh_to_pw(Mcoeff, Ncoeff, vobj):
    raise NotImplementedError('Not implemented yet')
#
def sigma_ext(maxL,nratio, k0,radius):
    """
    """
    L = np.arange(1,maxL+1)
    ef = np.asarray([mie_sca(L1, nratio, k0*radius) for L1 in L])
    sigma = np.real( ((2*L+1) * (ef).sum(axis=1) ).sum() )
    sigma *= -2*pi/k0**2
    return sigma
#
def sigma_sca(maxL,nratio, k0,radius):
    """
    """
    L = np.arange(1,maxL+1)
    ef = np.asarray([mie_sca(L1, nratio, k0*radius) for L1 in L])
    sigma =  ((2*L+1) * ( np.abs(ef)**2 ).sum(axis=1) ).sum()
    sigma *= 2*pi/k0**2
    return sigma
#
def mie_smatrix(k0, tmat, lvals, theta):
    """
    """
    S_par = np.zeros(theta.shape,dtype=CTYPE)
    S_per = np.zeros_like(S_par)
    x = np.cos(theta)
    for i,l in enumerate(lvals):
        if l==0:continue
        Pl = sp.legendre(l)
        DPl = np.polyder(Pl)
        
        piL = np.polyval(DPl, x)
        taL = l*(l+1)*np.polyval(Pl,x) - x*np.polyval(DPl,x)

        t1L, t2L = tmat[0][i], tmat[1][i]

        spar = t1L * piL + t2L * taL
        sper = t1L * taL + t2L * piL

        prefacL = (2*l+1)/(l*(l+1))
        S_par += prefacL * spar
        S_per += prefacL * sper

    S_par = 1/(1j*k0) * S_par
    S_per = 1/(1j*k0) * S_par
    return S_par, S_per
#
def interpret_pol(ehat, khat):
    if isinstance(ehat, str):
        indx = {'x' : 0, 's' : 0, 'y' : 1, 'p' : 1}
        if ehat.lower() in ('x','y'):
            _ehat = em.create_polbasis_xy(khat)
            ehat = _ehat[indx[ehat.lower()]]
        elif ehat.lower() in ('s','p'):
            _ehat = em.create_polbasis_sp(khat)
            ehat  = _ehat[indx[ehat.lower()]]
        else:
            raise NameError('Cannot handle polarization '+ehat)
    return ehat
#
def far_field_mie(maxL, nratio, radius, khatOUT, khatINC, ehatINC, minL=1):
    """
    Inputs
    ------
    maxL   : maximum value of L for spherical harmonics
    nratio : sphere refractive index to ambient ratio
    radius : unit of wavelength
    khatOUT: direction vectors for OUTGOING wavevectors shape = [3, nk]
    khatINC: direction vector for incoming wave: shape [3,]
    ehat   : Polarization vector of INCOMING wave
             can be a character: s,p,x,y
    
    Returns
    -------
    FarField = [carteisan, direction]
    """
    assert maxL>0
    ehatINC = interpret_pol(ehatINC, khatINC)

    k0 = 2*pi
    a, b, vobjk = pw_to_vsh(maxL, khatINC, ehatINC, return_vshobj=True)
    a, b = a[:,0], b[:,0]    
    lvals = vobjk.Lvalues

    vobj_k = vsh(maxL, np.atleast_2d(khatOUT),size_parameter=k0*radius, coord='Cartesian',kind=3,minL=minL)
    
    FarField = np.zeros(vobj_k.A1.shape[1:], dtype=np.complex)
    e = []
    f = []
    for i,l in enumerate(lvals):
        tsca = mie_sca(l, nratio, radius * k0)
        ei = tsca[0] * a[i] * np.exp(-1j*pi/2 * l)
        fi = tsca[1] * b[i] * np.exp( 1j*pi/2 * (-l+1))
        e += [tsca[0] * a[i]]
        f += [tsca[1] * b[i]]
        FarField += 1/(1j*k0) * ( vobj_k.A1[i] * ei + vobj_k.A2[i] * fi )
    e = np.array(e)
    f = np.array(f)
    return FarField, (a,b,e,f)
#
def far_field_mie_fwd(maxL, nratio, radius, ehatINC):
    k0 = 2*pi
    F = 0
    for l in range(1, maxL+1):
        tsca = mie_sca(l, nratio, radius * k0)
        F += 1/(2j*k0) * (2*l+1) * (tsca[0]+tsca[1])
    F = np.asarray(ehatINC) * F
    return  F
#
def exact_integral_on_spherical_surface(maxL, n, shell_radius, kvec_out, transverse = False):
    """
    """
    SphericalBesselJ = lambda n, z : sp.spherical_jn(n,z).astype(CTYPE)
    vobjk = vsh(maxL, shell_radius*kvec_out)
    # 
    if transverse:
        Nvec = vobjk.Ntrans
    else:
        Nvec = vobjk.N
    # Exact Integral
    Wa_shell = np.zeros_like(vobjk.A1)
    Wb_shell = np.zeros_like(vobjk.A2)
    #
    x = vobjk._points_sph[0][0]
    k0 = em.vecmag_abs(kvec_out)[0]
    nx = n*x    
    for i, (l,_) in enumerate(vobjk.angular_momentum_table):
        #--------------------------------------------------------------------
        # Wa_shell = Exact shell Integral for M term
        Wa_shell[i,:,:] = exp(1j*pi/2*(-l))/k0 * (n**2-1) * (x**2 * SphericalBesselJ(l, nx) * vobjk.M[i]) 
        #--------------------------------------------------------------------
        # Wb = Exact integral for N term
        # There is A2 term and A3 term
        # A3 term, prefactors       
        if transverse:
            khat_term = 0
        else:
            khat_term = sp.spherical_jn(l, x, derivative=True) * vobjk.A3[i]  
        psi_nx  = riccati1(l, nx)/nx**2
        A3term  = x**2 * psi_nx  * ( l*(l+1) * SphericalBesselJ(l, x)/x * vobjk.A2[i] + sqrt(l*(l+1.0)) * khat_term  )
        A2term  = x**2 * riccati1(l, nx, derivative=True)/nx *  Nvec[i]
        # Total Wb term
        Wb_shell[i,:,:] = exp(1j*pi/2*(-l+1))/k0 * (n**2-1) * (A2term + A3term)
        #---------------------------------------------------------------------
    return Wa_shell, Wb_shell
#
def compute_fields(maxL, points, n, save_results=None):
    wavelength = 1.0
    kvecINC    = [[0.,0.,-1.]]
    ehatINC    = [[0.,1.,0.]]
    k0         = 2*pi*wavelength
    radius     = 1.0
    ka         = k0*radius
    a, b, Lpw = pw_to_vsh(maxL, kvecINC, ehatINC)
    tint = np.array([mie_int(l,n,ka) for l in Lpw])
    tsca = np.array([mie_sca(l,n,ka,normalized=True) for l in Lpw])
    e,f = tsca[:,0]*a[:,0], tsca[:,1]*b[:,0]
    c,d = tint[:,0]*a[:,0], tint[:,1]*b[:,0]

    r = em.vecmag_abs(points)
    sel_i = r <= radius
    sel_o = ~sel_i
    vinc  = vsh(maxL, points, kind=1)
    vi = vsh(maxL, points[sel_i]*k0, kind=1)
    vs = vsh(maxL, points[sel_o]*k0, kind=3, size_parameter=ka)

    E  = np.zeros((3,len(points)), dtype=np.complex)
    Einc = vinc.realspace_field(a.squeeze(),b.squeeze())
    Ei = vi.realspace_field(c,d)
    Es = vs.realspace_field(e,f)
    E[:,sel_i] = Ei
    E[:,sel_o] = Es + Einc[:,sel_o]
    if save_results:
        pickle.dump(E, open(save_results,'wb'))
    return E
#
if __name__=='__main__':
    pass

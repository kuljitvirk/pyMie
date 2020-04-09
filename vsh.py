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

#=============================================================================
# Vector Spherical Harmonics values
#=============================================================================
def vector_sph_values():
    maxL  = 3   
    theta = [pi/4.] #np.linspace(0, pi, 5)
    phi   = [0.0] #np.linspace(0, 2*pi, 5)
    R     = np.vstack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))).T
    print('R = ',R)
    vobj  =  vsh(maxL, R, kind=1)
    print('\nA1')
    print('{:>3s} {:>3s} {:>14s}{:>14s} {:>14s}{:>14s} {:>14s} {:>14s}'.format('l','m','Re-x','Re-y','Re-z','Im-x','Im-y','Im-z'))
    for i,(l,m) in enumerate(vobj.angular_momentum_table):
        if m<0: continue
        print('{:3d} {:3d}'.format(l,m),
            '{:14.3f} {:14.5f} {:14.3f}'.format(*vobj.A1[i,:,0].real),
            '{:14.3f} {:14.3f} {:14.3f}'.format(*vobj.A1[i,:,0].imag))
    
    print('\nA2')
    print('{:>3s} {:>3s} {:>14s}{:>14s} {:>14s}{:>14s} {:>14s} {:>14s}'.format('l','m','Re-x','Re-y','Re-z','Im-x','Im-y','Im-z'))
    for i,(l,m) in enumerate(vobj.angular_momentum_table):
        if m<0: continue
        print('{:3d} {:3d}'.format(l,m),
            '{:14.3f} {:14.5f} {:14.3f}'.format(*vobj.A2[i,:,0].real),
            '{:14.3f} {:14.3f} {:14.3f}'.format(*vobj.A2[i,:,0].imag))
    return
def integral_identity(maxL=1,N=100,ksph = np.array([[1., pi/3, 0.]]), verbose=0):
    kcart = transform_coordinates_spherical_to_cartesian(ksph)
    t = np.linspace(0,pi,N)
    p = np.linspace(0,2*pi,N)
    dtheta = t[1]-t[0]
    dphi   = p[1]-p[0]

    t, p = [c.flatten() for c in np.meshgrid(t,p)]

    R = np.vstack((t*0+1, t, p)).T    
    R = transform_coordinates_spherical_to_cartesian(R)
    kR = np.matmul(R, kcart.T).squeeze()
    F  = np.exp(-1j*kR) * np.sin(t) * dphi * dtheta

    vobjk = vsh(maxL, kcart, kind=1)

    vobj = vsh(maxL, R, kind=1)
    I1 = (F[None,None,:] * vobj.A1).sum(axis=-1)
    I2 = (F[None,None,:] * vobj.A2).sum(axis=-1)
    err1 = np.zeros(I1.shape[0])
    err2 = np.zeros(I1.shape[0])
    if verbose:
        print('k = ',kcart,'\n','-'*100)
    for i, (l,m) in enumerate(vobj.angular_momentum_table):
        if m<0:continue
        Exact1 = 4*pi*np.exp(-1j*pi/2*l) * vobjk.M[i,:,0]
        Exact2 = 4*pi*np.exp(1j*pi/2*(-l+1)) * vobjk.N[i,:,0]
        err1[i] = np.abs(I1[i]-Exact1).max()
        err2[i] = np.abs(I2[i]-Exact2).max()
        if verbose:
            print('({:3d},{:3d})'.format(l,m))
            print('Exact1',Exact1)
            print('I1    ',I1[i])
            print()
            print('Exact2',Exact2)
            print('I2    ',I2[i])
            print('-'*100)
    return err1,err2
#=============================================================================
# PLANEWAVE EXPANSION TEST
#=============================================================================
def test_pw(maxL=50,maxR=5,nR = 20, dim = 1, loc=0.):
    """
    Test the planewave expansion in terms of spherical harmonics
    Sets wavelength = 1, and computes vector basis with argument k0 * r
    Therefore the spatial grid is in units of wavelength
    
    In a domain that can be a line or a 3D box:
        Creates s and p polarized planewaves over range of angles
        Defines exact plane wave over the domain
        Computes PW coefficients for each kvector
        Computes the real space field from PW coefficients
        Computes the maximum error between the exact and VSH result
    """
    wavelength = 1.0
    k0 = 2*pi/wavelength
    theta = np.linspace(0.,pi/2*0.9,9)
    khat = np.vstack((np.sin(theta), 0*theta, -np.cos(theta)))
    kvec = k0 * khat.T
    _,ehat = em.create_polbasis_sp(kvec)
    aPW, bPW,_ = pw_to_vsh(maxL, kvec, ehat)

    u = np.linspace(0.01,maxR,nR)
    if dim==1: # Z-axis
        points = np.vstack([c.flatten() for c in np.meshgrid([loc],[loc],u)]).T
    elif dim==2: # XZ plane
        points = np.vstack([c.flatten() for c in np.meshgrid(u,[loc],u)]).T
    elif dim==3: # 3D box full
        points = np.vstack([c.flatten() for c in np.meshgrid(u,u,u)]).T
    vobj = vsh(maxL, k0*points,kind=1)
    
    exactField = np.zeros((len(kvec),3, len(points)),dtype=CTYPE)
    for ik in range(len(kvec)):
        f = np.exp(1j* (kvec[ik,0]*points[:,0] + kvec[ik,1]*points[:,1] + kvec[ik,2]*points[:,2]))
        for j in range(3):
            exactField[ik,j] = ehat[ik,j] * f

    field = np.zeros((len(kvec), 3, len(points)), dtype=CTYPE)
    for ik in range(len(kvec)):
        field[ik] = vobj.realspace_field(aPW[:,ik], bPW[:,ik])
    maxError = np.abs(field - exactField).max()
    return maxError,field, exactField, kvec, points

def run_test_pw(save : str = None):
    import matplotlib as mpl
    mpl.use('Qt5Agg')
    from matplotlib import pyplot as plt
    import pickle

    arrayError = []
    for maxL in [20,30,60,70,80,85]:
        maxErr, field, exactField, kvec, points = test_pw(maxL=maxL, maxR = 5,loc=5, nR=11, dim=1)
        arrayError +=[(maxL,maxErr)]
        print('maxL = ',maxL, 'maxErr = ',maxErr,'max(R)',points.max(),'# field.shape',field.shape)
    arrayError = np.asarray(arrayError)
    if save:
        pickle.dump((arrayError,field,exactField,kvec,points),open('test_pw.pkl','wb'))
        print('Saved results to file',save)
    return
#=============================================================================
def check_identities():
    #run_test_pw(save='test_pw.pkl')
    #vector_sph_values()
    t = np.linspace(0,pi,3)
    p = np.linspace(0,2*pi,3)
    t,p=[c.flatten() for c in np.meshgrid(t,p)]
    ksph = np.vstack((1+0.*t, t, p)).T
    #ksph = np.array([[1., pi/3, 0.]])
    print('Num kpoints',ksph.shape[0])
    for N in [128,256,512]:
        E = []
        for ksph1 in ksph:
            err1,err2=integral_identity(maxL=1,N=N,ksph = np.atleast_2d(ksph1),verbose=0)
            E += [[err1.max(),err2.max()]]
        E = np.array(E)
        print('N',N, 'max(error)',E.max(axis=0))
    pass

def check_values():
    ksph = np.array([[2., pi/3, 0.]])
    kcart = transform_coordinates_spherical_to_cartesian(ksph)
    vobj = vsh(3, kcart)

    for i,(l,m) in enumerate(vobj.angular_momentum_table):
        print('A1 {:d}{:d}:'.format(l,m),vobj.A1[i,:,0])
    for i,(l,m) in enumerate(vobj.angular_momentum_table):
        print('A2 {:d}{:d}:'.format(l,m),vobj.A1[i,:,0])
    for i,(l,m) in enumerate(vobj.angular_momentum_table):
        print('M1 {:d}{:d}:'.format(l,m),vobj.M[i,:,0])
    for i,(l,m) in enumerate(vobj.angular_momentum_table):
        print('N1 {:d}{:d}:'.format(l,m),vobj.N[i,:,0])

def check_radial_integral_identity1(
    maxL = 1,
    nratio = 2.,
    N = 100,
    maxr = 2,
    ):
    """

    """

    falpha1 = lambda l,x : n*x**2 * (sp.spherical_jn(l-1,n*x) * sp.spherical_jn(l,x) - 1/n * sp.spherical_jn(l-1,x) * sp.spherical_jn(l,n*x)) 
    x = np.linspace(0, maxr, N)
    dx = x[1]-x[0]
    n = nratio
    Integrand = (1-n**2) * np.array([ dx * x**2 * sp.spherical_jn(l,x) * sp.spherical_jn(l,n*x) for l in range(1,maxL+1)])
    Integral = np.cumsum(Integrand, axis=1)
    alpha1 = np.array([ falpha1(l,x) for l in range(1,maxL+1)])

    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    ax.set_title('N = {}'.format(N))
    ax.plot(x, np.real(Integral).T , '-', label='Numerical')
    ax.plot(x, np.real(alpha1).T, '--', label='exact' )
    ax.legend()
    ax.grid(True)
    plt.show()

def check_radial_integral_identity2(
    maxL = 1,
    nratio = 2.,
    N = 100,
    ):
    """

    """
    n = nratio
    falpha2 = lambda l,x : riccati1(l,x)*riccati1(l,n*x,derivative=True) - n * riccati1(l,x,derivative=True)*riccati1(l,n*x)
    x = np.linspace(1e-6, 3, N)
    dx = x[1]-x[0]    

    Integrand = (1-n**2) * np.array([ 
        dx * (riccati1(l,n*x,derivative=True)*riccati1(l,x,derivative=True) + l*(l+1)*sp.spherical_jn(l,n*x).astype(CTYPE)*sp.spherical_jn(l,x).astype(CTYPE) )
         for l in range(1,maxL+1)])

    Integral = np.cumsum(Integrand, axis=1)
    alpha2 = np.array([ falpha2(l,x) for l in range(1,maxL+1)])

    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(ncols=2,figsize=(8,4))
    ax=axs[0]
    ax.set_title('N = {}, Re'.format(N))
    ax.plot(x, np.real(Integral).T , '-', label='Numerical')
    ax.plot(x, np.real(alpha2).T, '--', label='exact' )
    ax=axs[1]
    ax.set_title('N = {}, Im'.format(N))
    ax.plot(x, np.imag(Integral).T , '-', label='Numerical')
    ax.plot(x, np.imag(alpha2).T, '--', label='exact' )
    ax.legend()
    ax.grid(True)
    plt.show()
#
def gen_surface_spherical(N, r):
    t = np.linspace(0,np.pi,N)
    p = np.linspace(0,2*np.pi,N)
    dt, dp = t[1], p[1]
    t,p = np.meshgrid(t,p)
    x = r*np.sin(t)*np.cos(p)
    y = r*np.sin(t)*np.sin(p)
    z = r*np.cos(t)
    _points=np.vstack([c.flatten() for c in (x,y,z)]).T
    shape = x.shape
    measure = r**2 * np.sin(t) * dt * dp
    return _points,shape,measure.flatten()
#
def integral_on_spherical_surface(maxL, n, shell_radius, Npoints, kvec_out, transverse=False):
    k0 = 2*pi
    x, _, measure = gen_surface_spherical(Npoints, k0*shell_radius)
    vobj  = vsh(maxL, x, n = n)
    vobjk = vsh(maxL, shell_radius*kvec_out)
    khat  = kvec_out / em.vecmag_abs(kvec_out)[:,None]
    I1 = np.zeros_like(vobjk.A1)
    I2 = np.zeros_like(vobjk.A2)
    I3x3 = np.eye(3)
    for ik,kh in enumerate(khat):
        kh = kh[:,None]
        kr = np.dot(x, kh).squeeze()
        fac = np.exp(-1j*kr)*measure
        I1[:,:,ik] = (vobj.M*fac[None,None,:]).sum(axis=-1) / (4*pi) * (n**2-1)
        I2[:,:,ik] = (vobj.N*fac[None,None,:]).sum(axis=-1) / (4*pi) * (n**2-1)
        if transverse:
            P = I3x3 - np.matmul(kh, kh.T)
            for ii in range(len(I2)):
                I2[ii,:,ik] = np.matmul(I2[ii,:,ik], P)
    #
    return I1, I2, vobjk
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
    #check_identities()
    #run_test_pw()
    #check_radial_integral_identity1(maxL=1,nratio=2.,N=250, maxr=2*pi)
    #check_radial_integral_identity2(maxL=5,nratio=2.+1j,N=1000)
    # from matplotlib import pyplot as plt
    # plt.plot(M[:,:,0].real,'--')
    # plt.plot(N[:,:,0].real,'-')
    # plt.show()

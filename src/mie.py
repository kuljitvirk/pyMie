"""
Mie T-matrix and solutions for plane wave incidence are also provided
A term in the expansion is given by
E = alm Mlm(kr) + blm Nlm(kr)
where alm and blm depend on the ratio of sphere to background refractive index, and size parameter in free space
So parameters are:
(ka, nsphere/nbackground, position vector, incidence direction)
The numerical paramter is maxL = maximum value of the order of spherical harmonics
"""
import numpy as np
from numpy import exp, sqrt
from math import pi
import datetime
from dataclasses import dataclass
from scipy import special as sp
import pickle
from scipy.integrate import simps
from multiprocessing import Pool, cpu_count
from .vsh import *
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
def far_field(maxL, nratio, radius, khatOUT, khatINC, ehatINC, minL=1):
    """
    Args:
        maxL   : maximum value of L for spherical harmonics
        nratio : sphere refractive index to ambient ratio
        radius : radius / wavelength ( = size_parameter / ($\pi$))
        khatOUT: direction vectors for OUTGOING wavevectors shape = [3, nk]
        khatINC: direction vector for incoming wave: shape [3,]
        ehat   : Polarization vector of INCOMING wave
                can be a character: s,p,x,y
    
    Returns:
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
    for i,l in enumerate(lvals):
        tsca = mie_sca(l, nratio, radius * k0)
        ei = tsca[0] * a[i] * np.exp( 1j*pi/2 * (-l))
        fi = tsca[1] * b[i] * np.exp( 1j*pi/2 * (-l+1))
        FarField += 1/(1j*k0) * ( vobj_k.A1[i] * ei + vobj_k.A2[i] * fi )
    return FarField
#
def far_field_dyadic(maxL, n, radius, khatOUT, khatINC):
    """
    Args:
        maxL (int) : maximum total angular momentum number
        n (complex) : ratio of sphere refractive index to background
        radius (float) : radius / wavelength
        khatOUT (array) : outgoing kvectors
        khatINC  (array) : incoming kvector
    Returns:
        S = $[S_{\per\per},S_{\par\per},S_{\per\par},S_{\par\par}]$
    """
    eo, ei = em.dyadic_vectors(khatOUT, khatINC)
    S = np.zeros((len(khatOUT),4),dtype=np.complex)
    for i, khat in enumerate(khatOUT):
        Fper = far_field(maxL, n, radius, khat[None,:], khatINC, ei[i][0][None,:]).squeeze()
        Fpar = far_field(maxL, n, radius, khat[None,:], khatINC, ei[i][1][None,:]).squeeze()
        S[i] = [(eo[i,0]*Fper).sum(),(eo[i,1]*Fper).sum(),(eo[i,0]*Fpar).sum(),(eo[i,1]*Fpar).sum()]
    #
    return S
#
def far_field_fwd(maxL, nratio, radius, ehatINC):
    k0 = 2*pi
    F = 0
    for l in range(1, maxL+1):
        tsca = mie_sca(l, nratio, radius * k0)
        F += 1/(2j*k0) * (2*l+1) * (tsca[0]+tsca[1])
    F = np.asarray(ehatINC) * F
    return  F
#
def exact_integral_on_spherical_surface(maxL, n, shell_radius, kvec_out, transverse = False,ncpu=None):
    """
    """
    kvec_out = np.asarray(kvec_out)
    SphericalBesselJ = lambda n, z : sp.spherical_jn(n,z).astype(CTYPE)
    vobjk = vsh(maxL, shell_radius*kvec_out,ncpu=ncpu)
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
def compute_fields(
    maxL, 
    points, 
    radius, 
    n, 
    ehatINC    = [[0.,1.,0.]],
    save_results=None,
    ncpu = None
    ):
    """
    Returns:
        E, Einc
    """
    wavelength = 1.0
    kvecINC    = [[0.,0.,-1.]]
    k0         = 2*pi*wavelength
    ka         = k0*radius
    a, b, Lpw  = pw_to_vsh(maxL, kvecINC, ehatINC)

    tint = np.array([mie_int(l,n,ka) for l in Lpw])
    tsca = np.array([mie_sca(l,n,ka,normalized=True) for l in Lpw])
    e,f = tsca[:,0]*a[:,0], tsca[:,1]*b[:,0]
    c,d = tint[:,0]*a[:,0], tint[:,1]*b[:,0]

    points = np.asarray(points)
    r = em.vecmag_abs(points)
    sel_i = r <= radius
    sel_o = r >= radius
    vinc  = vsh(maxL, points*k0, kind=1, ncpu=ncpu)
    vi = vsh(maxL, points[sel_i]*k0, kind=1,n=n, ncpu=ncpu)
    vs = vsh(maxL, points[sel_o]*k0, kind=3, size_parameter=ka, ncpu=ncpu)

    Einc = vinc.realspace_field(a.squeeze(),b.squeeze())
    Ei = vi.realspace_field(c,d)
    Es = vs.realspace_field(e,f)
    E  = Einc.copy()
    E[:,sel_i]  = Ei 
    E[:,sel_o]  += Es
    return E, Einc #,(Ei, Es, Einc)
#
def far_field_volume_integral(
    radius  = 1,
    n       = 3+0.5j,
    maxL    = 5,
    N       = 100,
    nko     = 4,
    khatOUT = [[0.,0.,1.]],
    khatINC = [[0.,0.,-1.]],
    ehatINC = [[0.,1.,0.]],
    integrate = simps):
    """
    The integration over radial direction must be performed by a method of higher order than Riemann sums.
    The oscillatory nature of the integrals accumulate to significant relative errors if integration rule
    is not accurate enough. I have found that Simpson's rule gives adequate results with N=100. The maximum 
    absolute error is 0.06% of the exact result at the point where maximum error occurs.

    Result should match the statement:
    ::
        Fexact = far_field(maxL, n, radius,kvecO, khatINC,ehatINC)
    """
    k0      = 2*pi
    # Mie coefficients for the sphere's interior
    # - Incident Plane Wave amplitudes
    a, b, Lvalues = pw_to_vsh(maxL, khatINC, ehatINC, ncpu=1)
    a, b = a[:,0], b[:,0]
    # - T-matrix
    _Tint = np.array([mie_int(l, n, k0*radius) for l in Lvalues])
    c,d = _Tint[:,0]*a, _Tint[:,1]*b
    #
    shell_radii = np.linspace(0.001,radius, N)
    F = np.zeros((len(shell_radii),3),dtype=np.complex)
    wa = []
    wb = []
    for ir, shell_radius in enumerate(shell_radii):
        # Document: what does transverse=True do and what does that mean?
        _wa, _wb = exact_integral_on_spherical_surface(maxL, n, shell_radius, khatOUT, transverse=True, ncpu=1)
        wa += [_wa]
        wb += [_wb]
    # wa = [r, lm, cartesian, ko]
    wa = np.array(wa)
    wb = np.array(wb)
    Fr = (wa*c[None,:,None,None] + wb*d[None,:,None,None]).sum(axis=1)
    # Wa = [lm, cartesian,ko]
    Wa = integrate(wa, k0*shell_radii, axis=0)
    Wb = integrate(wb, k0*shell_radii, axis=0)
    # Perform the sum over (l,m)
    F = (Wa * c[:,None,None] + Wb * d[:,None,None]).sum(axis=0)
    return F

if __name__=='__main__':
    pass

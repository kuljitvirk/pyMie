"""
Kuljit S. Virk
"""
from itertools import product
import numpy as np
import os
SMALL_TOL12 = 1.e-12
SMALL_TOL8 = 1.e-8
SMALL_TOL4 = 1.e-4
SMALL_TOL6 = 1.e-6
cmplx_zeros = lambda shape : np.zeros(shape, dtype = np.complex)
def custom_print(*args, **kwargs):
    print('emops:',*args,**kwargs)
#-----------------------------------------------------------------------------
# Utility functions 
#-----------------------------------------------------------------------------
def cmplx_sqrt(z):
    """
    >>> cmplx_sqrt(25+12.j)
    (5.134727317381328+1.1685138526615968j)
    """
    magz = np.abs(z)
    zr = np.real(z)
    zi = np.imag(z)
    argz = np.arctan2(zi,zr)
    if np.isscalar(z):
        if argz < 0:
            argz += 2*np.pi
    else:
        sel = argz<0
        argz[sel] += 2*np.pi
    argz = 0.5*argz
    return np.sqrt(magz) * (np.cos(argz) + 1j* np.sin(argz))
def vecmag(vec,axis=-1):
    return np.sqrt(np.sum(np.array(vec)**2,axis=axis))
#
def vecmag_abs(vec,axis=-1):
    return np.sqrt(np.sum(np.abs(np.array(vec))**2,axis=axis))
#
def to_unitvector(K):
    Kmag = vecmag( np.atleast_2d(K))
    Kmag[Kmag < SMALL_TOL12] = 1.
    K2 = np.atleast_2d(K)
    Khat = K2 / np.tile( Kmag[:,np.newaxis], K2.shape[1] )
    return Khat
#
def to_na_units(K):
    """
    """
    K = np.atleast_2d(K)
    K0 = vecmag(K)
    return K / np.tile(K0[:,None],(1,K.shape[1]))
#
def convert_to_na_units(K):
    """
    """
    K0 = vecmag(K)
    if not np.isscalar(K0):
        K0 = np.tile(K0[:,None], (1, 3))
    return K / K0
#
def na(k):
    if k.ndim==1:
        return vecmag(k[:2])/vecmag(k)
    
    return vecmag(k[:,:2])/vecmag(k)
#
def aoi(k):
    return np.arcsin(na(k))
#
def get_wavelength(k):
    return np.real(2*np.pi/vecmag(k))
#
def tangential(normalvec, E):
    proj = (normalvec*E).sum(axis=-1)
    Et = E - np.tile(proj[:,np.newaxis],(1,3))*normalvec
    return Et
#
def create_wavevector(wavelength, _k2d, sign_kz):
    """
    """
    k2d = np.atleast_2d(np.array(_k2d))[:,:2]
    k0 = 2*np.pi/wavelength
    kzsq = (k0**2 - vecmag(k2d)**2).astype(np.complex)
    kvec = np.zeros((k2d.shape[0], 3), dtype=np.complex)
    kvec[:,:2] = k2d
    kvec[:,-1] = cmplx_sqrt(kzsq) * sign_kz
    if ( np.abs(np.imag(kvec)).max() < 1.e-16):
        kvec = np.real(kvec)
    return kvec
def magnify_kvector(kvec, Mag):
    kvec_mag = np.zeros((kvec.shape[0], 3), dtype=np.complex)
    kvec_mag[:,:2] = kvec[:,:2]/Mag
    sign_kz = np.ones(kvec.shape[0])
    sign_kz[np.real(kvec[:,-1]) < -1.e-12] = -1
    magkvecsq = (kvec**2).sum(axis=-1)
    assert(np.max(magkvecsq) - np.min(magkvecsq) < 1.e-12)
    kvec_mag[:, -1] = magkvecsq - (kvec_mag[:,:2]**2).sum(axis=-1)
    kvec_mag[:, -1] = sign_kz * np.sqrt(kvec_mag[:, -1])
    return kvec_mag
#
def backscatter_order(kvec):
    """

    """
    from .generic_toolkit import match_order
    _, Ib = match_order(kvec,-kvec)
    return Ib
#-----------------------------------------------------------------------------
# FFT routines for centered grids: take care of doing/undoing wrap-arounds
#-----------------------------------------------------------------------------
def get_fftshifts(nx,ny):
    # For even n: fftshift = ifftshift
    # ifftshift([-4,-3,-2,-1,0,1,2,3]) = [ 0,  1,  2,  3, -4, -3, -2, -1]
    # ifftshift([-4,-3,-2,-1,0,1,2,3,4]) = [ 0,  1,  2,  3,  4, -4, -3, -2, -1]
    # For both cases: ifftshift puts 0 frequency at 0-index, fftshift restores the array
    shift = np.fft.ifftshift
    ishift = np.fft.fftshift
    return shift,ishift
#
def transform_fft2(Field):
    """
    Performs FFT2D on Nx x Ny x D field
    Centered input, centered output

    fft2
    """
    nx,ny = Field.shape[:2]
    shift, ishift = get_fftshifts(nx,ny)
    Field_fft = ishift( np.fft.fft2( shift( Field, axes=(0,1) ), axes=(0,1) ), axes=(0,1) )
    return Field_fft
#
def transform_ifft2(Field):
    """
    Performs iFFT2D on Nx x Ny x D field
    Centered input, centered output

    Test: f = 2D array or higher D array
    f = transform_ifft2( transform_fft2(f) ) within machine precision

    """
    nx,ny = Field.shape[:2]
    shift, ishift = get_fftshifts(nx,ny)
    Field_ifft = shift( np.fft.ifft2( ishift( Field, axes=(0,1) ), axes=(0,1) ),axes=(0,1) )
    return Field_ifft
#-----------------------------------------------------------------------------
# Creation of polarization basis
# Transformation of Field amplitudes between fixed Cartesian to k-dependent
# polarization bases
#-----------------------------------------------------------------------------
def create_polbasis_sp(K, sgamma='x'):
    """
    Computes sp basis for given 3d Kpoints

    Inputs: ( with N rows, one for each K point)
    K[i,:] = [kx, ky, kz]

    Outputs: N rows, one for each K point

    s[i,:] = [sx, sy, sz]
             s-vector written in the same Cartesian representation
             as the input K-vectors

    p[i,:] = [px, py, pz]
             p-vector written in the same Cartesian representation
             as the input K-vectors

    Definitions:

    s = kappa_hat x zhat
    p = s x kvec_hat

    kappa_hat = [kx, ky, 0]/norm([kx,ky, 0])
    kvec_hat = [kx, ky,kz]/norm([kx,ky,kz])

    Conventions:

    At kappa = (0,0,0), we set kappa_hat = (1,0,0) => s = (0,-1,0)

    """
    K = np.atleast_2d(np.array(K))
    assert(K.shape[1]==3)
    kvec_hat = to_unitvector(K)

    kappa_mag = vecmag(kvec_hat[:,:2])
    # kvec_hat[kappa_mag < SMALL_TOL1,0] = 1.e-16
    # kvec_hat = convert_to_unitvector(kvec_hat)

    kappa = kvec_hat.copy()
    kappa[:,-1] = 0
    kappa_hat = to_unitvector(kappa)

    # Replace [0,0,0] with [1,0,0] for convention at origin
    kappa_mag = vecmag(kappa_hat)
    if sgamma=='y':
        kappa_hat[kappa_mag < SMALL_TOL12,0] = 0.
        kappa_hat[kappa_mag < SMALL_TOL12,1] = -1.
    else:
        kappa_hat[kappa_mag < SMALL_TOL12,0] = 1.
        kappa_hat[kappa_mag < SMALL_TOL12,1] = 0.
    zhat = np.zeros(K.shape)
    zhat[:,-1] = 1.
    # s = [ax,ay,az] x zhat = [-ay, ax, 0]
    s = np.cross( kappa_hat, zhat )
    p = np.cross( s, kvec_hat )
    # Check:
    test = np.nanmax(np.abs(np.cross(s, kvec_hat) - p))
    assert( test < SMALL_TOL12 )
    return s, p
#
def create_polbasis_xy(K):
    """
    Works for both signs of kz:
    Generates X,Y by an improper (det = -1) rotation of s,p
    1. compute s, p 
    2. Set sign of in-plane angle of k to be the sign of kz
    2. Perform the improper rotation
    #
    When kz > 0, gives the conventional result
    
    """
    #assert(all(K[:,-1]>0))
    s, p = create_polbasis_sp(K)
    K = np.atleast_2d(K)
    sel0 = vecmag_abs(K[:,:2]) < SMALL_TOL12
    # real part here is to force a type conversion: x,y components of K
    # must always be real valued
    phi = np.arctan2(np.real(K[:,1]), np.real(K[:,0]))
    phi[sel0] = 0.
    cosphi = np.tile( np.cos(phi)[:,None], (1,3) )
    sinphi = np.tile( np.sin(phi)[:,None], (1,3) )

    sign_kz = np.tile( np.sign(K[:,2])[:,None],(1,3) )
    s = s * sign_kz
    X =  sinphi * s - cosphi * p
    Y = -cosphi * s - sinphi * p
    X = X * sign_kz
    Y = Y * sign_kz
    return X, Y
#
def dyadic_vectors(khatOUT, khatINC):
    """
    Polarization vectors for S-matrix dyad. 
    .. math:
        \mathbf{e}_{i,\perp} = \frac{\hat{\mathbf{k}}_i \times \hat{\mathbf{k}}}{|\hat{\mathbf{k}}_i \times \hat{\mathbf{k}}|}
        \mathbf{e}_{i,\par}  = \frac{\hat{\mathbf{e}}_{i,\perp} \times \hat{\mathbf{k}_i}}{|\hat{\mathbf{e}}_{i,\perp} \times \hat{\mathbf{k}_i}|}
        \mathbf{e}_{o,\perp} = \mathbf{e}_{i,\perp}
        \mathbf{e}_{o,\par}  = \frac{\hat{\mathbf{e}}_{o,\perp} \times \hat{\mathbf{k}}}{|\hat{\mathbf{e}}_{o,\perp} \times \hat{\mathbf{k}}|}
    """
    ki = khatINC.squeeze()
    ehati = []
    ehato = []
    for i, khat in enumerate(khatOUT):
        ei = np.cross(ki, khat)
        ei = [ei,np.cross(ei, ki)]
        
        eo = np.cross(ki, khat)
        eo = [eo,np.cross(eo, khat)]
        
        ei = [a.squeeze()/em.vecmag(a) for a in ei]
        eo = [a.squeeze()/em.vecmag(a) for a in eo]
        
        ehati += [ei]
        ehato += [eo]
        
    return np.array(ehato), np.array(ehati)
#
def transform_polbasis_sp_to_cartesian(K,E,sgamma='x'):
    """
    Converts field written in s-p basis to Cartesian
    basis.

    Inputs: ( with N rows, one for each K point)
    K[i,:] = [kx, ky, kz]
    E[i,:] = [Es(k[i]), Ep(k[i])]

    Output:

    E[k,:] = [Ex(k), Ey(k), Ez(k)]

    """
    assert(K.shape[1]==3)
    K = np.atleast_2d(K)
    shat, phat = create_polbasis_sp(K,sgamma=sgamma)

    if E.ndim == 1:
        E = E[np.newaxis,:]

    if np.any(np.iscomplex(phat)):
        Ecart = np.zeros((E.shape[0], 3),dtype=np.complex)
    else:
        Ecart = np.zeros((E.shape[0], 3),dtype=E.dtype)
    for i in range(3):
        Ecart[:,i] = E[:,0]*shat[:,i] + E[:,1]*phat[:,i]
        Ecart[:,i] = E[:,0]*shat[:,i] + E[:,1]*phat[:,i]
    return Ecart
#
def transform_polbasis_cartesian_to_sp(K,E, sgamma='x'):
    """
    Converts field written in  Cartesian to s-p
    basis.

    Inputs: ( with N rows, one for each K point)
    K[i,:] = [kx, ky, kz]
    E[i,:] = [Ex, Ey, Ez]

    Output:

    E[k,:] = [Ex(k), Ey(k), Ez(k)]

    """
    K = np.atleast_2d(np.array(K))
    assert(K.shape[1]==3)
    
    shat, phat = create_polbasis_sp(K,sgamma=sgamma)

    if E.ndim == 1:
        E = E[np.newaxis,:]

    if np.any(np.iscomplex(phat)):
        Esp = np.zeros((E.shape[0], 2),dtype=np.complex)
    else:
        Esp = np.zeros((E.shape[0], 2),dtype=E.dtype)

    Esp[:,0] = (shat * E).sum(axis=-1)
    Esp[:,1] = (phat * E).sum(axis=-1)

    return Esp
#
def transform_polbasis_xy_to_cartesian(K,E):
    """
    Converts field written in xy basis to Cartesian
    basis.

    Inputs: ( with N rows, one for each K point)
    K[i,:] = [kx, ky, kz]
    E[i,:] = [Es(k[i]), Ep(k[i])]

    Output:

    E[k,:] = [Ex(k), Ey(k), Ez(k)]

    """
    assert(K.shape[1]==3)
    K = np.atleast_2d(K)
    xhat, yhat = create_polbasis_xy(K)

    if E.ndim == 1:
        E = E[np.newaxis,:]

    if np.any(np.iscomplex(yhat)):
        Ecart = np.zeros((E.shape[0], 3),dtype=np.complex)
    else:
        Ecart = np.zeros((E.shape[0], 3),dtype=E.dtype)
    for i in range(3):
        Ecart[:,i] = E[:,0]*xhat[:,i] + E[:,1]*yhat[:,i]
        Ecart[:,i] = E[:,0]*xhat[:,i] + E[:,1]*yhat[:,i]
    return Ecart
#
def transform_polbasis_cartesian_to_xy(K,E):
    """
    Converts field written in  Cartesian to xy
    basis.

    Inputs: ( with N rows, one for each K point)
    K[i,:] = [kx, ky, kz]
    E[i,:] = [Ex, Ey, Ez]

    Output:

    E[k,:] = [Ex(k), Ey(k), Ez(k)]

    """
    K = np.atleast_2d(np.array(K))
    assert(K.shape[1]==3)
    xhat, yhat = create_polbasis_xy(K)

    if E.ndim == 1:
        E = E[np.newaxis,:]

    if np.any(np.iscomplex(yhat)):
        Esp = np.zeros((E.shape[0], 2),dtype=np.complex)
    else:
        Esp = np.zeros((E.shape[0], 2),dtype=E.dtype)

    Esp[:,0] = (xhat * E).sum(axis=-1)
    Esp[:,1] = (yhat * E).sum(axis=-1)

    return Esp
#
def transform_polbasis_sp_to_xy(K,E):
    Ecart = transform_polbasis_sp_to_cartesian(K,E)
    Exy = transform_polbasis_cartesian_to_xy(K,Ecart)
    return Exy
def transform_polbasis_xy_to_sp(K,E):
    Ecart = transform_polbasis_xy_to_cartesian(K,E)
    Esp = transform_polbasis_cartesian_to_sp(K,Ecart)
    return Esp
#
def transform_matrix_sp_to_xy(Ko,Ki,M):
    """
    M =  [Mss Msp]
         [Mps Mpp]

    Returns:
    M = [Mhh Mhv]
        [Mvh Mvv]
    """
    cat = np.concatenate
    s, p = create_polbasis_sp(Ki)
    h, v = create_polbasis_xy(Ki)
    s_h = np.diag((s*h).sum(axis=-1))
    s_v = np.diag((s*v).sum(axis=-1))
    p_h = np.diag((p*h).sum(axis=-1))
    p_v = np.diag((p*v).sum(axis=-1))
    # [Msh Msv] = [Mss Msp][sh sv]
    # [Mph Mpv]   [Mps Mpp][ph pv]
    U = cat((cat((s_h,s_v),axis=1),cat((p_h,p_v),axis=1)),axis=0)
    M = np.matmul(M,U)

    s, p = create_polbasis_sp(Ko)
    h, v = create_polbasis_xy(Ko)
    h_s = np.diag((s*h).sum(axis=-1))
    v_s = np.diag((s*v).sum(axis=-1))
    h_p = np.diag((p*h).sum(axis=-1))
    v_p = np.diag((p*v).sum(axis=-1))
    # [Mhh Mhv] = [hs hp][Msh Msv]
    # [Mvh Mvv]   [vs vp][Mph Mpv]
    U = cat((cat((h_s,h_p),axis=1),cat((v_s,v_p),axis=1)),axis=0)
    M = np.matmul(U,M)
    return M
#
def transform_table_xy_to_sp(Ko,Ki,T):
    """
    T[:,0] = Txx
    T[:,1] = Tyx
    T[:,2] = Txy
    T[:,3] = Tyy

    Tdyad = t0 xxi + t1 yxi + t2 xyi + t3 yyi

    """
    s, p = create_polbasis_sp(Ko)
    X, Y = create_polbasis_xy(Ko)
    Xi, Yi = create_polbasis_xy(Ki)
    s_i, p_i = create_polbasis_sp(Ki) 
    # Txx Tyx Txy Tyy -> Tsx Tpx Tsy Tpy
    sx = (s*X).sum(axis=-1)
    sy = (s*Y).sum(axis=-1)
    px = (p*X).sum(axis=-1)
    py = (p*Y).sum(axis=-1)
    # 
    sixi = (s_i * Xi).sum(axis=-1)
    siyi = (s_i * Yi).sum(axis=-1)
    pixi = (p_i * Xi).sum(axis=-1)
    piyi = (p_i * Yi).sum(axis=-1)
    # Wsx Wpx 
    Ws_xi = sx*T[:,0] + sy*T[:,1]
    Wp_xi = px*T[:,0] + py*T[:,1]
    Ws_yi = sx*T[:,2] + sy*T[:,3]
    Wp_yi = px*T[:,2] + py*T[:,3]
    # Transform the incidence side
    # Wss <--- (Wsx, Wsy)
    Wss = Ws_xi * sixi + Ws_yi * siyi 
    # Wps <--- (Wpx, Wpy)
    Wps = Wp_xi * sixi + Wp_yi * siyi
    # Wsp <--- (Wsx, Wsy)
    Wsp = Ws_xi * pixi + Ws_yi * piyi 
    # Wpp <--- (Wpx, Wpy)
    Wpp = Wp_xi * pixi + Wp_yi * piyi
    #
    W = np.vstack((Wss, Wps, Wsp, Wpp)).T
    return W
#
def basis_transform(eI, Ein, inverse=False):
    """
    Returns
    -------
    [eI[0].Ein[0],   eI[0].Ein[1]]
    [eI[1].Ein[0],   eI[1].Ein[1]]
    
    Inverts the matrix if keyword, inverse=True
        
    Let '1' and '2' represent the excitations with basis vec '1' and '2'
    Let Q = dyad that transforms inc plane wave to outgoing field
    
    Let X(1) and X(2) be the incident plane waves
    Then the outgoing fields due to excitations 1 and 2 are
    
    E(1) = G . X(1)
    E(2) = G . X(2)
    
    Insert s.s + p.p + k.k = unit dyad
    E(1) = G . [s (s.X(1)) + p (p.X(1)) + k(k.X(1))]
    E(2) = G . [s (s.X(2)) + p (p.X(2)) + k(k.X(2))]
    
    Since X is a plane wave, k.X(1) = 0 = k.X(2)
    
    Now evaluate  G.s = E(s), G.p = E(p)
                  E(s) = response field due to s-excitation and so on
    
    The above equations can be put in the following matrix form
    
    [E(1)  E(2)] = [E(s) E(p)] [s.X(1)  s.X(2)]
    
    Note that E(1) and E(2) are 3-vectors (column) [Ex,Ey,Ez] in some frame
    Note: 
    's' argument of E in s.E(s) is for the incident direction
    
    Let: 
    J = [[s.X(1) s.X(2)],
         [s.X(1) s.X(2)]]
    and let Jinv = inv(J)
    
    Then:
    
    E(s) = E(1)*Jinv[0,0] + E(2)*Jinv[1,0]
    E(p) = E(1)*Jinv[0,1] + E(2)*Jinv[1,1]
    
    
    """
    J = np.zeros((2,2),dtype=np.complex)
    for i,j in product(range(2),range(2)):
        J[i,j] = np.dot(eI[i],Ein[j])
    if inverse:
        J = np.linalg.inv(J)
    return J

"""
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
sys.path.append(os.path.dirname(__file__)+'/../src')
from vsh import *
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

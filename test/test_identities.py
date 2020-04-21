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
    """
    Numerical values of A1 and A2 VSH, tested against a table made in Mathematica
    The Mathematica results are tested against a table in literature
    """
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
#
def integral_identity(maxL=1,N=100,ksph = np.array([[1., pi/3, 0.]]), verbose=0):
    """
    Fourier transform of VSH functions, over a unit sphere
    """
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
#
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
#
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
#
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
#
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

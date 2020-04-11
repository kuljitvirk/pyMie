import sys
import os
sys.path.append(os.path.dirname(__file__)+'/../src')
import numpy as np
from vsh import *
from mie import *

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

if __name__=='__main__':
    run_test_pw()

"""
Implementation of Vector Spherical Harmonics to facilitate computations involving
VSH as basis functions. 
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
#

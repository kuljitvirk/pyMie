import sys
import os
import time
import numpy as np
from math import pi
sys.path.append(os.path.dirname(__file__)+'/../src/')
from common import gen_surface_spherical
import vsh
import mie
from scipy.integrate import simps
def test(
    radius  = 1,
    n       = 3+0.5j,
    maxL    = 5,
    N       = 100,
    nko     = 4,
    khatINC = [[0.,0.,-1.]],
    ehatINC = [[0.,1.,0.]],
    integrate = simps):
    """
    The integration over radial direction must be performed by a method of higher order than Riemann sums.
    The oscillatory nature of the integrals accumulate to significant relative errors if integration rule
    is not accurate enough. I have found that Simpson's rule gives adequate results with N=100. The maximum 
    absolute error is 0.06% of the exact result at the point where maximum error corrs.
    """
    time_start = time.time()
    k0      = 2*pi
    if nko==1:
        kvecO = np.asarray([[0.,0.,1.]])*k0
    else:
        kvecO, _, _ = gen_surface_spherical(nko,k0)

    # Mie coefficients for the sphere's interior
    # - Incident Plane Wave amplitudes
    a, b, Lvalues = vsh.pw_to_vsh(maxL, khatINC, ehatINC, ncpu=1)
    a, b = a[:,0], b[:,0]
    # - T-matrix
    _Tint = np.array([mie.mie_int(l, n, k0*radius) for l in Lvalues])
    c,d = _Tint[:,0]*a, _Tint[:,1]*b

    shell_radii = np.linspace(0.001,radius, N)
    F = np.zeros((len(shell_radii),3),dtype=np.complex)
    wa = []
    wb = []
    for ir, shell_radius in enumerate(shell_radii):
        # Document: what does transverse=True do and what does that mean?
        _wa, _wb = mie.exact_integral_on_spherical_surface(maxL, n, shell_radius, kvecO, transverse=True, ncpu=1)
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

    # Exact integral from the series 
    Fexact = mie.far_field(maxL, n, radius,kvecO, khatINC,ehatINC)

    # Compute Error
    df = np.abs(F-Fexact)
    i1,i2=np.unravel_index(np.argmax(df),df.shape)
    time_end = time.time()
    print('Max. Abs. Error = {:.3e}'.format(df[i1,i2]))
    print('Function values at the location of max. error')
    print('Exact series    = {}'.format(Fexact[i1,i2]))
    print('Integral        = {}'.format(Fexact[i1,i2]))
    print('Duration: {}'.format(time_end-time_start))
    return kvecO,F,Fexact, (shell_radii,Fr)
if __name__=='__main__':
    test()

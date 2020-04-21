import sys
import os
import numpy as np
from math import pi
sys.path.append(os.path.dirname(__file__)+'/../src/')
from mie import far_field_mie, exact_integral_on_spherical_surface
from vsh import pw_to_vsh

def test():
    radius  = 0.5
    n       = 2+0.1j
    maxL    = 1
    ehatINC = [[0.,1.,0.]]
    kvecO   = [[0.,0.,1.]]

    Fexact,(a,b,e,f) = far_field_mie(maxL, n, radius,kvecO, [[0.,0.,-1.]],ehatINC)
    radii = np.linspace(0.01,radius, 100)
    F = np.zeros((len(radii),3),dtype=np.complex)
    for ir, shell_radius in enumerate(radii):
        Wa, Wb = exact_integral_on_spherical_surface(maxL, n, shell_radius, kvecO, ncpu=1)
        Wa = Wa.squeeze()
        Wb = Wb.squeeze()
        for i in range(len(a)):
            F[ir] += (Wa[i]*a[i] + Wb[i]*b[i])
        #
    #
    dr = radii[1]-radii[0]
    F = np.sum(F, axis=0)*dr
    print('Fexact',Fexact)
    print('F     ',F)
    return
if __name__=='__main__':
    test()

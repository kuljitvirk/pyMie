import numpy as np
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
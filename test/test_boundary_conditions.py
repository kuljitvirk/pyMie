import sys
import os
sys.path.append(os.path.dirname(__file__)+'/../src')
import numpy as np
from mie import compute_fields
import emops_simple as em

def test_boundary_conditions():
    maxL   = 15
    n      = 1.01
    radius = 0.5
    dr = 0.001

    N = 5
    t = np.linspace(0,np.pi,N)
    p = np.linspace(0,2*np.pi,N)
    x = np.sin(t)*np.cos(p)
    y = np.sin(t)*np.sin(p)
    z = np.cos(t)


    points_in  = (radius - dr)*np.vstack([c.flatten() for c in (x,y,z)]).T
    points_out = (radius + dr)*np.vstack([c.flatten() for c in (x,y,z)]).T

    Ein = compute_fields(maxL, points_in, n, radius)
    Eout = compute_fields(maxL, points_out, n, radius)
    Ein = Ein.T
    Eout = Eout.T

    rhat = points_in / np.expand_dims(em.vecmag_abs(points_in),1)
    
    Ein_perp = rhat * np.sum(Ein*rhat,axis=1)[:,None]
    Eout_perp = rhat * np.sum(Eout*rhat,axis=1)[:,None]

    Ein_par = Ein - Ein_perp
    Eout_par = Eout - Eout_perp

    error_perp = Eout_perp - Ein_perp * n**2 
    error_par  = Eout_par - Eout_par

    print('Max Error: Perpendicular' ,np.abs(error_perp).max())
    print('Max Error: Parallel' ,np.abs(error_perp).max())

    x = np.linspace(radius/2 , radius*3,100)
    points = np.vstack((x,x*0,0*x)).T
    E, (Ei, Es, Einc) = compute_fields(maxL, points, radius, n, ehatINC = [[0.,1.0,0.]], return_components=True)
    eps = (x>radius).astype(float) + (x<=radius)*n**2
    D = eps[None,:]*E

    from matplotlib.pyplot import figure, subplots, plot, show
    fig,axs = subplots(figsize=(25,5),ncols=3)
    for i in range(3):
        ax = axs[i]
        ax.plot(x, Einc[i].T.real,':',label='inc')
        ax.plot(x, E[i].T.real,'-',label='Tot')
        ax.plot(x[x<radius], Ei[i].T.real,'--')
        #ax.plot(x[x>radius], Es[i].T.real,'-.')
        ax.legend()
    show()

if __name__=='__main__':
    test_boundary_conditions()
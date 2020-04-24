# pyMie

pyMie computes the near and far zone electromagnetic fields of a system in which a dielectric sphere is excited by a fixed source. Almost all the up to date codes available focus entirely on the scattered field properties fields outside a perfectly conducting sphere. This package provides the ability to compute all the far field properties, as well as the full solutions to Maxwell equations anywhere inside or outside the sphere, to arbitrary accuracy. 

The package relies on high order expansions of spherical harmonics up to order 85 within the scipy.special package. To go beyond this order, I will implement .....

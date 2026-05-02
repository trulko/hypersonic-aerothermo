"""
Run the geometry optimizer to find the waverider trailing edge parameters that maximize L/D, subject 
to the specified constraints. Then plot the resulting geometry and print a report of the results.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from GeometryOptimizer import GeometryOptimizer

wv = GeometryOptimizer(
    M1 = 6,           # Freestream Mach number
    gamma = 1.4,      # Ratio of specific heats
    min_height = 3,   # [m]
    min_area = 100,   # [m^2]
    min_volume = 250, # [m^3]
    T_inf = 216.65,   # K   (~20 km standard atmosphere)
    p_inf = 5474.9,   # Pa  (~20 km standard atmosphere)
    T_allow = 2500.0, # K   (refractory composite limit)
    emissivity = 0.9, # [-] (typical for high-temp composites)
    safety_factor = 1.5, # [-] (safety factor for the bluntness sizing)
    viscous = True,   # Include viscous effects in the optimization?
    resample = 200,   # per-streamline resampling resolution for the boundary layer integration
    n_theta = 20,     # number of polar angle samples for Taylor-Maccoll
    N_opt = 220,      # Leading edge resolution during optimization (lower is faster, but less accurate)
    N_l_opt = 20,     # Surface resolution during optimization (lower is faster,
    N = 500,          # Resolution of the leading edge
    N_l = 30,         # Resulution of the upper, lower surfaces
    maxiter = 40,     # Maximum number of generations for the optimizer
    popsize = 18,     # Population size multiplier for the optimizer
)

# Plotting requires pyvista to be installed
output_dir = "../runs/optimized/"
wv.plot(output_dir)

# Print
wv.report()

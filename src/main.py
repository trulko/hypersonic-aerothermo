"""
Example script: design a waverider and save geometry plots.

Run from src/:
    python main.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from waverider import design_waverider
from mesh_panelization import (
    panelize_geometry,
    panelization_volume,
    panelization_wetted_area,
    plot_panelization,
)

output_dir = "runs/M6_beta16.5"
os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

# Given the input parameters, design the optimal waverider geometry
geom = design_waverider(
    M1=6,
    gamma=1.4,
    beta=16.5,
    L=20, # meters
    N=500,
    N_l=50,
    N_up=25,
    output_dir=output_dir,
)

# Make the computational mesh over the waverider geometry
lower_mesh, upper_mesh = panelize_geometry(geom)

# Compute some mesh statistics
wetted = panelization_wetted_area(lower_mesh, upper_mesh)
volume = panelization_volume(lower_mesh, upper_mesh)
n_tri  = lower_mesh["triangles"].shape[0] + upper_mesh["triangles"].shape[0]

# Report
sc = geom["shock_conditions"]
print(f"\nShock conditions")
print(f"  M2            = {sc['M2']:.4f}")
print(f"  theta         = {sc['theta_deg']:.4f} deg")
print(f"  cone angle    = {sc['cone_half_angle_deg']:.4f} deg")
print(f"\nMesh statistics")
print(f"  Triangles     = {n_tri}  ({lower_mesh['triangles'].shape[0]} lower, {upper_mesh['triangles'].shape[0]} upper)")
print(f"  Wetted area   = {wetted:.3f} m^2")
print(f"  Volume (approx) = {volume:.3f} m^3")

# Make a nice plot
mesh_plot_path = os.path.join(output_dir, "plots", "mesh.png")
plot_panelization(lower_mesh, upper_mesh, save_path=mesh_plot_path, show=True)
print(f"Mesh plot saved to {mesh_plot_path}")

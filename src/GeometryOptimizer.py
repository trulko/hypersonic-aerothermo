from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.dirname(__file__))

from Waverider import Waverider
from TE_Formation import TEG

BETA_MAX = 18.0

# Bounds for beta
def _default_beta_bounds_deg(M1: float) -> tuple[float, float]:
    mach_angle = math.degrees(math.asin(1.0 / M1))
    beta_min = max(mach_angle + 0.5, 1.0)
    beta_max = BETA_MAX
    return beta_min, beta_max

# Bounds for trailing edge parameters
def default_bounds(M1: float) -> list[tuple[float, float]]:
    beta_min, beta_max = _default_beta_bounds_deg(M1)
    return [
        (beta_min, beta_max),
        (TEG.R1_FRAC_MIN, TEG.R1_FRAC_MAX),
        (TEG.W_FRAC_MIN, TEG.W_FRAC_MAX),
        (TEG.N_SHAPE_MIN, TEG.N_SHAPE_MAX),
    ]

# Objective function: Maximize L/D
def _objective(
    params: np.ndarray,
    M1: float,
    gamma: float,
    min_height: float,
    min_area: float,
    min_volume: float,
    T_inf: float,
    p_inf: float,
    T_allow: float,
    emissivity: float,
    safety_factor: float,
    resample: int,
    n_theta: int,
    N: int,
    N_l: int,
    viscous: bool = True
) -> float:
    beta, r1_frac, w2_frac, n_shape = params
    try:
        # Generate admissible geometry
        wv = Waverider(
            M1 = M1,
            gamma = gamma,
            min_height = min_height,
            min_area = min_area,
            min_volume = min_volume,
            beta = float(beta),
            R1_frac = float(r1_frac),
            W2_frac = float(w2_frac),
            n_shape = float(n_shape),
            N = N, N_l = N_l,
        )
        # Compute L/D (inviscid or viscous)
        if viscous:
            wv.aerothermodynamics(
                T_inf = T_inf,
                p_inf = p_inf,
                T_allow = T_allow,
                emissivity = emissivity,
                safety_factor = safety_factor,
                resample = resample,
                n_theta = n_theta,
            )
            ld = wv.LD_total
        else:
            ld = wv.inviscid_aerodynamics()
    except Exception: return float("inf")
    if not np.isfinite(ld): return float("inf")
    return float(-ld)

# Progress bar
class _ProgressBar:
    def __init__(self, total: int, width: int = 30) -> None:
        self.total = max(int(total), 1)
        self.width = max(int(width), 10)

    def update(self, step: int, convergence: float) -> None:
        step = min(max(step, 0), self.total)
        filled = int(self.width * step / self.total)
        bar = "#" * filled + "." * (self.width - filled)
        msg = f"[{bar}] {step}/{self.total}  conv={convergence:.2e}"
        print("\r" + msg, end="", flush=True)

    def close(self) -> None:
        print("")

# Main optimization routine
def _optimize_waverider(
    M1: float,
    gamma: float,
    min_height: float,
    min_area: float,
    min_volume: float,
    T_inf: float,
    p_inf: float,
    T_allow: float,
    emissivity: float,
    safety_factor: float,
    viscous: bool,
    resample: int,
    n_theta: int,
    N_opt: int,
    Nl_opt: int,
    maxiter: int,
    popsize: int ,
    seed: int | None,
) -> dict:
    """
    Return the waverider trailing edge parameters that maximize inviscid L/D.
    """
    bar = _ProgressBar(total=maxiter)
    iter_state = {"count": 0}

    def _callback(xk: np.ndarray, convergence: float) -> bool:
        iter_state["count"] += 1
        if bar is not None:
            bar.update(iter_state["count"], convergence)
        return False
    
    inputs = (M1, gamma, min_height, min_area, min_volume, T_inf, p_inf, T_allow,
              emissivity, safety_factor, resample, n_theta, N_opt, Nl_opt, viscous)

    result = differential_evolution(
        _objective,
        bounds = default_bounds(M1),
        args = inputs,
        maxiter = maxiter,
        popsize = popsize,
        seed = seed,
        workers = -1, # Run in parallel on all cores
        polish = False,
        updating = "deferred",
        callback = _callback,
    )
    if bar is not None: bar.close()
    beta, r1_frac, w2_frac, n_shape = result.x

    return {
        "beta": float(beta),
        "R1_frac": float(r1_frac),
        "W2_frac": float(w2_frac),
        "n_shape": float(n_shape),
        "success": bool(result.success),
        "message": str(result.message),
    }

def GeometryOptimizer(
    M1: float,
    gamma: float,
    min_height: float,
    min_area: float,
    min_volume: float,
    T_inf: float,
    p_inf: float,
    T_allow: float,
    emissivity: float,
    safety_factor: float,
    viscous: bool = True,
    resample: int = 200,
    n_theta: int = 20,
    N_opt: int = 220,
    N_l_opt: int = 20,
    N: int = 500,
    N_l: int = 30,
    maxiter: int = 40,
    popsize: int = 18,
    seed: int | None = 2,
) -> Waverider:
    """
    Generate an optimal waverider geometry for the given flight conditions and constraints

    Parameters:
        M1: Freestream Mach number
        gamma: Ratio of specific heats
        min_height: Minimum height constraint (m)
        min_area: Minimum planform area constraint (m^2)
        min_volume: Minimum volume constraint (m^3)
        T_inf: Freestream temperature for aerothermo (K)
        p_inf: Freestream pressure for aerothermo (Pa)
        T_allow: Maximum allowed temperature for aerothermo (K)
        emissivity: Emissivity for aerothermo [-]
        safety_factor: Safety factor for bluntness sizing in aerothermo [-]
        viscous: Whether to include viscous effects in the L/D evaluation
        resample: Resampling resolution for boundary layer integration in aerothermo
        n_theta: Number of polar angle samples for Taylor-Maccoll in aerothermo
        N_opt: Leading edge resolution during optimization (lower is faster, but less accurate)
        N_l: Surface resolution for geometry generation
        N: Leading edge resolution for geometry generation
        Nl_opt: Surface resolution during optimization (lower is faster, but less accurate)
        maxiter: Maximum number of generations for the optimizer
        popsize: Population size multiplier for the optimizer (higher is more thorough, but slower)
        seed: Random seed for reproducibility
    Returns:
        Waverider object
    """
    # Run optimization to find best geometry parameters
    optimal_parameters = _optimize_waverider(
        M1, gamma, min_height, min_area, min_volume, T_inf, p_inf, T_allow, 
        emissivity, safety_factor, viscous, resample, n_theta, N_opt, N_l_opt, 
        maxiter, popsize, seed
    )
    # Generate the final waverider with the optimal parameters at high resolution
    wv = Waverider(
        M1 = M1,
        gamma = gamma,
        min_height = min_height,
        min_area = min_area,
        min_volume = min_volume,
        beta = optimal_parameters["beta"],
        R1_frac = optimal_parameters["R1_frac"],
        W2_frac = optimal_parameters["W2_frac"],
        n_shape = optimal_parameters["n_shape"],
        N = N,
        N_l = N_l,
    )
    # Run aerothermodynamics to get final L/D and other results
    wv.aerothermodynamics(
        T_inf = T_inf,
        p_inf = p_inf,
        T_allow = T_allow,
        emissivity = emissivity,
        safety_factor = safety_factor,
        resample = resample,
        n_theta = n_theta,
    )
    return wv

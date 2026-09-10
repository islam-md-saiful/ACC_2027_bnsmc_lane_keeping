from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import torch

from bntsmc.common import DEFAULT_DTYPE, DEVICE

"""vehicle parameters"""
@dataclass
class VehicleParams:
    m: float = 1575.0
    Iz: float = 2875.0
    lf: float = 1.2
    lr: float = 1.6
    Cf: float = 80000.0
    Cr: float = 80000.0
    vx: float = 15.0

"""proposed BNTSMC parameters fo lane keeping control"""

@dataclass
class BNTSMCParams:
    ell: float = 1.8
    alpha: float = 0.62
    eta: float = 8.0
    k0: float = 2.6
    ku: float = 42.0
    kD: float = 1.30
    phi: float = 0.060
    eps: float = 1.0e-6
    delta_max: float = 0.50
    param_dt: float = 1.0e-3
    unc_floor: float = 0.004
    unc_scale: float = 1.0

""" control parameters for TSMC """
@dataclass
class FixedTSMCParams:
    ell: float = 1.4
    lam: float = 3.0
    rho: float = 1.65
    gamma: float = 0.70
    alpha: float = 0.70
    eta: float = 5.0
    k: float = 2.05
    kD: float = 0.60
    phi: float = 0.080
    eps: float = 1.0e-6
    delta_max: float = 0.50

"""control parameters for SMC"""
@dataclass
class SMCParams:
    ell: float = 1.4
    c: float = 3.0
    eta: float = 4.5
    k: float = 1.80
    kD: float = 0.50
    phi: float = 0.080
    delta_max: float = 0.50


@dataclass
class Scenario:
    name: str
    label: str
    kappa: Callable[[torch.Tensor], torch.Tensor]
    kappa_dot: Callable[[torch.Tensor], torch.Tensor]
    disturbance: Callable[[torch.Tensor], torch.Tensor]
    plant_params: VehicleParams
    controller_params: VehicleParams
    x0: torch.Tensor


VP_NOMINAL = VehicleParams()
CP_BN = BNTSMCParams()
CP_TSMC = FixedTSMCParams()
CP_SMC = SMCParams()


def kappa_nominal(t: torch.Tensor) -> torch.Tensor:
    return 0.006 * torch.sin(0.35 * t) + 0.003 * torch.sin(0.85 * t)


def kappa_dot_nominal(t: torch.Tensor) -> torch.Tensor:
    return (
        0.006 * 0.35 * torch.cos(0.35 * t)
        + 0.003 * 0.85 * torch.cos(0.85 * t)
    )


def kappa_s_curve(t: torch.Tensor) -> torch.Tensor:
    amplitude = 0.004
    sharpness = 1.20

    return amplitude * (
        torch.tanh(sharpness * (t - 4.0))
        - 2.0 * torch.tanh(sharpness * (t - 10.0))
        + torch.tanh(sharpness * (t - 16.0))
    )


def kappa_dot_s_curve(t: torch.Tensor) -> torch.Tensor:
    amplitude = 0.004
    sharpness = 1.20

    z1 = torch.tanh(sharpness * (t - 4.0))
    z2 = torch.tanh(sharpness * (t - 10.0))
    z3 = torch.tanh(sharpness * (t - 16.0))

    return amplitude * sharpness * (
        (1.0 - z1**2)
        - 2.0 * (1.0 - z2**2)
        + (1.0 - z3**2)
    )


def disturbance_zero(t: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            torch.zeros_like(t),
            torch.zeros_like(t),
            torch.zeros_like(t),
            torch.zeros_like(t),
        ]
    )


def disturbance_small(t: torch.Tensor) -> torch.Tensor:
    d_ey = torch.zeros_like(t)
    d_epsi = torch.zeros_like(t)
    d_vy = 0.025 * torch.sin(1.20 * t)
    d_r = 0.010 * torch.sin(1.70 * t)

    return torch.stack([d_ey, d_epsi, d_vy, d_r])


def disturbance_side_wind(t: torch.Tensor) -> torch.Tensor:
    d_ey = torch.zeros_like(t)
    d_epsi = torch.zeros_like(t)

    side_gust = 0.20 * torch.exp(-((t - 10.0) / 1.20) ** 2)

    d_vy = (
        0.18 * torch.sin(1.40 * t)
        + 0.08 * torch.sin(4.00 * t)
        + side_gust
    )

    d_r = (
        0.08 * torch.sin(2.00 * t)
        + 0.03 * torch.sin(5.00 * t)
        + 0.35 * side_gust
    )

    return torch.stack([d_ey, d_epsi, d_vy, d_r])


def build_scenarios(device: torch.device = DEVICE) -> list[Scenario]:
    plant_tire_uncertain = replace(
        VP_NOMINAL,
        Cf=0.75 * VP_NOMINAL.Cf,
        Cr=0.70 * VP_NOMINAL.Cr,
    )

    unseen_initial_conditions = {
        "IC1_small_offset": [0.30, 0.06, 0.00, 0.00],
        "IC2_large_offset": [0.85, 0.16, 0.00, 0.00],
        "IC3_negative_offset": [-0.60, -0.10, 0.00, 0.00],
        "IC4_with_lateral_velocity": [0.45, 0.08, 0.40, 0.10],
        "IC5_opposite_heading": [0.70, -0.12, -0.30, -0.08],
    }

    base_scenarios = [
        {
            "name": "nominal_curved",
            "label": r"Nominal curved road",
            "kappa": kappa_nominal,
            "kappa_dot": kappa_dot_nominal,
            "disturbance": disturbance_small,
            "plant_params": VP_NOMINAL,
            "controller_params": VP_NOMINAL,
        },
        {
            "name": "s_curve",
            "label": r"S-curve road",
            "kappa": kappa_s_curve,
            "kappa_dot": kappa_dot_s_curve,
            "disturbance": disturbance_small,
            "plant_params": VP_NOMINAL,
            "controller_params": VP_NOMINAL,
        },
        {
            "name": "tire_uncertainty_side_wind",
            "label": r"Tire uncertainty + side wind",
            "kappa": kappa_nominal,
            "kappa_dot": kappa_dot_nominal,
            "disturbance": disturbance_side_wind,
            "plant_params": plant_tire_uncertain,
            "controller_params": VP_NOMINAL,
        },
    ]

    scenarios: list[Scenario] = []

    for base in base_scenarios:
        for ic_name, ic_value in unseen_initial_conditions.items():
            x0 = torch.tensor(ic_value, dtype=DEFAULT_DTYPE, device=device)

            scenarios.append(
                Scenario(
                    name=f"{base['name']}_{ic_name}",
                    label=base["label"] + rf", {ic_name}",
                    kappa=base["kappa"],
                    kappa_dot=base["kappa_dot"],
                    disturbance=base["disturbance"],
                    plant_params=base["plant_params"],
                    controller_params=base["controller_params"],
                    x0=x0,
                )
            )

    return scenarios


def bicycle_dynamics(
    x: torch.Tensor,
    delta: torch.Tensor,
    t: torch.Tensor,
    scenario: Scenario,
) -> torch.Tensor:
    _, epsi, vy, r = x

    vp = scenario.plant_params
    m, Iz, lf, lr, Cf, Cr, vx = (
        vp.m,
        vp.Iz,
        vp.lf,
        vp.lr,
        vp.Cf,
        vp.Cr,
        vp.vx,
    )

    kappa = scenario.kappa(t)

    ey_dot = vy + vx * epsi
    epsi_dot = r - vx * kappa

    vy_dot = (
        -((2.0 * Cf + 2.0 * Cr) / (m * vx)) * vy
        - (vx + (2.0 * lf * Cf - 2.0 * lr * Cr) / (m * vx)) * r
        + (2.0 * Cf / m) * delta
    )

    r_dot = (
        -((2.0 * lf * Cf - 2.0 * lr * Cr) / (Iz * vx)) * vy
        - ((2.0 * lf**2 * Cf + 2.0 * lr**2 * Cr) / (Iz * vx)) * r
        + (2.0 * lf * Cf / Iz) * delta
    )

    dx = torch.stack([ey_dot, epsi_dot, vy_dot, r_dot])
    dx = dx + scenario.disturbance(t).to(x.device)

    return dx


def combined_lane_error(
    x: torch.Tensor,
    t: torch.Tensor,
    scenario: Scenario,
    ell: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ey, epsi, vy, r = x

    vx = scenario.controller_params.vx
    kappa = scenario.kappa(t)

    ec = ey + ell * epsi
    ec_dot = vy + vx * epsi + ell * (r - vx * kappa)

    return ec, ec_dot


def nominal_FL_GL(
    x: torch.Tensor,
    t: torch.Tensor,
    scenario: Scenario,
    ell: float,
) -> tuple[torch.Tensor, float]:
    _, _, vy, r = x

    vp = scenario.controller_params
    m, Iz, lf, lr, Cf, Cr, vx = (
        vp.m,
        vp.Iz,
        vp.lf,
        vp.lr,
        vp.Cf,
        vp.Cr,
        vp.vx,
    )

    kappa = scenario.kappa(t)
    kappa_dot = scenario.kappa_dot(t)

    a33 = -((2.0 * Cf + 2.0 * Cr) / (m * vx))
    a34 = -(vx + (2.0 * lf * Cf - 2.0 * lr * Cr) / (m * vx))
    a43 = -((2.0 * lf * Cf - 2.0 * lr * Cr) / (Iz * vx))
    a44 = -((2.0 * lf**2 * Cf + 2.0 * lr**2 * Cr) / (Iz * vx))

    b3 = 2.0 * Cf / m
    b4 = 2.0 * lf * Cf / Iz

    FL = (
        (a33 + ell * a43) * vy
        + (a34 + vx + ell * a44) * r
        - vx**2 * kappa
        - ell * vx * kappa_dot
    )

    GL = b3 + ell * b4

    return FL, GL
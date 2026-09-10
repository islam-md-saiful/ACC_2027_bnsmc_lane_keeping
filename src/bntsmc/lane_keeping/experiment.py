from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from bntsmc.common import DEFAULT_DTYPE, DEVICE, configure_matplotlib, make_timestamped_output_dir, save_pdf, set_seed
from bntsmc.lane_keeping.controller import compute_control
from bntsmc.lane_keeping.onnx_inference import ONNXBNNPredictor
from bntsmc.lane_keeping.system import Scenario, bicycle_dynamics, build_scenarios



# --------------------Paths------------------------------


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ONNX_PATH = PROJECT_ROOT / "models" / "bntsmc_lane.onnx"


# -----------------------Controller plot styles--------------------------------


CONTROLLER_STYLES = {
    "BNTSMC": {"label": "BNTSMC", "color": "red"},
    "TSMC": {"label": "TSMC", "color": "blue"},
    "SMC": {"label": "SMC", "color": "black"},
}

#-------------------------------- RK4 integration-------------------------------------


def rk4_step(
    predictor,
    x: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    scenario: Scenario,
    controller_name: str,
) -> tuple[torch.Tensor, torch.Tensor, dict]:

    delta1, info1 = compute_control(
        predictor,
        x,
        t,
        scenario,
        controller_name,
    )

    k1 = bicycle_dynamics(
        x,
        delta1,
        t,
        scenario,
    )

    delta2, _ = compute_control(
        predictor,
        x + 0.5 * dt * k1,
        t + 0.5 * dt,
        scenario,
        controller_name,
    )

    k2 = bicycle_dynamics(
        x + 0.5 * dt * k1,
        delta2,
        t + 0.5 * dt,
        scenario,
    )

    delta3, _ = compute_control(
        predictor,
        x + 0.5 * dt * k2,
        t + 0.5 * dt,
        scenario,
        controller_name,
    )

    k3 = bicycle_dynamics(
        x + 0.5 * dt * k2,
        delta3,
        t + 0.5 * dt,
        scenario,
    )

    delta4, _ = compute_control(
        predictor,
        x + dt * k3,
        t + dt,
        scenario,
        controller_name,
    )

    k4 = bicycle_dynamics(
        x + dt * k3,
        delta4,
        t + dt,
        scenario,
    )

    x_next = x + (dt / 6.0) * (
        k1
        + 2.0 * k2
        + 2.0 * k3
        + k4
    )

    return x_next, delta1, info1



# ---------------------------Simulate controller-------------------------------------


def simulate_controller(
    predictor,
    scenario: Scenario,
    controller_name: str,
    save_dir: str | Path,
    timestamp: str,
    T: float = 22.0,
    dt: float = 0.01,
) -> dict:

    n_steps = int(T / dt) + 1

    t_hist = torch.zeros(n_steps, dtype=DEFAULT_DTYPE, device=DEVICE)
    x_hist = torch.zeros(n_steps, 4, dtype=DEFAULT_DTYPE, device=DEVICE)
    delta_hist = torch.zeros(n_steps, dtype=DEFAULT_DTYPE, device=DEVICE)

    sB_hist = torch.zeros(n_steps, dtype=DEFAULT_DTYPE, device=DEVICE)
    V_hist = torch.zeros(n_steps, dtype=DEFAULT_DTYPE, device=DEVICE)

    mu_lambda_hist = torch.zeros(n_steps, dtype=DEFAULT_DTYPE, device=DEVICE)
    mu_rho_hist = torch.zeros(n_steps, dtype=DEFAULT_DTYPE, device=DEVICE)
    mu_gamma_hist = torch.zeros(n_steps, dtype=DEFAULT_DTYPE, device=DEVICE)

    sigma_s_hist = torch.zeros(n_steps, dtype=DEFAULT_DTYPE, device=DEVICE)
    kB_hist = torch.zeros(n_steps, dtype=DEFAULT_DTYPE, device=DEVICE)
    kappa_hist = torch.zeros(n_steps, dtype=DEFAULT_DTYPE, device=DEVICE)

    x = scenario.x0.clone().to(DEVICE)

    for i in range(n_steps):

        t = torch.tensor(
            i * dt,
            dtype=DEFAULT_DTYPE,
            device=DEVICE,
        )

        delta, info = compute_control(
            predictor,
            x,
            t,
            scenario,
            controller_name,
        )

        t_hist[i] = t
        x_hist[i, :] = x
        delta_hist[i] = delta

        sB_hist[i] = info["sB"]
        V_hist[i] = info["V"]

        mu_lambda_hist[i] = info["mu_lambda"]
        mu_rho_hist[i] = info["mu_rho"]
        mu_gamma_hist[i] = info["mu_gamma"]

        sigma_s_hist[i] = info["sigma_s"]
        kB_hist[i] = info["kB"]
        kappa_hist[i] = info["kappa"]

        if i < n_steps - 1:

            x, _, _ = rk4_step(
                predictor,
                x,
                t,
                dt,
                scenario,
                controller_name,
            )

    data = {
        "scenario_name": scenario.name,
        "scenario_label": scenario.label,
        "controller_name": controller_name,
        "controller_label": CONTROLLER_STYLES[controller_name]["label"],

        "t": t_hist.detach().cpu().numpy(),
        "x": x_hist.detach().cpu().numpy(),

        "ey": x_hist[:, 0].detach().cpu().numpy(),
        "epsi": x_hist[:, 1].detach().cpu().numpy(),
        "vy": x_hist[:, 2].detach().cpu().numpy(),
        "r": x_hist[:, 3].detach().cpu().numpy(),

        "delta": delta_hist.detach().cpu().numpy(),

        "sB": sB_hist.detach().cpu().numpy(),
        "V": V_hist.detach().cpu().numpy(),

        "mu_lambda": mu_lambda_hist.detach().cpu().numpy(),
        "mu_rho": mu_rho_hist.detach().cpu().numpy(),
        "mu_gamma": mu_gamma_hist.detach().cpu().numpy(),

        "sigma_s": sigma_s_hist.detach().cpu().numpy(),
        "kB": kB_hist.detach().cpu().numpy(),

        "kappa": kappa_hist.detach().cpu().numpy(),
    }

    save_dir = Path(save_dir)

    np.savez(
        save_dir / f"data_{scenario.name}_{controller_name}_{timestamp}.npz",
        **{
            key: value
            for key, value in data.items()
            if isinstance(value, np.ndarray)
        },
    )

    return data



# ----------------------------Settling time--------------------------------


def settling_time(
    t: np.ndarray,
    y: np.ndarray,
    band: float,
) -> float:

    abs_y = np.abs(y)

    for i in range(len(t)):

        if np.all(
            abs_y[i:] <= band
        ):
            return float(t[i])

    return float("nan")


# -------------------------------------------------------------
# -------------------------Metrics-------------------------------
# -------------------------------------------------------------

def compute_metrics(
    all_results: Dict[str, Dict],
    save_dir: str | Path,
    timestamp: str,
) -> Path:

    rows = []

    def integrate(y, t):

        if hasattr(np, "trapezoid"):
            return np.trapezoid(y, t)

        return np.trapz(y, t)

    for scenario_name, scenario_bundle in all_results.items():

        for controller_name in [
            "BNTSMC",
            "TSMC",
            "SMC",
        ]:

            data = scenario_bundle[
                "results"
            ][controller_name]

            t = data["t"]
            ey = data["ey"]
            epsi = data["epsi"]
            delta = data["delta"]

            # RMSE
            rmse_ey = np.sqrt(
                np.mean(ey**2)
            )

            rmse_epsi = np.sqrt(
                np.mean(epsi**2)
            )

            # IAE
            iae_ey = integrate(
                np.abs(ey),
                t,
            )

            iae_epsi = integrate(
                np.abs(epsi),
                t,
            )

            # ITAE
            itae_ey = integrate(
                t * np.abs(ey),
                t,
            )

            itae_epsi = integrate(
                t * np.abs(epsi),
                t,
            )

            # Peak errors
            peak_abs_ey = np.max(
                np.abs(ey)
            )

            peak_abs_epsi = np.max(
                np.abs(epsi)
            )

            max_ey = np.max(ey)
            min_ey = np.min(ey)

            # Settling times
            Ts_ey = settling_time(
                t,
                ey,
                band=0.02,
            )

            Ts_epsi = settling_time(
                t,
                epsi,
                band=0.005,
            )

            # Steering control energy
            control_energy = integrate(
                delta**2,
                t,
            )

            rows.append(
                {
                    "Scenario": scenario_name,
                    "Controller": controller_name,

                    "RMSE_ey": rmse_ey,
                    "RMSE_epsi": rmse_epsi,

                    "IAE_ey": iae_ey,
                    "IAE_epsi": iae_epsi,

                    "ITAE_ey": itae_ey,
                    "ITAE_epsi": itae_epsi,

                    "PeakAbs_ey": peak_abs_ey,
                    "PeakAbs_epsi": peak_abs_epsi,

                    "Max_ey": max_ey,
                    "Min_ey": min_ey,

                    "Ts_ey": Ts_ey,
                    "Ts_epsi": Ts_epsi,

                    "ControlEnergy": control_energy,
                }
            )

    metrics_df = pd.DataFrame(rows)
    save_dir = Path(save_dir)

    # --------------------------------------------------------
    # Save all metrics
    # --------------------------------------------------------

    metrics_path = (save_dir/ f"metrics_all_{timestamp}.csv")
    metrics_df.to_csv(metrics_path,index=False,)

    # --------------------------------------------------------
    # Figure 6 metrics (Nominal road condition)
    # --------------------------------------------------------

    fig6_name = ("nominal_curved_IC1_small_offset")
    fig6_df = metrics_df[metrics_df["Scenario"] == fig6_name].copy()

    controller_order = ["BNTSMC","TSMC","SMC",]

    fig6_df["Controller"] = pd.Categorical(fig6_df["Controller"],categories=controller_order,ordered=True,)
    fig6_df = fig6_df.sort_values("Controller")
    fig6_table = fig6_df[
        [
            "Controller",

            "RMSE_ey",
            "RMSE_epsi",

            "IAE_ey",
            "IAE_epsi",

            "ITAE_ey",
            "ITAE_epsi",

            "PeakAbs_ey",
            "PeakAbs_epsi",

            "Ts_ey",
            "Ts_epsi",

            "ControlEnergy",
        ]
    ].copy()

    numeric_columns = (fig6_table.columns.drop("Controller") )
    fig6_table[numeric_columns] = (fig6_table[numeric_columns].astype(float).round(6))

    print("\n")
    print("=" * 120)
    print("FIGURE 6: CONTROLLER PERFORMANCE METRICS")
    print("Scenario 1 - Nominal Curved-Road Tracking, IC1")
    print("=" * 120)
    print(fig6_table.to_string(index=False))

    print("=" * 120)

    # --------------------------------------------------------
    # Save Figure 6 CSV
    # --------------------------------------------------------

    fig6_csv_path = (save_dir/ f"Fig6_controller_metrics_{timestamp}.csv")

    fig6_table.to_csv(fig6_csv_path,index=False,)

    # --------------------------------------------------------
    # Save Figure 6 LaTeX
    # --------------------------------------------------------

    fig6_latex_path = (save_dir / f"Fig6_controller_metrics_{timestamp}.tex")

    latex_table = fig6_table.to_latex(
        index=False,
        float_format="%.4f",
        escape=False,
        caption=(
            "Performance comparison of BNTSMC, "
            "TSMC, and SMC for the nominal "
            "curved-road scenario."
        ),
        label="tab:lane_fig6_metrics",
    )

    with open(
        fig6_latex_path,
        "w",
    ) as f:

        f.write(
            latex_table
        )

    print("\nSaved all metrics to:")
    print(metrics_path)

    print("\nSaved Figure 6 controller table to:")
    print(fig6_csv_path)

    print("\nSaved Figure 6 LaTeX table to:")
    print(fig6_latex_path)

    return metrics_path


# ============================================================
# Plot style
# ============================================================

def apply_latex_axis_style(
    ax,
    xlabel: str = r"Time (s)",
    ylabel: str | None = None,
) -> None:
    
    ax.tick_params(axis="both",direction="out",colors="black",labelsize=14,)
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["top"].set_color("black")
    ax.spines["right"].set_color("black")
    ax.set_xlabel(xlabel,fontsize=14,)

    if ylabel is not None:

        ax.set_ylabel(ylabel,fontsize=14,)
    ax.grid(True,linestyle="--",linewidth=0.6,alpha=0.55,)


# ============================================================
# Road curvature plot
# ============================================================

def plot_road_curvature(
    bundle: Dict,
    save_dir: str | Path,
    timestamp: str,
) -> None:

    scenario_name = bundle["scenario_name"]
    scenario_label = bundle["scenario_label"]
    ref_data = bundle["results"]["BNTSMC"]
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot(
        ref_data["t"],
        ref_data["kappa"],
        linewidth=1,
        color="black",
 #       label=scenario_label,
     )
    ax.axhline(0.0,linewidth=0.5,linestyle="--",color="gray",)
    apply_latex_axis_style(ax,xlabel=r"Time (s)",ylabel=r"$\kappa(t)$ (1/m)",)

#    ax.legend(
#        frameon=True,
#        fontsize=14,
#    )
    save_pdf(fig,save_dir=save_dir,filename=f"{scenario_name}_Fig_01_road_curvature_kappa",timestamp=timestamp,)
    plt.close(fig)


# ============================================================
# Controller comparison plot
# ============================================================

def plot_controller_comparison(
    bundle: Dict,
    y_key: str,
    ylabel: str,
    filename: str,
    save_dir: str | Path,
    timestamp: str,
    horizontal_zero: bool = False,
) -> None:

    scenario_name = bundle[
        "scenario_name"
    ]

    fig, ax = plt.subplots(
        figsize=(5, 2.5)
    )

    for controller_name in [
        "BNTSMC",
        "TSMC",
        "SMC",
    ]:

        data = bundle[
            "results"
        ][
            controller_name
        ]

        style = CONTROLLER_STYLES[
            controller_name
        ]

        ax.plot(
            data["t"],
            data[y_key],
            linewidth=1,
            color=style["color"],
            label=style["label"],
        )

    if horizontal_zero:

        ax.axhline(
            0.0,
            linewidth=0.5,
            linestyle="--",
            color="gray",
        )

    apply_latex_axis_style(
        ax,
        xlabel=r"Time (s)",
        ylabel=ylabel,
    )

    ax.legend(
        frameon=True,
        fontsize=14,
    )

    save_pdf(
        fig,
        save_dir=save_dir,
        filename=f"{scenario_name}_{filename}",
        timestamp=timestamp,
    )

    plt.close(fig)


# ============================================================
# BNTSMC posterior-mean plots
# ============================================================

def plot_bntsmc_posterior_mean(
    bundle: Dict,
    y_key: str,
    ylabel: str,
    filename: str,
    save_dir: str | Path,
    timestamp: str,
) -> None:

    scenario_name = bundle[
        "scenario_name"
    ]

    data = bundle[
        "results"
    ][
        "BNTSMC"
    ]

    fig, ax = plt.subplots(
        figsize=(5, 2.5)
    )

    ax.plot(
        data["t"],
        data[y_key],
        linewidth=1.2,
        color=CONTROLLER_STYLES["BNTSMC"]["color"],
        label="BNTSMC",
    )

    apply_latex_axis_style(
        ax,
        xlabel=r"Time (s)",
        ylabel=ylabel,
    )

    ax.legend(
        frameon=True,
        fontsize=14,
    )

    save_pdf(
        fig,
        save_dir=save_dir,
        filename=f"{scenario_name}_{filename}",
        timestamp=timestamp,
    )

    plt.close(fig)


# ============================================================
# Generate all plots
# ============================================================

def make_all_required_plots(
    all_results: Dict[str, Dict],
    save_dir: str | Path,
    timestamp: str,
) -> None:

    for _, bundle in all_results.items():

        plot_road_curvature(
            bundle,
            save_dir=save_dir,
            timestamp=timestamp,
        )

        # ----------------------------------------------------
        # Tracking responses
        # ----------------------------------------------------

        plot_controller_comparison(
            bundle,
            y_key="ey",
            ylabel=r"$e_y(t)$ (m)",
            filename="Fig_02_lateral_error_ey",
            save_dir=save_dir,
            timestamp=timestamp,
            horizontal_zero=True,
        )

        plot_controller_comparison(
            bundle,
            y_key="epsi",
            ylabel=r"$e_\psi(t)$ (rad)",
            filename="Fig_03_heading_error_epsi",
            save_dir=save_dir,
            timestamp=timestamp,
            horizontal_zero=True,
        )

        plot_controller_comparison(
            bundle,
            y_key="delta",
            ylabel=r"$\delta(t)$ (rad)",
            filename="Fig_04_steering_input_delta",
            save_dir=save_dir,
            timestamp=timestamp,
            horizontal_zero=True,
        )

        # ----------------------------------------------------
        # Sliding mode variables
        # ----------------------------------------------------

        plot_controller_comparison(
            bundle,
            y_key="sB",
            ylabel=r"$s(t)$",
            filename="Fig_05_sliding_surface_s",
            save_dir=save_dir,
            timestamp=timestamp,
            horizontal_zero=True,
        )

        plot_controller_comparison(
            bundle,
            y_key="V",
            ylabel=r"$V(t)$",
            filename="Fig_06_lyapunov_function_V",
            save_dir=save_dir,
            timestamp=timestamp,
        )

        # ----------------------------------------------------
        # Post-training BNN inference
        # ----------------------------------------------------

        plot_bntsmc_posterior_mean(
            bundle,
            y_key="mu_lambda",
            ylabel=r"$\mu_{\lambda}(t)$",
            filename="Fig_07_posterior_mean_mu_lambda",
            save_dir=save_dir,
            timestamp=timestamp,
        )

        plot_bntsmc_posterior_mean(
            bundle,
            y_key="mu_rho",
            ylabel=r"$\mu_{\rho}(t)$",
            filename="Fig_08_posterior_mean_mu_rho",
            save_dir=save_dir,
            timestamp=timestamp,
        )

        plot_bntsmc_posterior_mean(
            bundle,
            y_key="mu_gamma",
            ylabel=r"$\mu_{\gamma}(t)$",
            filename="Fig_09_posterior_mean_mu_gamma",
            save_dir=save_dir,
            timestamp=timestamp,
        )

        # ----------------------------------------------------
        # Uncertainty and adaptive gain
        # ----------------------------------------------------

        plot_controller_comparison(
            bundle,
            y_key="sigma_s",
            ylabel=r"$\sigma_s(t)$",
            filename="Fig_10_sliding_uncertainty_sigma_s",
            save_dir=save_dir,
            timestamp=timestamp,
        )

        plot_controller_comparison(
            bundle,
            y_key="kB",
            ylabel=r"$k_B(t)$",
            filename="Fig_11_reaching_gain_k",
            save_dir=save_dir,
            timestamp=timestamp,
        )


# ============================================================
# Main lane-keeping experiment
# ============================================================

def run_lane_keeping_experiment(
    T: float = 22.0,
    dt: float = 0.01,
    output_root: str | Path | None = None,
    onnx_path: str | Path = DEFAULT_ONNX_PATH,
) -> None:

    set_seed()
    configure_matplotlib()

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    save_dir, timestamp = make_timestamped_output_dir(
        example_name="lane_keeping",
        output_root=output_root,
    )

    print(f"Using device: {DEVICE}")
    print(f"Saving lane-keeping results in: {save_dir}")

    # ========================================================
    # Load already-trained ONNX BNN
    # NO TRAINING IS PERFORMED HERE
    # ========================================================

    onnx_path = Path(
        onnx_path
    )

    if not onnx_path.exists():

        raise FileNotFoundError(
            f"\nTrained ONNX model was not found:\n"
            f"{onnx_path}\n\n"
            "Run scripts/train_lane_keeping.py first."
        )

    print("\nLoading trained ONNX BNN:")
    print(onnx_path)

    predictor = ONNXBNNPredictor(
        onnx_path
    )

    print("ONNX BNN loaded successfully.")
    print("Running post-training inference only.\n")

    # --------------------------------------------------------
    # Test scenarios
    # --------------------------------------------------------

    scenarios = build_scenarios(
        device=DEVICE
    )

    all_results = {}

    # --------------------------------------------------------
    # Simulations
    # --------------------------------------------------------

    for scenario in scenarios:

        print(
            f"Simulating scenario: "
            f"{scenario.name}"
        )

        scenario_bundle = {
            "scenario_name": scenario.name,
            "scenario_label": scenario.label,
            "results": {},
        }

        for controller_name in [
            "BNTSMC",
            "TSMC",
            "SMC",
        ]:

            print(
                f"  Controller: "
                f"{controller_name}"
            )

            data = simulate_controller(
                predictor=predictor,
                scenario=scenario,
                controller_name=controller_name,
                save_dir=save_dir,
                timestamp=timestamp,
                T=T,
                dt=dt,
            )

            scenario_bundle[
                "results"
            ][controller_name] = data

        all_results[
            scenario.name
        ] = scenario_bundle

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    print("\nComputing metrics...")

    metrics_path = compute_metrics(
        all_results=all_results,
        save_dir=save_dir,
        timestamp=timestamp,
    )

    print(
        f"Saved metrics: "
        f"{metrics_path}"
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    print("\nGenerating PDF figures...")

    make_all_required_plots(
        all_results=all_results,
        save_dir=save_dir,
        timestamp=timestamp,
    )

    print("\nLane-keeping experiment completed.")
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from bntsmc.common import DEFAULT_DTYPE, DEVICE, sat
from bntsmc.lane_keeping.system import (
    BNTSMCParams,
    CP_BN,
    CP_SMC,
    CP_TSMC,
    FixedTSMCParams,
    SMCParams,
    Scenario,
    combined_lane_error,
    nominal_FL_GL,
)


class BayesLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features, dtype=DEFAULT_DTYPE).normal_(0.0, 0.08))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features, dtype=DEFAULT_DTYPE).fill_(-5.0))

        self.bias_mu = nn.Parameter(torch.empty(out_features, dtype=DEFAULT_DTYPE).normal_(0.0, 0.08))
        self.bias_rho = nn.Parameter(torch.empty(out_features, dtype=DEFAULT_DTYPE).fill_(-5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        eps_w = torch.randn_like(weight_sigma)
        eps_b = torch.randn_like(bias_sigma)

        weight = self.weight_mu + weight_sigma * eps_w
        bias = self.bias_mu + bias_sigma * eps_b

        return F.linear(x, weight, bias)

    def forward_mean(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_mu, self.bias_mu)

    def kl_divergence(self) -> torch.Tensor:
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        prior_sigma_tensor = torch.tensor(
            self.prior_sigma,
            dtype=DEFAULT_DTYPE,
            device=self.weight_mu.device,
        )

        prior_var = prior_sigma_tensor**2

        kl_w = torch.sum(
            torch.log(prior_sigma_tensor / weight_sigma)
            + (weight_sigma**2 + self.weight_mu**2) / (2.0 * prior_var)
            - 0.5
        )

        kl_b = torch.sum(
            torch.log(prior_sigma_tensor / bias_sigma)
            + (bias_sigma**2 + self.bias_mu**2) / (2.0 * prior_var)
            - 0.5
        )

        return kl_w + kl_b


class BayesianSlidingNet(nn.Module):

    def __init__(self, input_dim: int = 7, hidden_dim: int = 64):
        super().__init__()

        self.b1 = BayesLinear(input_dim, hidden_dim)
        self.b2 = BayesLinear(hidden_dim, hidden_dim)
        self.b3 = BayesLinear(hidden_dim, 3)

    def _map_output(self, raw: torch.Tensor) -> torch.Tensor:
        lam = 1.5 + 7.5 * torch.sigmoid(raw[..., 0])
        rho = 0.5 + 5.0 * torch.sigmoid(raw[..., 1])
        gamma = 0.35 + 0.50 * torch.sigmoid(raw[..., 2])

        return torch.stack([lam, rho, gamma], dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.b1(z))
        h = torch.tanh(self.b2(h))
        raw = self.b3(h)

        return self._map_output(raw)

    def forward_mean(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.b1.forward_mean(z))
        h = torch.tanh(self.b2.forward_mean(h))
        raw = self.b3.forward_mean(h)

        return self._map_output(raw)

    def kl_divergence(self) -> torch.Tensor:
        return (
            self.b1.kl_divergence()
            + self.b2.kl_divergence()
            + self.b3.kl_divergence()
        )


def normalize_features(raw_z: torch.Tensor) -> torch.Tensor:
    scale = torch.tensor(
        [1.0, 0.25, 3.0, 1.0, 0.015, 30.0, 25.0],
        dtype=DEFAULT_DTYPE,
        device=raw_z.device,
    )

    return raw_z / scale


def make_bnn_input(
    x: torch.Tensor,
    t: torch.Tensor,
    scenario: Scenario,
) -> torch.Tensor:
    kappa = scenario.kappa(t)

    raw_z = torch.stack(
        [
            x[0],
            x[1],
            x[2],
            x[3],
            kappa,
            torch.tensor(
                scenario.controller_params.vx,
                dtype=DEFAULT_DTYPE,
                device=x.device,
            ),
            t,
        ]
    )

    return normalize_features(raw_z)

def make_raw_bnn_input(
    x: torch.Tensor,
    t: torch.Tensor,
    scenario: Scenario,
) -> torch.Tensor:
    
    """Raw input for ONNX inference: [ey, epsi, vy, r, kappa, vx, t]. 
    Do not normalize here because the ONNX model performs normalization internally."""

    kappa = scenario.kappa(t)

    return torch.stack(
        [
            x[0],
            x[1],
            x[2],
            x[3],
            kappa,
            torch.tensor(
                scenario.controller_params.vx,
                dtype=DEFAULT_DTYPE,
                device=x.device,
            ),
            t,
        ]
    )


def predict_bnn_from_onnx(
    predictor,
    x: torch.Tensor,
    t: torch.Tensor,
    scenario: Scenario,
) -> tuple[torch.Tensor, torch.Tensor]:

    raw_z = make_raw_bnn_input(
        x=x,
        t=t,
        scenario=scenario,
    )

    theta_mean_np, theta_std_np = predictor(
        raw_z.detach().cpu().numpy()
    )

    theta_mean = torch.as_tensor(
        theta_mean_np,
        dtype=DEFAULT_DTYPE,
        device=x.device,
    )

    theta_std = torch.as_tensor(
        theta_std_np,
        dtype=DEFAULT_DTYPE,
        device=x.device,
    )

    return theta_mean, theta_std



def target_sliding_params(raw_z: torch.Tensor) -> torch.Tensor:
    ey = raw_z[:, 0]
    epsi = raw_z[:, 1]
    vy = raw_z[:, 2]
    r = raw_z[:, 3]
    kappa = raw_z[:, 4]
    vx = raw_z[:, 5]

    ell = CP_BN.ell

    ec = ey + ell * epsi
    ec_dot = vy + vx * epsi + ell * (r - vx * kappa)

    q_ec = torch.tanh(2.80 * torch.abs(ec))
    q_ecd = torch.tanh(1.00 * torch.abs(ec_dot))
    q_kappa = torch.tanh(90.0 * torch.abs(kappa))

    lambda_target = 3.0 + 3.8 * q_ec + 1.8 * q_ecd + 0.7 * q_kappa
    rho_target = 1.1 + 3.0 * q_ec + 1.2 * q_kappa
    gamma_target = 0.76 - 0.28 * q_ec + 0.04 * q_kappa

    lambda_target = torch.clamp(lambda_target, 1.5, 9.0)
    rho_target = torch.clamp(rho_target, 0.5, 5.5)
    gamma_target = torch.clamp(gamma_target, 0.35, 0.85)

    return torch.stack([lambda_target, rho_target, gamma_target], dim=-1)


def generate_training_data(
    n_samples: int = 9000,
    device: torch.device = DEVICE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ey = torch.empty(n_samples, dtype=DEFAULT_DTYPE).uniform_(-1.00, 1.00)
    epsi = torch.empty(n_samples, dtype=DEFAULT_DTYPE).uniform_(-0.20, 0.20)
    vy = torch.empty(n_samples, dtype=DEFAULT_DTYPE).uniform_(-2.00, 2.00)
    r = torch.empty(n_samples, dtype=DEFAULT_DTYPE).uniform_(-0.65, 0.65)
    kappa = torch.empty(n_samples, dtype=DEFAULT_DTYPE).uniform_(-0.016, 0.016)
    vx = torch.empty(n_samples, dtype=DEFAULT_DTYPE).uniform_(12.0, 22.0)
    t = torch.empty(n_samples, dtype=DEFAULT_DTYPE).uniform_(0.0, 22.0)

    raw_z = torch.stack([ey, epsi, vy, r, kappa, vx, t], dim=1)
    z = normalize_features(raw_z)
    theta_target = target_sliding_params(raw_z)

    return raw_z.to(device), z.to(device), theta_target.to(device)


def train_bnn(
    model: BayesianSlidingNet,
    save_dir: str | Path,
    timestamp: str,
    epochs: int = 7000,
    batch_size: int = 256,
    lr: float = 2.0e-3,
) -> Path:
    raw_z, z, theta_target = generate_training_data()

    dataset_size = z.shape[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    lambda_slide = 3.0e-4
    lambda_unc = 1.0e-3
    beta_kl = 1.5e-6
    mc_train = 3

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(dataset_size, device=DEVICE)
        epoch_loss = 0.0

        for i in range(0, dataset_size, batch_size):
            idx = permutation[i : i + batch_size]

            zb = z[idx]
            rawb = raw_z[idx]
            targetb = theta_target[idx]

            pred_mc = []
            for _ in range(mc_train):
                pred_mc.append(model(zb))
            pred_mc = torch.stack(pred_mc, dim=0)

            pred_mean = pred_mc.mean(dim=0)
            pred_std = pred_mc.std(dim=0) + 1.0e-8

            lam = pred_mean[:, 0]
            rho = pred_mean[:, 1]
            gamma = pred_mean[:, 2]

            sig_lam = pred_std[:, 0]
            sig_rho = pred_std[:, 1]
            sig_gamma = pred_std[:, 2]

            ey = rawb[:, 0]
            epsi = rawb[:, 1]
            vy = rawb[:, 2]
            r = rawb[:, 3]
            kappa = rawb[:, 4]
            vx = rawb[:, 5]

            ec = ey + CP_BN.ell * epsi
            ec_dot = vy + vx * epsi + CP_BN.ell * (r - vx * kappa)

            abs_ec = torch.abs(ec) + CP_BN.eps

            sB = ec_dot + lam * ec + rho * torch.pow(abs_ec, gamma) * torch.sign(ec)

            sigma_s_train = (
                torch.abs(ec) * sig_lam
                + torch.pow(abs_ec, gamma) * sig_rho
                + torch.abs(
                    rho * torch.pow(abs_ec, gamma) * torch.log(abs_ec)
                )
                * sig_gamma
            )

            loss_sup = torch.mean((pred_mean - targetb) ** 2)
            loss_slide = torch.mean(sB**2)
            loss_unc = torch.mean((sigma_s_train / (1.0 + torch.abs(sB))) ** 2)
            loss_kl = model.kl_divergence() / dataset_size

            loss = (
                loss_sup
                + lambda_slide * loss_slide
                + lambda_unc * loss_unc
                + beta_kl * loss_kl
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d} | Loss = {epoch_loss:.6f}")

    save_dir = Path(save_dir)
    model_path = save_dir / f"bntsmc_bnn_lane_{timestamp}.pt"
    torch.save(model.state_dict(), model_path)

    return model_path


@torch.no_grad()
def bnn_predict_mean_uncertainty(
    model: BayesianSlidingNet,
    z: torch.Tensor,
    mc_samples: int = 35,
) -> tuple[torch.Tensor, torch.Tensor]:
    z_batch = z.unsqueeze(0)

    theta_mean = model.forward_mean(z_batch).squeeze(0)

    devices = [DEVICE] if DEVICE.type == "cuda" else []

    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(12345)

        if DEVICE.type == "cuda":
            torch.cuda.manual_seed_all(12345)

        preds = []
        for _ in range(mc_samples):
            pred = model(z_batch).squeeze(0)
            preds.append(pred)

    preds = torch.stack(preds, dim=0)
    theta_std = preds.std(dim=0) + 1.0e-8

    return theta_mean, theta_std


@torch.no_grad()
def bnn_predict_mean_only(
    model: BayesianSlidingNet,
    z: torch.Tensor,
) -> torch.Tensor:
    return model.forward_mean(z.unsqueeze(0)).squeeze(0)


@torch.no_grad()

def bntsmc_control(
    predictor,
    x: torch.Tensor,
    t: torch.Tensor,
    scenario: Scenario,
    cp: BNTSMCParams,
) -> tuple[torch.Tensor, dict]:
   # z = make_bnn_input(x, t, scenario).to(DEVICE)

   # theta_mean, theta_std = bnn_predict_mean_uncertainty(model, z, mc_samples=35)


    theta_mean, theta_std = predict_bnn_from_onnx(predictor=predictor,x=x,t=t,scenario=scenario,)


    mu_lam = theta_mean[0]
    mu_rho = theta_mean[1]
    mu_gamma = theta_mean[2]

    sig_lam = theta_std[0]
    sig_rho = theta_std[1]
    sig_gamma = theta_std[2]

    t_next = t + torch.tensor(cp.param_dt, dtype=DEFAULT_DTYPE, device=x.device,)
    theta_next, _ = predict_bnn_from_onnx(predictor=predictor, x=x, t=t_next, scenario=scenario,)
   # theta_next = bnn_predict_mean_only(model, z_next)

    mu_lam_dot = (theta_next[0] - mu_lam) / cp.param_dt
    mu_rho_dot = (theta_next[1] - mu_rho) / cp.param_dt
    mu_gamma_dot = (theta_next[2] - mu_gamma) / cp.param_dt

    ec, ec_dot = combined_lane_error(x, t, scenario, cp.ell)

    abs_ec = torch.abs(ec) + cp.eps
    sign_ec = torch.sign(ec)

    abs_ec_gamma = torch.pow(abs_ec, mu_gamma)

    sB = ec_dot + mu_lam * ec + mu_rho * abs_ec_gamma * sign_ec

    sigma_s = (
        torch.abs(ec) * sig_lam
        + abs_ec_gamma * sig_rho
        + torch.abs(mu_rho * abs_ec_gamma * torch.log(abs_ec)) * sig_gamma
    )

    sigma_s = cp.unc_floor + cp.unc_scale * sigma_s

    kB = cp.k0 + cp.ku * sigma_s

    psi_dot = (
        mu_lam_dot * ec
        + mu_lam * ec_dot
        + mu_rho_dot * abs_ec_gamma * sign_ec
        + mu_rho * mu_gamma * torch.pow(abs_ec, mu_gamma - 1.0) * ec_dot
        + mu_rho * abs_ec_gamma * torch.log(abs_ec) * sign_ec * mu_gamma_dot
    )

    FL, GL = nominal_FL_GL(x, t, scenario, cp.ell)

    reaching = (
        kB
        * torch.pow(torch.abs(sB) + cp.eps, cp.alpha)
        * sat(sB / cp.phi)
        + cp.eta * sB
        + cp.kD * sat(sB / cp.phi)
    )

    delta = (-FL - psi_dot - reaching) / GL
    delta = torch.clamp(delta, -cp.delta_max, cp.delta_max)

    info = {
        "controller_name": "BNTSMC",
        "ec": ec,
        "ec_dot": ec_dot,
        "sB": sB,
        "V": 0.5 * sB**2,
        "sigma_s": sigma_s,
        "kB": kB,
        "mu_lambda": mu_lam,
        "mu_rho": mu_rho,
        "mu_gamma": mu_gamma,
        "kappa": scenario.kappa(t),
    }

    return delta, info


@torch.no_grad()
def fixed_tsmc_control(
    x: torch.Tensor,
    t: torch.Tensor,
    scenario: Scenario,
    cp: FixedTSMCParams,
) -> tuple[torch.Tensor, dict]:
    ec, ec_dot = combined_lane_error(x, t, scenario, cp.ell)

    abs_ec = torch.abs(ec) + cp.eps
    sign_ec = torch.sign(ec)

    s = ec_dot + cp.lam * ec + cp.rho * torch.pow(abs_ec, cp.gamma) * sign_ec

    psi_dot = cp.lam * ec_dot + cp.rho * cp.gamma * torch.pow(
        abs_ec,
        cp.gamma - 1.0,
    ) * ec_dot

    FL, GL = nominal_FL_GL(x, t, scenario, cp.ell)

    kB = torch.tensor(cp.k, dtype=DEFAULT_DTYPE, device=x.device)
    sigma_s = torch.tensor(0.0, dtype=DEFAULT_DTYPE, device=x.device)

    reaching = (cp.k + cp.kD) * sat(s / cp.phi)

    delta = (-FL - psi_dot - reaching) / GL
    delta = torch.clamp(delta, -cp.delta_max, cp.delta_max)

    info = {
        "controller_name": "TSMC",
        "ec": ec,
        "ec_dot": ec_dot,
        "sB": s,
        "V": 0.5 * s**2,
        "sigma_s": sigma_s,
        "kB": kB,
        "mu_lambda": torch.tensor(cp.lam, dtype=DEFAULT_DTYPE, device=x.device),
        "mu_rho": torch.tensor(cp.rho, dtype=DEFAULT_DTYPE, device=x.device),
        "mu_gamma": torch.tensor(cp.gamma, dtype=DEFAULT_DTYPE, device=x.device),
        "kappa": scenario.kappa(t),
    }

    return delta, info


@torch.no_grad()
def smc_control(
    x: torch.Tensor,
    t: torch.Tensor,
    scenario: Scenario,
    cp: SMCParams,
) -> tuple[torch.Tensor, dict]:
    ec, ec_dot = combined_lane_error(x, t, scenario, cp.ell)

    s = ec_dot + cp.c * ec
    psi_dot = cp.c * ec_dot

    FL, GL = nominal_FL_GL(x, t, scenario, cp.ell)

    kB = torch.tensor(cp.k, dtype=DEFAULT_DTYPE, device=x.device)
    sigma_s = torch.tensor(0.0, dtype=DEFAULT_DTYPE, device=x.device)

    reaching = (cp.k + cp.kD) * sat(s / cp.phi)

    delta = (-FL - psi_dot - reaching) / GL
    delta = torch.clamp(delta, -cp.delta_max, cp.delta_max)

    info = {
        "controller_name": "SMC",
        "ec": ec,
        "ec_dot": ec_dot,
        "sB": s,
        "V": 0.5 * s**2,
        "sigma_s": sigma_s,
        "kB": kB,
        "mu_lambda": torch.tensor(cp.c, dtype=DEFAULT_DTYPE, device=x.device),
        "mu_rho": torch.tensor(0.0, dtype=DEFAULT_DTYPE, device=x.device),
        "mu_gamma": torch.tensor(1.0, dtype=DEFAULT_DTYPE, device=x.device),
        "kappa": scenario.kappa(t),
    }

    return delta, info


def compute_control(
    predictor,
    x: torch.Tensor,
    t: torch.Tensor,
    scenario: Scenario,
    controller_name: str,
) -> tuple[torch.Tensor, dict]:

    if controller_name == "BNTSMC":
        return bntsmc_control(
            predictor,
            x,
            t,
            scenario,
            CP_BN,
        )

    if controller_name == "TSMC":
        return fixed_tsmc_control(
            x,
            t,
            scenario,
            CP_TSMC,
        )

    if controller_name == "SMC":
        return smc_control(
            x,
            t,
            scenario,
            CP_SMC,
        )

    raise ValueError(
        f"Unknown controller: {controller_name}"
    )
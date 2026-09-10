from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from bntsmc.lane_keeping.controller import BayesianSlidingNet   
# Meaning: Python is importing the class BayesianSlidingNet from your controller.py file



# ---------------ONNX wrapper for trained Bayesian neural network------------------------------------


class BNNOnnxWrapper(nn.Module): # ONNX wrapper needs access to the trained Bayesian network


    """ ONNX deployment model.
        
        Raw input:
        [ey, epsi, vy, r, kappa, vx, t]

    Outputs:
        theta_mean =
            [mu_lambda, mu_rho, mu_gamma]

        theta_std =
            [sigma_lambda, sigma_rho, sigma_gamma]

    The posterior-mean weights and 35 Bayesian posterior weight samples are stored inside the exported ONNX model."""

    def __init__(
        self,
        model: BayesianSlidingNet,
        mc_samples: int = 35,
        seed: int = 12345,
    ):
        super().__init__()

        self.mc_samples = mc_samples

        device = model.b1.weight_mu.device
        dtype = model.b1.weight_mu.dtype

        # ----------------------------------------------------
        # Input normalization ---------------------------------
        # Raw input: [ey, epsi, vy, r, kappa, vx, t]-----------
        # Z_norm = (z)/ feature_scale -------------------------
        # ----------------------------------------------------


        self.register_buffer(
            "feature_scale",     # Z_norm = (z)/ feature_scale 
            torch.tensor(
                [
                    1.0,
                    0.25,
                    3.0,
                    1.0,
                    0.015,
                    30.0,
                    25.0,
                ],
                dtype=dtype,
                device=device,
            ),
        )

        # ----------------------------------------------------
        # Store posterior-mean weights and biases---------------
        # ----------------------------------------------------


                # Posterior mean weights are copied from the trained BNN

        for name, layer in [
            ("b1", model.b1),
            ("b2", model.b2),
            ("b3", model.b3),
        ]:

            self.register_buffer(
                f"{name}_weight_mu",
                layer.weight_mu.detach().clone(),
            )

            self.register_buffer(
                f"{name}_bias_mu",
                layer.bias_mu.detach().clone(),
            )

        # ----------------------------------------------------
        # Generate fixed Bayesian posterior samples
        # This follows the same seed used in the original PyTorch inference:
        # torch.manual_seed(12345)  -->> makes random samples reproducible
        # ----------------------------------------------------

        samples = {
            "b1_w": [],
            "b1_b": [],
            "b2_w": [],
            "b2_b": [],
            "b3_w": [],
            "b3_b": [],
        }

        devices = (
            [device]
            if device.type == "cuda"
            else []
        )

        with torch.random.fork_rng(
            devices=devices
        ):

            torch.manual_seed(seed)

            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            for _ in range(mc_samples):

                for name, layer in [
                    ("b1", model.b1),
                    ("b2", model.b2),
                    ("b3", model.b3),
                ]:

                    # Bayesian posterior standard deviation
                    weight_sigma = F.softplus(
                        layer.weight_rho
                    )

                    bias_sigma = F.softplus(
                        layer.bias_rho
                    )

                    # Sample posterior weights
                    weight = (
                        layer.weight_mu
                        + weight_sigma
                        * torch.randn_like(
                            weight_sigma
                        )
                    )

                    # Sample posterior bias
                    bias = (
                        layer.bias_mu
                        + bias_sigma
                        * torch.randn_like(
                            bias_sigma
                        )
                    )

                    samples[
                        f"{name}_w"
                    ].append(
                        weight.detach().clone()
                    )

                    samples[
                        f"{name}_b"
                    ].append(
                        bias.detach().clone()
                    )

        # ----------------------------------------------------
        # Store MC samples inside ONNX wrapper
        # ----------------------------------------------------

        for name in [
            "b1",
            "b2",
            "b3",
        ]:

            self.register_buffer(
                f"{name}_weight_mc",
                torch.stack(
                    samples[f"{name}_w"],
                    dim=0,
                ),
            )

            self.register_buffer(
                f"{name}_bias_mc",
                torch.stack(
                    samples[f"{name}_b"],
                    dim=0,
                ),
            )

 
    # ----------------------------- Map raw network outputs to BNTSMC parameters-----------------------------------

    @staticmethod
    def map_output(  # --->>> map_output() is to convert the BNN’s three raw outputs into valid controller parameters. 
        raw: torch.Tensor,
    ) -> torch.Tensor:
        lam = (1.5 + 7.5 * torch.sigmoid(raw[..., 0]))  # sigmoid bound is used for: 0 < σ(h1​) < 1, so as 1.5 < λ < 9. 
        rho = (0.5 + 5.0 * torch.sigmoid(raw[..., 1])) 
        gamma = (0.35 + 0.50 * torch.sigmoid(raw[..., 2])) 
        return torch.stack([lam,rho,gamma,], dim=-1,) 

    
    #--------------------------------ONNX forward inference------------------------------------------


    def forward(
        self,
        raw_z: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        # ----------------------------------------------------
        # Normalize raw input----------------------------------
        # ----------------------------------------------------

        z = (raw_z/ self.feature_scale)

        # ---------------------------------------------
        # Posterior-mean inference--------------------
        # -------------------------------------------

        h_mean = torch.tanh(F.linear(z, self.b1_weight_mu, self.b1_bias_mu,))  # Layer 1: 7 -> 64
        h_mean = torch.tanh(F.linear(h_mean, self.b2_weight_mu, self.b2_bias_mu,))  # Layer 2: 64 -> 64

        raw_mean = F.linear( h_mean, self.b3_weight_mu, self.b3_bias_mu,)  # Output layer: 64 -> 3
        theta_mean = self.map_output(raw_mean)  # Map to [mu_lambda, mu_rho, mu_gamma]


        # Bayesian MC inference-------------------------------------------
        # z: [batch, 7]
        # b1_weight_mc: [35, 64, 7]
        # h_mc:  [35, batch, 64]
        # The normalized input is passed through 35 fixed Bayesian weight samples in parallel, 
        # producing 35 predictions \(\theta^{(j)}=[\lambda^{(j)},\rho^{(j)},\gamma^{(j)}]\).
        # These 35 predictions are then used to compute the predictive uncertainties \(\sigma_\lambda,\sigma_\rho,\sigma_\gamma\).

        h_mc = torch.matmul(z.unsqueeze(0),self.b1_weight_mc.transpose(1, 2,),)
        h_mc = ( h_mc + self.b1_bias_mc.unsqueeze(1))
        h_mc = torch.tanh(h_mc)

        # ----------------------------------------------------
        # Bayesian hidden layer 2------------------
        # ----------------------------------------------------

        h_mc = torch.matmul(
            h_mc,
            self.b2_weight_mc.transpose(
                1,
                2,
            ),
        )

        h_mc = (
            h_mc
            + self.b2_bias_mc.unsqueeze(1)
        )

        h_mc = torch.tanh(
            h_mc
        )

        # ----------------------------------------------------
        # Bayesian output layer----------------------
        # ----------------------------------------------------

        raw_mc = torch.matmul(
            h_mc,
            self.b3_weight_mc.transpose(
                1,
                2,
            ),
        )

        raw_mc = (
            raw_mc
            + self.b3_bias_mc.unsqueeze(1)
        )

        theta_mc = self.map_output(
            raw_mc
        )

  
        # Predictive standard deviation
        # Equivalent to: preds.std(dim=0) with unbiased=True
 

        mc_mean = torch.mean(
            theta_mc,
            dim=0,
        )

        variance = torch.sum(
            (
                theta_mc
                - mc_mean.unsqueeze(0)
            ) ** 2,
            dim=0,
        ) / float(
            self.mc_samples - 1
        )

        theta_std = (
            torch.sqrt(
                variance
                + 1.0e-16
            )
            + 1.0e-8
        )

        return (
            theta_mean,
            theta_std,
        )



#------------------------------------ Export trained BNN to ONNX------------------------------------


def export_bnn_to_onnx(         # After your 7000-epoch training finishes, "export_bnn_to_onnx(...)"" function is called. 
                                # t receives the trained PyTorch BNN and creates: models/bntsmc_lane.onnx
    
    model: BayesianSlidingNet, onnx_path: str | Path, mc_samples: int = 35,) -> Path:
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True,exist_ok=True,)
    model.eval()

    # --------------------------------------------------------
    # Build ONNX wrapper from trained model--------------
    # --------------------------------------------------------

    wrapper = BNNOnnxWrapper(model=model,mc_samples=mc_samples,seed=12345,)

    # --------------------------------------------------------
    # Export using CPU for easier deployment-------------------
    # --------------------------------------------------------

    wrapper = (
        wrapper
        .cpu()
        .eval()
    )

    dtype = (
        wrapper
        .b1_weight_mu
        .dtype
    )

    dummy_input = torch.zeros(
        (1, 7),
        dtype=dtype,
    )

    # --------------------------------------------------------
    # ----------------Export ONNX-----------------------------
    # --------------------------------------------------------

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_path),

        input_names=[
            "raw_input",
        ],

        output_names=[
            "theta_mean",
            "theta_std",
        ],

        dynamic_axes={
            "raw_input": {
                0: "batch_size",
            },

            "theta_mean": {
                0: "batch_size",
            },

            "theta_std": {
                0: "batch_size",
            },
        },

        opset_version=18,
    )

    print("\nSaved trained ONNX BNN:")

    print(onnx_path)

    return onnx_path



# -----------------------Main ONNX predictor used by experiment.py-------------------------------


class ONNXBNNPredictor:
    """
    Load the ONNX BNN once and perform repeated inference.

    Example:

        predictor = ONNXBNNPredictor(
            "models/bntsmc_lane.onnx"
        )

        mean, std = predictor(
            [ey, epsi, vy, r, kappa, vx, t]
        )
    """

    def __init__(
        self,
        onnx_path: str | Path,
    ):

        self.onnx_path = Path(
            onnx_path
        )

        if not self.onnx_path.exists():

            raise FileNotFoundError(
                "\nTrained ONNX BNN was not found:\n"
                f"{self.onnx_path}\n\n"
                "Run scripts/train_lane_keeping.py "
                "once before inference."
            )

        # ----------------------------------------------------
        # -------------Load ONNX model ONCE---------------
        # ----------------------------------------------------

        self.session = ort.InferenceSession(
            str(
                self.onnx_path
            ),
            providers=[
                "CPUExecutionProvider",
            ],
        )

        # ----------------------------------------------------
        # Get input/output information
        # ----------------------------------------------------

        self.input_name = (
            self.session
            .get_inputs()[0]
            .name
        )

        input_type = (
            self.session
            .get_inputs()[0]
            .type
        )

        # Match NumPy datatype with ONNX
        self.dtype = (
            np.float64
            if "double" in input_type
            else np.float32
        )

    # ========================================================
    # Repeated ONNX inference
    # ========================================================

    def __call__(
        self,
        input_vector,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:


        x = np.asarray(
            input_vector,
            dtype=self.dtype,
        )

        # Single input vector
        if x.ndim == 1:

            x = x.reshape(
                1,
                7,
            )

        if x.shape[-1] != 7:

            raise ValueError(
                "BNN ONNX input must contain "
                "7 features:\n"
                "[ey, epsi, vy, r, kappa, vx, t]"
            )

        # ----------------------------------------------------
        # ONNX inference
        # ----------------------------------------------------

        theta_mean, theta_std = (
            self.session.run(
                [
                    "theta_mean",
                    "theta_std",
                ],
                {
                    self.input_name:
                        x,
                },
            )
        )

        # For one input, return vectors of shape (3,)
        if theta_mean.shape[0] == 1:

            return (
                theta_mean[0],
                theta_std[0],
            )

        # Batch inference
        return (
            theta_mean,
            theta_std,
        )



# ----------------- low-level ONNX loader-------------------

def load_bnn_onnx(
    onnx_path: str | Path,
) -> ort.InferenceSession:
    
    """Load ONNX model and return the raw ONNX Runtime session. ONNXBNNPredictor is preferred for normal use."""

    onnx_path = Path(
        onnx_path
    )

    if not onnx_path.exists():

        raise FileNotFoundError(
            f"ONNX model not found:\n"
            f"{onnx_path}"
        )

    return ort.InferenceSession(
        str(onnx_path),
        providers=[
            "CPUExecutionProvider",
        ],
    )



# ----------------------low-level ONNX inference function---------------------------------


def bnn_onnx_inference(
    session: ort.InferenceSession,
    input_vector,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    
    """Run inference using an already-loaded ONNX session. Input:[ey, epsi, vy, r, kappa, vx, t] """

    input_type = (
        session
        .get_inputs()[0]
        .type
    )

    dtype = (
        np.float64
        if "double" in input_type
        else np.float32
    )

    x = np.asarray(
        input_vector,
        dtype=dtype,
    )

    if x.ndim == 1:

        x = x.reshape(
            1,
            7,
        )

    if x.shape[-1] != 7:

        raise ValueError(
            "ONNX BNN input must contain "
            "7 features."
        )

    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    theta_mean, theta_std = (
        session.run(
            [
                "theta_mean",
                "theta_std",
            ],
            {
                input_name:
                    x,
            },
        )
    )

    return (
        theta_mean,
        theta_std,
    )



# ------------------------------ simple output dictionary:  output = f(input, ONNX model) -------------------------------------


def bnn_output(
    session: ort.InferenceSession,
    ey: float,
    epsi: float,
    vy: float,
    r: float,
    kappa: float,
    vx: float,
    t: float,
) -> dict:

    input_vector = [
        ey,
        epsi,
        vy,
        r,
        kappa,
        vx,
        t,
    ]

    theta_mean, theta_std = (
        bnn_onnx_inference(
            session=session,
            input_vector=input_vector,
        )
    )

    return {
        "mu_lambda":
            float(
                theta_mean[0, 0]
            ),

        "mu_rho":
            float(
                theta_mean[0, 1]
            ),

        "mu_gamma":
            float(
                theta_mean[0, 2]
            ),

        "sigma_lambda":
            float(
                theta_std[0, 0]
            ),

        "sigma_rho":
            float(
                theta_std[0, 1]
            ),

        "sigma_gamma":
            float(
                theta_std[0, 2]
            ),
    }
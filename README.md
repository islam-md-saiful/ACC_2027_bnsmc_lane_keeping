# Bayesian Neural Terminal Sliding Mode Control for Lane Keeping

This repository contains the implementation of a **Bayesian Neural Terminal Sliding Mode Controller (BNTSMC)** applied to an autonomous vehicle lane-keeping system.

The Bayesian neural network (BNN) is trained offline and saved as an **ONNX model**. During the lane-keeping simulations, the trained model is used for **post-training online inference** without retraining the network.

---

## Repository Structure

```text
ACC_2027_bnsmc_lane_keeping/
│
├── models/
│   └── bntsmc_lane.onnx   # Permanently trained Bayesian neural network model
│       
│
├── scripts/
│   ├── train_lane_keeping.py   # Offline BNN training script
│   │   
│   │
│   └── lane_keeping_impl.py   # Lane-keeping simulation using trained BNN inference
│       
│
├── src/
│   └── bntsmc/
│       └── lane_keeping/
│           ├── system.py
│           ├── controller.py
│           ├── experiment.py
│           └── onnx_inference.py
│
├── pyproject.toml
├── uv.lock
└── README.md
```

<br>

---

## Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- Required Python dependencies defined in `pyproject.toml`

<br>

---

# How to Run

## 1. Clone the Repository

Clone the GitHub repository:

```bash
git clone https://github.com/islam-md-saiful/ACC_2027_bnsmc_lane_keeping.git
```

Enter the project directory:

```bash
cd ACC_2027_bnsmc_lane_keeping
```

---

## 2. Create a Virtual Environment

Create a Python 3.12 virtual environment using `uv`:

```bash
uv venv --python 3.12
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

---

## 3. Install the Dependencies

Synchronize the environment with the dependencies specified in `pyproject.toml`:

```bash
uv sync
```

Build the project:

```bash
uv build
```

---

# 4. BNN Training

The BNN training script is located at:

```text
scripts/train_lane_keeping.py
```

Run the training script from the root directory:

```bash
python scripts/train_lane_keeping.py
```

The training procedure is performed **offline**. After training, the trained Bayesian neural network is exported to ONNX format.

The trained model is saved in:

```text
models/bntsmc_lane.onnx
```

---

# 5. Lane-Keeping Simulation and Inference

After the BNN has been trained and the ONNX model has been saved, run the lane-keeping simulation using:

```bash
python scripts/lane_keeping_impl.py
```

The simulation loads the permanently trained model:

```text
models/bntsmc_lane.onnx
```

and performs **post-training online Bayesian inference** during the lane-keeping simulation.

The BNN is therefore **not retrained during the vehicle simulation**.

---

# BNTSMC Lane-Keeping Controller

The proposed **Bayesian Neural Terminal Sliding Mode Controller (BNTSMC)** is applied to the lane-keeping dynamics of the vehicle.

The BNN provides online inference of the parameters required by the terminal sliding-mode controller. The Bayesian predictive uncertainty is also used by the controller to adapt its control action under uncertain operating conditions.

The main implementation files are:

```text
src/bntsmc/lane_keeping/system.py
```

Vehicle lane-keeping dynamic model.

```text
src/bntsmc/lane_keeping/controller.py
```

BNTSMC and comparison controller implementations.

```text
src/bntsmc/lane_keeping/experiment.py
```

Simulation scenarios, experiment configuration, and performance evaluation.

```text
src/bntsmc/lane_keeping/onnx_inference.py
```

ONNX-based inference of the trained Bayesian neural network.

---

# Test Scenarios

The lane-keeping case study is evaluated under three test scenarios.

### Scenario 1: Nominal Curved-Road Tracking

The vehicle follows a curved road under nominal vehicle and tire parameters.

This scenario evaluates the basic tracking performance of the proposed BNTSMC controller.

### Scenario 2: S-Curve Road Tracking

The vehicle follows an S-shaped road trajectory with varying road curvature.

This scenario evaluates the ability of the controller to handle continuously changing road geometry.

### Scenario 3: Tire Uncertainty With Side-Wind Disturbance

The vehicle is subjected to tire-parameter uncertainty together with an external side-wind disturbance.

This scenario evaluates the robustness and uncertainty-handling capability of the proposed BNTSMC controller.

---

# Initial Conditions

For a fair comparison, the **same five initial conditions (ICs)** are used for all controllers in all three test scenarios.

Using identical initial conditions ensures that the performance comparison between the proposed BNTSMC and the baseline controllers is consistent and fair.

---

# Typical Workflow

The complete workflow is:

```text
Offline Training
      │
      ▼
train_lane_bnn.py
      │
      ▼
Bayesian Neural Network
      │
      ▼
models/bntsmc_lane.onnx
      │
      │
      ▼
Post-Training Online Inference
      │
      ▼
lane_keeping_impl.py
      │
      ▼
BNTSMC Controller
      │
      ▼
Lane-Keeping Vehicle Dynamics
      │
      ▼
Three Test Scenarios
      │
      ▼
Performance Evaluation
```




## Copyright

© 2026 Md Saiful Islam. All rights reserved.

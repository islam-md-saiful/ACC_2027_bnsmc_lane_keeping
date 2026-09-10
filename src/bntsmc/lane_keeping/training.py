from pathlib import Path

from bntsmc.common import DEVICE, set_seed
from bntsmc.lane_keeping.controller import (
    BayesianSlidingNet,
    train_bnn,
)
from bntsmc.lane_keeping.onnx_inference import (
    export_bnn_to_onnx,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models"
ONNX_PATH = MODEL_DIR / "bntsmc_lane.onnx"


def train_lane_bnn() -> Path:
    """
    Train the BNN once for 7000 epochs and export it to ONNX.
    """

    set_seed()

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = BayesianSlidingNet(
        input_dim=7,
        hidden_dim=64,
    ).to(DEVICE)

    print("\nTraining BNN for 7000 epochs...")

    train_bnn(
        model=model,
        save_dir=MODEL_DIR,
        timestamp="final",
        epochs=7000,
        batch_size=256,
        lr=2.0e-3,
    )

    model.eval()

    export_bnn_to_onnx(
        model=model,
        onnx_path=ONNX_PATH,
        mc_samples=35,
    )

    print("\nTraining completed.")
    print(f"ONNX model saved at:\n{ONNX_PATH}")

    return ONNX_PATH
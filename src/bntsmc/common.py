from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


SEED = 7
DEFAULT_DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED, deterministic_cuda: bool = True) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(DEFAULT_DTYPE)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def make_timestamp() -> str:
    """Create timestamp string for saved results."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def make_timestamped_output_dir(
    example_name: str,
    output_root: str | Path | None = None,
) -> tuple[Path, str]:
    """ ------- DynaNODE/outputs/lane_keeping/20260623_183026/ ------- """
    timestamp = make_timestamp()

    if output_root is None:
        output_root = get_project_root() / "outputs"
    else:
        output_root = Path(output_root)

    save_dir = output_root / example_name / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    return save_dir, timestamp


def configure_matplotlib() -> None:
    """ ------- Matplotlib settings ------- """
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def sat(x: torch.Tensor) -> torch.Tensor:
    """Saturation function."""
    return torch.clamp(x, -1.0, 1.0)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to NumPy array."""
    return x.detach().cpu().numpy()


def save_pdf(fig, save_dir: str | Path, filename: str, timestamp: str) -> None:
    """Save matplotlib figure as PDF."""
    save_path = Path(save_dir) / f"{filename}_{timestamp}.pdf"
    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
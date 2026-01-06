import numpy as np
from pathlib import Path

def read_bin_points(path: Path) -> np.ndarray:
    # SemanticKITTI: float32 [x,y,z,intensity]
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)

def read_poses_txt(path: Path) -> np.ndarray:
    poses = np.loadtxt(str(path), dtype=np.float64)
    if poses.ndim == 1:
        poses = poses[None, :]
    assert poses.shape[1] == 12, f"poses must be Nx12, got {poses.shape}"
    return poses

def parse_calib_txt(path: Path) -> dict:
    d = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        d[k.strip()] = np.fromstring(v.strip(), sep=" ", dtype=np.float64)
    return d

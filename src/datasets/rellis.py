import json, glob
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset

from src.utils.io import read_bin_points, read_poses_txt, parse_calib_txt
from src.utils.se3 import row12_to_T

def find_bins(seq_bin_dir: Path):
    bins = sorted(seq_bin_dir.glob("*.bin"))
    if len(bins) == 0:
        bins = sorted(Path(p) for p in glob.glob(str(seq_bin_dir / "**" / "*.bin"), recursive=True))
    return bins

def sample_fixed_points(pts: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    n = pts.shape[0]
    if n == num_points:
        return pts
    if n > num_points:
        idx = rng.choice(n, size=num_points, replace=False)
    else:
        idx = rng.choice(n, size=num_points, replace=True)
    return pts[idx]

def get_T_cam_velo(calib_dict: dict) -> np.ndarray:
    # KITTI style: Tr is 3x4 cam <- velo
    if "Tr" not in calib_dict:
        raise KeyError("calib.txt missing Tr")
    Tr = calib_dict["Tr"]
    if Tr.size != 12:
        raise ValueError(f"Tr must have 12 numbers, got {Tr.size}")
    return row12_to_T(Tr)

class RellisDirectionPairs(Dataset):
    """Dynamic loader for RELLIS sequences.

    Returns:
      pc_t:  (N,4) float32
      pc_t1: (N,4) float32
      y:     (2,)  float32  direction unit vector in XY
    """

    def __init__(self, split_json: str, split: str, num_points: int = 8192, seed: int = 42):
        assert split in ["train", "val"]
        self.split = split
        self.num_points = int(num_points)
        self.rng = np.random.default_rng(seed)

        obj = json.loads(Path(split_json).read_text(encoding="utf-8"))
        sp = obj["split"]
        self.root_pose = Path(sp["root_pose"])
        self.root_bin = Path(sp["root_bin"])
        self.seq_info = sp["sequences"]

        self.items = []         # list[(seq, pair_idx)]
        self._bins = {}         # seq -> list[Path]
        self._poses = {}        # seq -> ndarray (N,12) trimmed
        self._T_cam_velo = {}   # seq -> 4x4

        self._build_index()

    def _load_seq(self, seq: str):
        if seq in self._bins:
            return

        bins = find_bins(self.root_bin / seq)
        poses = read_poses_txt(self.root_pose / seq / "poses.txt")
        calib = parse_calib_txt(self.root_pose / seq / "calib.txt")

        # CRITICAL: poses lines can be > bin frames. We always trim to N = #bins.
        n = min(len(bins), poses.shape[0])
        bins = bins[:n]
        poses = poses[:n]

        self._bins[seq] = bins
        self._poses[seq] = poses
        self._T_cam_velo[seq] = get_T_cam_velo(calib)

    def _build_index(self):
        for seq, info in self.seq_info.items():
            self._load_seq(seq)
            n = self._poses[seq].shape[0]
            pairs = max(0, n - 1)
            if pairs == 0:
                continue

            vr = info["val_pair_range"]  # [a,b] or None
            if vr is None:
                if self.split == "train":
                    self.items.extend([(seq, i) for i in range(pairs)])
                continue

            a, b = int(vr[0]), int(vr[1])
            if self.split == "val":
                self.items.extend([(seq, i) for i in range(a, b + 1)])
            else:
                # train: before + after; never cross boundary
                if a > 0:
                    self.items.extend([(seq, i) for i in range(0, a)])
                if b + 1 < pairs:
                    self.items.extend([(seq, i) for i in range(b + 1, pairs)])

    def __len__(self):
        return len(self.items)

    def _direction_label(self, T_w_frame0: np.ndarray, T_w_frame1: np.ndarray) -> np.ndarray:
        p0 = T_w_frame0[:3, 3]
        p1 = T_w_frame1[:3, 3]
        dx, dy = float(p1[0] - p0[0]), float(p1[1] - p0[1])
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 1e-6:
            return np.array([0.0, 0.0], dtype=np.float32)
        return np.array([dx / dist, dy / dist], dtype=np.float32)

    def __getitem__(self, idx: int):
        seq, i = self.items[idx]
        self._load_seq(seq)

        bins = self._bins[seq]
        poses = self._poses[seq]
        T_cam_velo = self._T_cam_velo[seq]

        pc0 = sample_fixed_points(read_bin_points(bins[i]), self.num_points, self.rng)
        pc1 = sample_fixed_points(read_bin_points(bins[i + 1]), self.num_points, self.rng)

        # Unify pose frame with calib Tr:
        # T_w_velo = T_w_cam @ T_cam_velo
        T_w_cam0 = row12_to_T(poses[i])
        T_w_cam1 = row12_to_T(poses[i + 1])
        T_w_frame0 = T_w_cam0 @ T_cam_velo
        T_w_frame1 = T_w_cam1 @ T_cam_velo

        y = self._direction_label(T_w_frame0, T_w_frame1)

        return torch.from_numpy(pc0), torch.from_numpy(pc1), torch.from_numpy(y)

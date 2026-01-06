
# RELLIS-3D Direction Transformer (LiDAR .bin -> driving direction)

This repo trains a Transformer to predict **driving direction** from consecutive LiDAR point clouds.

## Task
- **Input:** consecutive SemanticKITTI-format LiDAR point clouds (`.bin`, float32 `[x,y,z,intensity]`)
- **Label:** 2D **direction unit vector** computed from consecutive poses in `poses.txt`
- **Important constraint (handled):** `poses.txt` often has **MORE lines** than the number of `.bin` frames.
  We always use `N = number_of_bin_frames` and ignore extra pose lines.

## Expected data layout (your case)
You have 5 sequences: `00000` ~ `00004`.

- Pose/calib root (each seq has `poses.txt` and `calib.txt`)
  - `D:/RELLIS3D/Rellis_3D_lidar_poses/Rellis-3D/00000/poses.txt`
  - `D:/RELLIS3D/Rellis_3D_lidar_poses/Rellis-3D/00000/calib.txt`
- Bin root (each seq has `.bin` frames, possibly in subfolders)
  - `D:/RELLIS3D/Rellis_3D_vel_cloud_node_kitti_bin/Rellis-3D/00000/*.bin`

## Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Configure paths
Edit `configs/rellis.yaml`:
- `data.root_pose`
- `data.root_bin`
- (optional) `data.sequences` (default is 0..4)

## 2) Create split (contiguous validation blocks per sequence)
This produces `splits/rellis_seed42.json` by default.

```bash
python scripts/make_split.py --config configs/rellis.yaml
```

Why contiguous validation blocks?
- Direction label uses **adjacent frames** (t, t+1).
- We avoid cross-boundary pairs like (train last frame, val first frame), which would produce incorrect direction.

## 3) Train
```bash
python train.py --config configs/rellis.yaml
```

Checkpoints:
- `runs/<exp_name>/best.pt`
- `runs/<exp_name>/last.pt`

## Label definition
For each pair (t, t+1):
1. Read KITTI-style pose rows (12 floats) -> 4x4 `T_w_cam`
2. Read calibration `Tr` (3x4) -> 4x4 `T_cam_velo`
3. Unify frame: `T_w_frame = T_w_cam @ T_cam_velo`
4. Direction in XY:
   - `dp = t_{t+1} - t_t`
   - `dir = normalize([dp.x, dp.y])`
5. If motion is near-zero, label is `[0,0]` and the loss masks it out.

## Notes
- `calib.txt` in your uploaded example is identity, so pose and velodyne are already aligned,
  but this repo still applies `Tr` generically (works if other sequences are not identity).

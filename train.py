import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.rellis import RellisDirectionPairs
from src.models.transformer import DirectionTransformer


@dataclass
class EvalMetrics:
    # masked (non-stationary) sample count
    n_eval: int
    # losses
    cosine_loss: float
    # direction quality
    mean_cos: float
    mean_angle_deg: float
    median_angle_deg: float
    acc_10deg: float
    acc_20deg: float
    acc_30deg: float


def masked_cosine_loss(pred: torch.Tensor, target: torch.Tensor):
    """Cosine loss with mask for near-stationary targets (target norm ~= 0)."""
    tgt_norm = torch.linalg.norm(target, dim=-1)
    mask = tgt_norm > 1e-6
    if mask.sum() == 0:
        return (pred * 0.0).sum(), mask
    pred_m = pred[mask]
    tgt_m = torch.nn.functional.normalize(target[mask], dim=-1, eps=1e-6)
    loss = 1.0 - (pred_m * tgt_m).sum(dim=-1).mean()
    return loss, mask


@torch.no_grad()
def evaluate(model, loader, device) -> EvalMetrics:
    """Recommended evaluation for direction regression:
    - cosine loss (1 - cosine similarity)
    - mean/median angular error in degrees
    - accuracy within angle thresholds (10/20/30 deg)
    All metrics are computed on non-stationary samples only.
    """
    model.eval()

    losses = []
    cos_sims = []
    angles = []

    for pc0, pc1, y in loader:
        pc0, pc1, y = pc0.to(device), pc1.to(device), y.to(device)
        pred = model(pc0, pc1)

        loss, mask = masked_cosine_loss(pred, y)
        losses.append(float(loss.item()))

        if mask.sum() == 0:
            continue

        pred_m = pred[mask]
        tgt_m = torch.nn.functional.normalize(y[mask], dim=-1, eps=1e-6)

        cs = (pred_m * tgt_m).sum(dim=-1).clamp(-1.0, 1.0)  # cosine similarity
        ang = torch.acos(cs) * (180.0 / torch.pi)          # degrees

        cos_sims.append(cs.detach().cpu())
        angles.append(ang.detach().cpu())

    if len(angles) == 0:
        # No valid samples (all stationary). Return zeros safely.
        return EvalMetrics(
            n_eval=0,
            cosine_loss=float(sum(losses) / max(1, len(losses))),
            mean_cos=0.0,
            mean_angle_deg=0.0,
            median_angle_deg=0.0,
            acc_10deg=0.0,
            acc_20deg=0.0,
            acc_30deg=0.0,
        )

    angles_all = torch.cat(angles, dim=0).numpy()
    cos_all = torch.cat(cos_sims, dim=0).numpy()

    import numpy as np

    mean_cos = float(np.mean(cos_all))
    mean_ang = float(np.mean(angles_all))
    med_ang = float(np.median(angles_all))

    acc_10 = float(np.mean(angles_all <= 10.0))
    acc_20 = float(np.mean(angles_all <= 20.0))
    acc_30 = float(np.mean(angles_all <= 30.0))

    return EvalMetrics(
        n_eval=int(angles_all.shape[0]),
        cosine_loss=float(sum(losses) / max(1, len(losses))),
        mean_cos=mean_cos,
        mean_angle_deg=mean_ang,
        median_angle_deg=med_ang,
        acc_10deg=acc_10,
        acc_20deg=acc_20,
        acc_30deg=acc_30,
    )


def format_seconds(sec: float) -> str:
    sec = max(0.0, float(sec))
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path("runs") / cfg["exp_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    split_json = cfg["data"]["split_json"]
    num_points = int(cfg["dataset"]["num_points"])
    seed = int(cfg.get("seed", 42))

    train_ds = RellisDirectionPairs(split_json, "train", num_points=num_points, seed=seed)
    val_ds = RellisDirectionPairs(split_json, "val", num_points=num_points, seed=seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
        drop_last=False,
    )

    mcfg = cfg["model"]
    model = DirectionTransformer(
        num_tokens=int(mcfg["num_tokens"]),
        d_model=int(mcfg["d_model"]),
        nhead=int(mcfg["nhead"]),
        depth=int(mcfg["depth"]),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=0.01)

    epochs = int(cfg["train"]["epochs"])
    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.json"
    summary_txt_path = run_dir / "summary.txt"

    best = 1e9
    epoch_times = []

    overall_start = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()

        # tqdm shows per-epoch ETA by default; we also estimate remaining epochs below.
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", dynamic_ncols=True)
        train_losses = []

        for pc0, pc1, y in pbar:
            pc0, pc1, y = pc0.to(device), pc1.to(device), y.to(device)

            pred = model(pc0, pc1)
            loss, _ = masked_cosine_loss(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_losses.append(float(loss.item()))
            # tqdm will show loop remaining time; add running average loss
            pbar.set_postfix(train_loss=float(sum(train_losses) / max(1, len(train_losses))))

        # Validation metrics
        val_metrics = evaluate(model, val_loader, device)

        # Timing
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining = (epochs - epoch) * avg_epoch_time
        elapsed = time.time() - overall_start

        train_loss_avg = float(sum(train_losses) / max(1, len(train_losses)))

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "cfg": cfg,
            "train_loss": train_loss_avg,
            "val_metrics": asdict(val_metrics),
        }
        torch.save(ckpt, run_dir / "last.pt")

        if val_metrics.cosine_loss < best:
            best = val_metrics.cosine_loss
            torch.save(ckpt, run_dir / "best.pt")

        # Print epoch summary with remaining time
        print(
            f"[epoch {epoch}/{epochs}] "
            f"train_loss={train_loss_avg:.6f} | "
            f"val_cosloss={val_metrics.cosine_loss:.6f} | "
            f"mean_ang={val_metrics.mean_angle_deg:.2f}° | "
            f"acc@10={val_metrics.acc_10deg:.3f} acc@20={val_metrics.acc_20deg:.3f} acc@30={val_metrics.acc_30deg:.3f} | "
            f"epoch_time={format_seconds(epoch_time)} elapsed={format_seconds(elapsed)} remaining≈{format_seconds(remaining)}"
        )

        # Append metrics row to file
        row = {
            "epoch": epoch,
            "train_loss": train_loss_avg,
            "val": asdict(val_metrics),
            "best_val_cosloss": best,
            "time": {
                "epoch_sec": epoch_time,
                "elapsed_sec": elapsed,
                "remaining_sec_est": remaining,
                "avg_epoch_sec": avg_epoch_time,
            },
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    # Final summary (also saved)
    final = {
        "exp_name": cfg["exp_name"],
        "epochs": epochs,
        "best_val_cosloss": best,
        "metrics_file": str(metrics_path),
        "checkpoints": {
            "best": str(run_dir / "best.pt"),
            "last": str(run_dir / "last.pt"),
        },
        "total_time_sec": time.time() - overall_start,
    }

    summary_path.write_text(json.dumps(final, indent=2), encoding="utf-8")

    # Human-readable summary
    summary_txt_path.write_text(
        "\n".join([
            f"Experiment: {cfg['exp_name']}",
            f"Epochs: {epochs}",
            f"Best val cosine loss: {best:.6f}",
            f"Metrics log: {metrics_path}",
            f"Best checkpoint: {run_dir / 'best.pt'}",
            f"Last checkpoint: {run_dir / 'last.pt'}",
            f"Total time: {format_seconds(final['total_time_sec'])}",
        ]) + "\n",
        encoding="utf-8",
    )

    print("\n=== FINAL SUMMARY ===")
    print(summary_txt_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

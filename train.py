import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.rellis import RellisDirectionPairs
from src.models.transformer import DirectionTransformer

def cosine_loss(pred, target):
    # pred/target: (B,2). target can be [0,0] when near-stationary; mask those out.
    tgt_norm = torch.linalg.norm(target, dim=-1)
    mask = tgt_norm > 1e-6
    if mask.sum() == 0:
        return (pred * 0.0).sum()
    pred_m = pred[mask]
    tgt_m = torch.nn.functional.normalize(target[mask], dim=-1, eps=1e-6)
    return 1.0 - (pred_m * tgt_m).sum(dim=-1).mean()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    for pc0, pc1, y in loader:
        pc0, pc1, y = pc0.to(device), pc1.to(device), y.to(device)
        pred = model(pc0, pc1)
        loss = cosine_loss(pred, y)
        losses.append(float(loss.item()))
    return sum(losses) / max(1, len(losses))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path("runs") / cfg["exp_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

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

    best = 1e9
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for pc0, pc1, y in pbar:
            pc0, pc1, y = pc0.to(device), pc1.to(device), y.to(device)

            pred = model(pc0, pc1)
            loss = cosine_loss(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            pbar.set_postfix(loss=float(loss.item()))

        val_loss = evaluate(model, val_loader, device)
        ckpt = {"epoch": epoch, "model": model.state_dict(), "cfg": cfg, "val_loss": val_loss}
        torch.save(ckpt, run_dir / "last.pt")
        if val_loss < best:
            best = val_loss
            torch.save(ckpt, run_dir / "best.pt")
        print(f"[val] cosine_loss={val_loss:.6f} (best={best:.6f})")

if __name__ == "__main__":
    main()

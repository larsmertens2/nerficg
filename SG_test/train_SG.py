import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
NPZ_ROOT = SCRIPT_DIR / "npz_files"

class MultiSGVisibility(nn.Module):
    def __init__(self, num_gaussians, num_lobes, device, bias):
        super().__init__()
        self.num_lobes = num_lobes
        self.bias = bias 
        self.axis_raw = nn.Parameter(torch.randn(num_gaussians, num_lobes, 3, device=device))
        self.sharpness_raw = nn.Parameter(torch.full((num_gaussians, num_lobes), 5.0, device=device))
        self.amplitude_raw = nn.Parameter(torch.full((num_gaussians, num_lobes), 4.0, device=device))

    def forward(self, batch_indices, cam_pos, g_pos_batch): 
        vec = g_pos_batch.unsqueeze(0) - cam_pos.unsqueeze(1)
        dist = torch.norm(vec, dim=-1, keepdim=True) + 1e-7
        view_dirs = vec / dist 

        axis = torch.nn.functional.normalize(self.axis_raw[batch_indices], dim=-1)
        sharpness = torch.nn.functional.softplus(self.sharpness_raw[batch_indices]) 
        amplitude = torch.nn.functional.softplus(self.amplitude_raw[batch_indices])

        dot = torch.einsum('cbj,blj->cbl', view_dirs, axis)
        exponent = sharpness * (dot - 1.0)
        logits = torch.sum(amplitude * torch.exp(exponent), dim=-1)

        preds = torch.sigmoid(logits - self.bias)
        return preds, sharpness, amplitude


def _iter_model_roots():
    model_roots = [
        path for path in sorted(NPZ_ROOT.iterdir())
        if path.is_dir() and (path / "camera_data.npz").exists() and (path / "gaussians_atlas.npz").exists()
    ]
    if not model_roots:
        raise FileNotFoundError(f"Geen geldige modelmappen gevonden in {NPZ_ROOT}")
    return model_roots


def train(threshold, num_lobes, model_root: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. Data Laden ---
    cam_data = np.load(model_root / "camera_data.npz")
    g_data = np.load(model_root / "gaussians_atlas.npz")
    
    cam_pos = torch.tensor(cam_data["camera_c2w"][:, :3, 3], dtype=torch.float32).to(device)
    targets = torch.tensor(cam_data["contributions"], dtype=torch.float16).to(device)
    g_pos_all = torch.tensor(g_data["means"], dtype=torch.float32).to(device)

    num_gaussians = g_pos_all.shape[0]
    num_total_cams = cam_pos.shape[0]
    
    # --- 2. Config & WandB ---
    config = {
        "num_gaussians": num_gaussians,
        "num_lobes": num_lobes,
        "learning_rate": 0.01,
        "batch_size": 15000,      
        "cams_per_batch": 100,     
        "threshold": threshold,
        "epochs": 1000,         
        "fn_weight": 3.0, 
        "bias": 4.6
    }
    
    t_str = str(threshold).replace('.', '_')
    run_name = f"SG_{model_root.name}_l{num_lobes}_t{t_str}_optimized"
    
    run = wandb.init(project="sg_visibility_ship", config=config, name=run_name, reinit=True)
    c = wandb.config

    model = MultiSGVisibility(c.num_gaussians, c.num_lobes, device, c.bias)
    optimizer = optim.Adam(model.parameters(), lr=c.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    # --- 3. Training Loop ---
    for epoch in range(c.epochs):
        indices = torch.randperm(num_gaussians, device=device)
        epoch_loss = 0
        total_fn, total_fp, total_tp, total_elements = 0, 0, 0, 0

        model.train()
        for i in range(0, num_gaussians, c.batch_size):
            optimizer.zero_grad()
            
            idx = indices[i : i + c.batch_size]
            g_pos_batch = g_pos_all[idx]
            
            # --- CAMERA SAMPLING ---
            # Kies 32 willekeurige camera-indices voor deze batch
            cam_idx = torch.randint(0, num_total_cams, (c.cams_per_batch,), device=device)
            batch_cam_pos = cam_pos[cam_idx]
            
            # Pak alleen de relevante targets [32, batch_size]
            target_batch = (targets[cam_idx][:, idx] > c.threshold).float()
            
            # Forward pass met gesamplede camera's
            preds, sharpness, amplitude = model(idx, batch_cam_pos, g_pos_batch)
            
            weights = torch.ones_like(target_batch)
            weights[target_batch == 1.0] = c.fn_weight

            loss = torch.nn.functional.binary_cross_entropy(
                preds.clamp(1e-7, 1.0 - 1e-7), 
                target_batch, 
                weight=weights
            )

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Metrics (alleen voor de laatste batch van de epoch om tijd te sparen)
            if i + c.batch_size >= num_gaussians:
                with torch.no_grad():
                    binary_preds = (preds > 0.5).float()
                    total_fn += ((target_batch == 1.0) & (binary_preds == 0.0)).sum().item()
                    total_fp += ((target_batch == 0.0) & (binary_preds == 1.0)).sum().item()
                    total_tp += ((target_batch == 1.0) & (binary_preds == 1.0)).sum().item()
                    total_elements += target_batch.numel()

        avg_loss = epoch_loss / (num_gaussians / c.batch_size)
        scheduler.step(avg_loss)
        
        if epoch % 10 == 0:
            recall = total_tp / (total_tp + total_fn + 1e-7)
            precision = total_tp / (total_tp + total_fp + 1e-7)
            
            wandb.log({
                "epoch": epoch,
                "train/loss": avg_loss,
                "metrics/recall": recall,
                "metrics/precision": precision,
                "stats/lr": optimizer.param_groups[0]['lr']
            })
            print(f"E {epoch:3} | Loss: {avg_loss:.5f} | Recall: {recall:.2f} | FPS op GPU verbeterd!")

    # --- 4. Opslaan in mappenstructuur ---
    t_folder = f"threshold_{t_str}"
    l_folder = f"lobes_{c.num_lobes}"
    base_dir = model_root / "SG" / t_folder / l_folder
    base_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"sg_l{c.num_lobes}_t{t_str}.npz"
    full_path = base_dir / filename
    
    with torch.no_grad():
        results = {
            "axis": torch.nn.functional.normalize(model.axis_raw, dim=-1).cpu().numpy(),
            "sharpness": torch.nn.functional.softplus(model.sharpness_raw).cpu().numpy(),
            "amplitude": torch.nn.functional.softplus(model.amplitude_raw).cpu().numpy(),
            "num_lobes": model.num_lobes,
            "threshold": c.threshold,
            "bias": c.bias
        }
        np.savez(full_path, **results)
    
    print(f"Opslaan voltooid: {full_path}")
    wandb.finish()
    torch.cuda.empty_cache()

if __name__ == "__main__":

    if torch.cuda.is_available():
        print(f"Running on: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda:0") 
    else:
        print("CUDA niet gevonden, switched naar CPU (pas op: traag!)")
        device = torch.device("cpu")

    thresholds = [0.01]
    lobes_options = [4, 6, 10]

    for model_root in _iter_model_roots():
        for t in thresholds:
            for l in lobes_options:
                print(f"\n--- START SG TRAINING | Model: {model_root.name} | Threshold: {t} | Lobes: {l} ---")
                train(threshold=t, num_lobes=l, model_root=model_root)
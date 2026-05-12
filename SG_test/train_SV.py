import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
NPZ_ROOT = SCRIPT_DIR / "npz_files"

class SphericalVoronoi(nn.Module):
    def __init__(self, num_gaussians, num_sites, startTemp):
        super().__init__()
        self.sites_raw = nn.Parameter(torch.randn(num_gaussians, num_sites, 3) * startTemp)
        
        # 2. De waarden (ck) MOETEN aanwezig zijn, anders voorspel je altijd 1.0
        self.values_raw = nn.Parameter(torch.randn(num_gaussians, num_sites))

    def forward(self, idx, cam_pos, g_pos_batch):
        # A. Richting van camera naar Gaussian
        vec = g_pos_batch.unsqueeze(0) - cam_pos.unsqueeze(1) 
        omega = torch.nn.functional.normalize(vec, dim=-1) # [Cams, Batch, 3]

        # B. Haal de sites op
        sites = self.sites_raw[idx] # [Batch, Sites, 3]

        # C. Bereken logits: tau * (s_hat . omega)
        # Omdat we tau uit de norm halen, is tau * s_hat gewoon 'sites'
        logits = torch.einsum('cbj,blj->cbl', omega, sites) 
        
        # D. Softmax gewichten (welke site is dominant?)
        weights = torch.nn.functional.softmax(logits, dim=-1)

        # E. Gebruik de geleerde waarden (ck) om de zichtbaarheid te bepalen
        c_k = torch.sigmoid(self.values_raw[idx]) # [Batch, Sites]
        
        # De som van gewichten * labels geeft de uiteindelijke kans (0 tot 1)
        preds = torch.sum(weights * c_k.unsqueeze(0), dim=-1)

        return preds

def _iter_model_roots():
    model_roots = [
        path for path in sorted(NPZ_ROOT.iterdir())
        if path.is_dir() and (path / "camera_data.npz").exists() and (path / "gaussians_atlas.npz").exists()
    ]
    if not model_roots:
        raise FileNotFoundError(f"Geen geldige modelmappen gevonden in {NPZ_ROOT}")
    return model_roots


def train(treshhold, sites, model_root: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Config ---
    config = {
        "num_sites": sites,
        "lr": 0.001,
        "batch_size": 15000,
        "cams_per_batch": 100,
        "epochs": 1000,
        "threshold": treshhold, 
        "startTemp": 5.0
    }
    
    t_str = str(treshhold).replace('.', '_')
    run_name = f"SV_{model_root.name}_s{config['num_sites']}_t{t_str}_temp{str(config['startTemp']).replace('.', '_')}"
    run = wandb.init(
        project="voronoi_culling_v8", 
        config=config,
        name=run_name,  
        reinit=True     
    )
    c = wandb.config

    # --- Data Laden ---
    cam_data = np.load(model_root / "camera_data.npz")
    g_data = np.load(model_root / "gaussians_atlas.npz")
    
    cam_pos = torch.tensor(cam_data["camera_c2w"][:, :3, 3], dtype=torch.float32).to(device)
    targets = torch.tensor(cam_data["contributions"], dtype=torch.float32).to(device)
    g_pos_all = torch.tensor(g_data["means"], dtype=torch.float32).to(device)

    num_gaussians = g_pos_all.shape[0]
    model = SphericalVoronoi(num_gaussians, c.num_sites, c.startTemp).to(device)
    optimizer = optim.Adam([
        {
            "params": model.sites_raw, 
            "lr": c.lr * 1.0, 
        },
        {
            "params": model.values_raw, 
            "lr": c.lr * 2.0 
        }
    ], lr=c.lr)


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     mode='min', 
                                                     factor=0.5, 
                                                     patience=20)

    # --- Training Loop ---
    num_total_cams = cam_pos.shape[0]

    for epoch in range(c.epochs):
        indices = torch.randperm(num_gaussians, device=device)
        epoch_loss = 0
        total_fn, total_fp, total_pixels = 0, 0, 0

        for i in range(0, num_gaussians, c.batch_size):
            optimizer.zero_grad()
            
            # 1. Selecteer Gaussian batch
            idx = indices[i : i + c.batch_size]
            g_pos_batch = g_pos_all[idx]
            
            # 2. CAMERA SAMPLING: Kies een subset van camera's voor deze batch
            # Dit versnelt de training enorm bij veel camera's
            cam_idx = torch.randint(0, num_total_cams, (c.cams_per_batch,), device=device)
            batch_cam_pos = cam_pos[cam_idx]
            
            # 3. Pak de bijbehorende targets [Num_Cams_Batch, Num_Gaussians_Batch]
            # Let op de volgorde van indices in targets: targets[cam_indices][:, gaussian_indices]
            target_batch = (targets[cam_idx][:, idx] > c.threshold).float()
            
            # 4. Forward pass met gesamplede camera's
            # Zorg dat je SphericalVoronoi model batch_cam_pos accepteert
            preds = model(idx, batch_cam_pos, g_pos_batch) 

            # 5. Weighting & Loss
            weights = torch.ones_like(target_batch)
            weights[target_batch == 1.0] = 5.0

            loss = torch.nn.functional.binary_cross_entropy(
                preds.clamp(1e-7, 1-1e-7), 
                target_batch, 
                weight=weights
            )
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # --- Metrics ---
            with torch.no_grad():
                binary_preds = (preds > 0.5).float()
                total_fn += ((target_batch == 1.0) & (binary_preds == 0.0)).sum().item()
                total_fp += ((target_batch == 0.0) & (binary_preds == 1.0)).sum().item()
                total_pixels += target_batch.numel()

        # Bereken gemiddelden voor de hele epoch
        avg_loss = epoch_loss / (num_gaussians / c.batch_size)
        
        scheduler.step(avg_loss)

        # Extra statistieken over de parameters zelf
        # --- Uitgebreide Metrics & Stats ---
        with torch.no_grad():
            # 1. Bereken de temperaturen (tau) uit de norm van de sites
            # sites_raw shape: [num_gaussians, num_sites, 3]
            tau = torch.norm(model.sites_raw, dim=-1) # [num_gaussians, num_sites]
            
            # 2. Bereken de actuele waarden (ck)
            current_vals = torch.sigmoid(model.values_raw)

            lr_sites = optimizer.param_groups[0]['lr']
            lr_values = optimizer.param_groups[1]['lr']
            
            # 3. Stel de log dictionary samen
            log_dict = {
                "epoch": epoch,
                "loss": avg_loss,
                "lr/sites": lr_sites,
                "lr/values": lr_values,
                
                # --- Fouten (Absoluut & Relatief) ---
                "metrics/fn_absolute": total_fn,
                "metrics/fp_absolute": total_fp,
                "metrics/fn_rate_gaten": total_fn / total_pixels,
                "metrics/fp_rate_te_veel": total_fp / total_pixels,
                "metrics/total_accuracy": 1.0 - ((total_fn + total_fp) / total_pixels),
                
                # --- Temperatuur Stats (Tau) ---
                "stats/tau_mean": tau.mean().item(),
                "stats/tau_min": tau.min().item(),
                "stats/tau_max": tau.max().item(),
                "stats/tau_std": tau.std().item(),
                
                # --- Site Waarden Stats (ck) ---
                "stats/val_mean": current_vals.mean().item(),
                "stats/val_min": current_vals.min().item(),
                "stats/val_max": current_vals.max().item(),
                "stats/val_std": current_vals.std().item(),
                
                # --- Gradiënt Sterkte (helpt bij debuggen van dode neuronen) ---
                "stats/grad_norm_sites": model.sites_raw.grad.norm().item() if model.sites_raw.grad is not None else 0,
            }
            
            # 4. Optioneel: Histogrammen (geeft veel inzicht in de verdeling)
            log_dict["histograms/tau_dist"] = wandb.Histogram(tau.cpu().numpy())
            log_dict["histograms/val_dist"] = wandb.Histogram(current_vals.cpu().numpy())
            
        wandb.log(log_dict)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.5f} | Gaten: {log_dict['metrics/fn_rate_gaten']:.4f}")


    t_folder = f"threshold_{str(c.threshold).replace('.', '_')}"
    s_folder = f"sites_{c.num_sites}"
    
    base_dir = model_root / "SV" / t_folder / s_folder
    base_dir.mkdir(parents=True, exist_ok=True)
    
    t_str = str(c.threshold).replace('.', '_')
    temp_str = str(c.startTemp).replace('.', '_')
    filename = f"sv_s{c.num_sites}_t{t_str}_temp{temp_str}.npz"
    
    full_path = base_dir / filename

    # 3. Opslaan
    values_final = torch.sigmoid(model.values_raw).detach().cpu().numpy()
    sites_final = model.sites_raw.detach().cpu().numpy()
    
    with torch.no_grad():
        np.savez(full_path, 
                 sites=sites_final,
                 values=values_final,
                 num_sites=c.num_sites,
                 threshold=c.threshold,
                 startTemp=c.startTemp)
        
    print(f"Model succesvol opgeslagen: {full_path}")
    wandb.finish()
    torch.cuda.empty_cache()

if __name__ == "__main__":

    thresholds = [0.01]
    sites_options = [8, 10,15]
    for model_root in _iter_model_roots():
        for t in thresholds:
            for s in sites_options:
                print(f"\n--- START TRAINING MODEL: {model_root.name} | THRESHOLD: {t} ---")
                train(t, s, model_root)
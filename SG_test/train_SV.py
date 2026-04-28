import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import os

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

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Config ---
    config = {
        "num_sites": 10,
        "lr": 0.001,
        "batch_size": 2500,
        "epochs": 1000,
        "threshold": 0.05, 
        "startTemp": 5.0
    }
    
    run = wandb.init(project="voronoi_culling_v6", config=config)
    c = wandb.config

    # --- Data Laden ---
    cam_data = np.load("npz_files/camera_data.npz")
    g_data = np.load("npz_files/gaussians_atlas.npz")
    
    cam_pos = torch.tensor(cam_data["camera_c2w"][:, :3, 3], dtype=torch.float32).to(device)
    targets = torch.tensor(cam_data["contributions"], dtype=torch.float32).to(device)
    g_pos_all = torch.tensor(g_data["means"], dtype=torch.float32).to(device)

    num_gaussians = g_pos_all.shape[0]
    model = SphericalVoronoi(num_gaussians, c.num_sites, c.startTemp).to(device)
    optimizer = optim.Adam([
        {
            "params": model.sites_raw, 
            "lr": c.lr * 1.0, # Iets voorzichtiger voor de geometrie
        },
        {
            "params": model.values_raw, 
            "lr": c.lr * 2.0 # Agressiever voor de zichtbaarheid-labels
        }
    ], lr=c.lr)


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     mode='min', 
                                                     factor=0.5, 
                                                     patience=20)

    # --- Training Loop ---
    for epoch in range(c.epochs):
        indices = torch.randperm(num_gaussians, device=device)
        epoch_loss = 0
        
        # Metrics aggregators
        total_fn = 0  # Gaten (Zou zichtbaar moeten zijn, maar is geculd)
        total_fp = 0  # Te veel (Zou geculd moeten worden, maar blijft zichtbaar)
        total_pixels = 0

        for i in range(0, num_gaussians, c.batch_size):
            optimizer.zero_grad()
            
            idx = indices[i : i + c.batch_size]
            target_batch = (targets[:, idx] > c.threshold).float() # [Cams, Batch]
            
            weights = torch.ones_like(target_batch)
            weights[target_batch == 1.0] = 5.0

            preds = model(idx, cam_pos, g_pos_all[idx]) # [Cams, Batch]

            # 2. Bereken de loss met dit gewicht
            loss = torch.nn.functional.binary_cross_entropy(
                preds.clamp(1e-7, 1-1e-7), 
                target_batch, 
                weight=weights
            )
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            # --- Uitgebreide Metrics Berekening ---
            with torch.no_grad():
                # Harde drempel op 0.5 voor classificatie-fouten
                binary_preds = (preds > 0.5).float()
                
                # False Negatives: Target is 1, Pred is 0
                fn = ((target_batch == 1.0) & (binary_preds == 0.0)).sum().item()
                # False Positives: Target is 0, Pred is 1
                fp = ((target_batch == 0.0) & (binary_preds == 1.0)).sum().item()
                
                total_fn += fn
                total_fp += fp
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

    # --- Opslaan ---
    os.makedirs("./npz_files", exist_ok=True)
    
    # Optioneel: vervang de punt in de threshold (0.5 -> 0_5) voor bestandsnaam-veiligheid
    t_str = str(c.threshold).replace('.', '_')
    temp_str = str(c.startTemp).replace('.', '_')
    
    filename = f"./npz_files/sv_s{c.num_sites}_t{t_str}_temp{temp_str}.npz"


    values_final = torch.sigmoid(model.values_raw).detach().cpu().numpy()
    sites_final = model.sites_raw.detach().cpu().numpy()
    
    with torch.no_grad():
        # Sla ook de hyperparameters zelf op in de NPZ, 
        # zodat je later niet hoeft te raden welke settings bij welke data horen.
        np.savez(filename, 
                 sites=sites_final,
                 values=values_final,
                 num_sites=c.num_sites,
                 threshold=c.threshold,
                 startTemp=c.startTemp)
        
    print(f"Model succesvol opgeslagen: {filename}")
    wandb.finish()

if __name__ == "__main__":
    train()
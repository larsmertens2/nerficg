import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import os

class SphericalVoronoi(nn.Module):
    def __init__(self, num_gaussians, num_sites):
        super().__init__()
        # De locaties (sk) van de sites
        self.sites_raw = nn.Parameter(torch.randn(num_gaussians, num_sites, 3))
        
        # De zichtbaarheidswaarden (ck) per site
        self.values_raw = nn.Parameter(torch.randn(num_gaussians, num_sites))
        
        # De scherpte (tau) per site (Paper Eq. 5)
        self.tau = nn.Parameter(torch.full((num_gaussians, num_sites), 10.0))

    def forward(self, idx, cam_pos, g_pos_batch):
        # 1. Bereken de view-direction 'omega' (Vector van Camera naar Gaussian)
        # Shape: [Cams, Batch, 3]
        vec = g_pos_batch.unsqueeze(0) - cam_pos.unsqueeze(1) 
        omega = torch.nn.functional.normalize(vec, dim=-1)

        # 2. Site richtingen (sk) normaliseren
        # Shape: [Batch, Sites, 3]
        sites = torch.nn.functional.normalize(self.sites_raw[idx], dim=-1)

        # 3. Bereken de Softmax Weights (wk) volgens de paper
        # Dot product tussen omega en sites: [Cams, Batch, Sites]
        dot = torch.einsum('cbj,blj->cbl', omega, sites) #sk * w
        
        # Pas temperatuur toe (tau moet positief zijn, dus softplus)
        tau = torch.nn.functional.softplus(self.tau[idx])
        logits = dot * tau # Verscherp de scores => tau*(sk * w)
        
        # De Softmax kiest de winnende site(s)
        weights = torch.nn.functional.softmax(logits, dim=-1) # automatisch  exp(τksk · ω) /som(exp(τksk · ω))

        # 4. Bereken de finale zichtbaarheid (fSV)

        c_k = torch.sigmoid(self.values_raw[idx]) #sigmoid voor tussen 0 en 1
        preds = torch.sum(weights * c_k.unsqueeze(0), dim=-1) # totale som

        return preds

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Config ---
    config = {
        "num_sites": 8,
        "lr": 0.001,
        "batch_size": 2048,
        "epochs": 1000,
        "threshold": 0.001
    }
    
    run = wandb.init(project="voronoi_culling_v2", config=config)
    c = wandb.config

    # --- Data Laden ---
    cam_data = np.load("npz_files/camera_data.npz")
    g_data = np.load("npz_files/gaussians_atlas.npz")
    
    cam_pos = torch.tensor(cam_data["camera_c2w"][:, :3, 3], dtype=torch.float32).to(device)
    targets = torch.tensor(cam_data["contributions"], dtype=torch.float32).to(device)
    g_pos_all = torch.tensor(g_data["means"], dtype=torch.float32).to(device)

    num_gaussians = g_pos_all.shape[0]
    model = SphericalVoronoi(num_gaussians, c.num_sites).to(device)
    optimizer = optim.Adam(model.parameters(), lr=c.lr)

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
            
            preds = model(idx, cam_pos, g_pos_all[idx]) # [Cams, Batch]
            
            loss = torch.nn.functional.binary_cross_entropy(preds.clamp(1e-7, 1-1e-7), target_batch)
            
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
        
        # Extra statistieken over de parameters zelf
        with torch.no_grad():
            current_tau = torch.nn.functional.softplus(model.tau)
            current_vals = torch.sigmoid(model.values_raw)
            
            log_dict = {
                "loss": avg_loss,
                "epoch": epoch,
                "metrics/false_negatives_rate (gaten)": total_fn / total_pixels,
                "metrics/false_positives_rate (te veel)": total_fp / total_pixels,
                "stats/tau_mean": current_tau.mean().item(),
                "stats/tau_max": current_tau.max().item(),
                "stats/val_mean": current_vals.mean().item(),
                "stats/val_min": current_vals.min().item(),
                "stats/val_max": current_vals.max().item(),
            }
            
        wandb.log(log_dict)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.5f} | Gaten: {log_dict['metrics/false_negatives_rate (gaten)']:.4f}")

    # --- Opslaan ---
    os.makedirs("./npz_files", exist_ok=True)
    with torch.no_grad():
        np.savez("./npz_files/trained_sv_final.npz", 
                 sites=torch.nn.functional.normalize(model.sites_raw, dim=-1).cpu().numpy(),
                 values=torch.sigmoid(model.values_raw).cpu().numpy(),
                 tau=torch.nn.functional.softplus(model.tau).cpu().numpy(),
                 num_sites=c.num_sites)
    
    wandb.finish()

if __name__ == "__main__":
    train()
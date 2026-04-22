import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import os

class MultiSGVisibility(nn.Module):
    def __init__(self, num_gaussians, num_lobes, device):
        super().__init__()
        self.num_lobes = num_lobes
        # We initialiseren met een kleine variantie om de lobes te verspreiden
        self.axis_raw = nn.Parameter(torch.randn(num_gaussians, num_lobes, 3, device=device))
        self.sharpness_raw = nn.Parameter(torch.full((num_gaussians, num_lobes), 5.0, device=device))
        self.amplitude_raw = nn.Parameter(torch.full((num_gaussians, num_lobes), 0.1, device=device))

    def forward(self, batch_indices, cam_pos, g_pos_batch):
        # A. Richting van camera naar Gaussian
        vec = g_pos_batch.unsqueeze(0) - cam_pos.unsqueeze(1)
        dist = torch.norm(vec, dim=-1, keepdim=True) + 1e-7
        view_dirs = vec / dist # [Cams, Batch, 3]

        # B. Parametrisatie (Softplus voor positiviteit)
        axis = torch.nn.functional.normalize(self.axis_raw[batch_indices], dim=-1)
        sharpness = torch.nn.functional.softplus(self.sharpness_raw[batch_indices]) 
        amplitude = torch.nn.functional.softplus(self.amplitude_raw[batch_indices])

        dot = torch.einsum('cbj,blj->cbl', view_dirs, axis)
        exponent = sharpness * (dot - 1.0)
        # Sommeer over de verschillende lobes
        preds = torch.sum(amplitude * torch.exp(exponent), dim=-1)

        return preds, sharpness, amplitude

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. Data Laden ---
    cam_data = np.load("npz_files/camera_data.npz")
    g_data = np.load("npz_files/gaussians_atlas.npz")
    
    cam_pos = torch.tensor(cam_data["camera_c2w"][:, :3, 3], dtype=torch.float32).to(device)
    targets = torch.tensor(cam_data["contributions"], dtype=torch.float32).to(device)
    g_pos_all = torch.tensor(g_data["means"], dtype=torch.float32).to(device)

    num_gaussians = g_pos_all.shape[0]
    
    # --- 2. Config & WandB ---
    config = {
        "num_gaussians": num_gaussians,
        "num_lobes": 6,
        "learning_rate": 0.001,
        "batch_size": 2500,
        "threshold": 0.01,
        "epochs": 1000,
        "fn_weight": 5.0 # Penalty voor het missen van zichtbare Gaussians
    }
    
    run = wandb.init(project="sg_visibility_v2", config=config)
    c = wandb.config

    model = MultiSGVisibility(c.num_gaussians, c.num_lobes, device)
    
    # Optimizer met verschillende LR groepen (net als in de Voronoi versie)
    optimizer = optim.Adam([
        {'params': model.axis_raw,      'lr': c.learning_rate},
        {'params': model.sharpness_raw,  'lr': c.learning_rate * 0.5},
        {'params': model.amplitude_raw,  'lr': c.learning_rate * 0.5}
    ])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    # --- 3. Training Loop ---
    for epoch in range(c.epochs):
        indices = torch.randperm(num_gaussians, device=device)
        epoch_loss = 0
        total_fn, total_fp, total_elements = 0, 0, 0
        total_tp = 0

        for i in range(0, num_gaussians, c.batch_size):
            optimizer.zero_grad()
            
            idx = indices[i : i + c.batch_size]
            g_pos_batch = g_pos_all[idx]
            # Ground truth binair maken
            target_batch = (targets[:, idx] > c.threshold).float()
            
            preds, sharpness, amplitude = model(idx, cam_pos, g_pos_batch)
            
            # Gewichten toepassen: Recall is belangrijker (False Negatives zijn duur)
            weights = torch.ones_like(target_batch)
            weights[target_batch == 1.0] = c.fn_weight

            # BCE Loss met clamping voor stabiliteit
            loss = torch.nn.functional.binary_cross_entropy(
                preds.clamp(1e-7, 1.0 - 1e-7), 
                target_batch, 
                weight=weights
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()

            # --- Metrics ---
            with torch.no_grad():
                binary_preds = (preds > 0.5).float()
                fn = ((target_batch == 1.0) & (binary_preds == 0.0)).sum().item()
                fp = ((target_batch == 0.0) & (binary_preds == 1.0)).sum().item()
                tp = ((target_batch == 1.0) & (binary_preds == 1.0)).sum().item()
                
                total_fn += fn
                total_fp += fp
                total_tp += tp
                total_elements += target_batch.numel()

        # Epoch stats
        avg_loss = epoch_loss / (num_gaussians / c.batch_size)
        scheduler.step(avg_loss)
        
        recall = total_tp / (total_tp + total_fn + 1e-7)
        precision = total_tp / (total_tp + total_fp + 1e-7)

        # Logging naar WandB
        wandb.log({
            "epoch": epoch,
            "train/loss": avg_loss,
            "metrics/recall": recall,
            "metrics/precision": precision,
            "metrics/fn_rate": total_fn / total_elements,
            "metrics/fp_rate": total_fp / total_elements,
            "stats/avg_sharpness": sharpness.mean().item(),
            "stats/avg_amplitude": amplitude.mean().item(),
            "stats/lr": optimizer.param_groups[0]['lr'],
            "histograms/amplitude": wandb.Histogram(amplitude.detach().cpu().numpy()),
            "histograms/sharpness": wandb.Histogram(sharpness.detach().cpu().numpy())
        })

        if epoch % 10 == 0:
            print(f"E {epoch:3} | Loss: {avg_loss:.5f} | Recall: {recall:.2f} | Precision: {precision:.2f} | FN: {total_fn}")

    # --- 4. Opslaan ---
    save_path = "./npz_files/trained_sg_visibility.npz"
    os.makedirs("./npz_files", exist_ok=True)
    
    with torch.no_grad():
        results = {
            "axis": torch.nn.functional.normalize(model.axis_raw, dim=-1).cpu().numpy(),
            "sharpness": torch.nn.functional.softplus(model.sharpness_raw).cpu().numpy(),
            "amplitude": torch.nn.functional.softplus(model.amplitude_raw).cpu().numpy(),
            "num_lobes": model.num_lobes,
            "threshold": c.threshold
        }
    
    np.savez(save_path, **results)
    print(f"Opslaan voltooid: {save_path}")
    wandb.finish()

if __name__ == "__main__":
    train()
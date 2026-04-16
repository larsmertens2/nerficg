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
        self.axis_raw = nn.Parameter(torch.randn(num_gaussians, num_lobes, 3, device=device))
        self.sharpness_raw = nn.Parameter(torch.full((num_gaussians, num_lobes), 5.0, device=device))
        self.amplitude_raw = nn.Parameter(torch.full((num_gaussians, num_lobes), 0.1, device=device))

    def forward(self, batch_indices, cam_pos, cam_forward, g_pos_batch):
        vec = g_pos_batch.unsqueeze(0) - cam_pos.unsqueeze(1)
        dist = torch.norm(vec, dim=-1, keepdim=True) + 1e-7
        view_dirs = vec / dist

        axis = torch.nn.functional.normalize(self.axis_raw[batch_indices], dim=-1)
        sharpness = torch.nn.functional.softplus(self.sharpness_raw[batch_indices]) 
        amplitude = torch.nn.functional.softplus(self.amplitude_raw[batch_indices])

        dot = torch.einsum('cbj,blj->cbl', view_dirs, axis)
        exponent = sharpness * (dot - 1.0)
        preds = torch.sum(amplitude * torch.exp(exponent), dim=-1)

        cos_theta = (view_dirs * cam_forward.unsqueeze(1)).sum(dim=-1)
        preds = preds * (cos_theta > 0).float()

        return preds, sharpness, amplitude

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. Data Laden ---
    cam_data = np.load("npz_files/camera_data.npz")
    g_data = np.load("npz_files/gaussians_atlas.npz")
    
    cam_pos = torch.tensor(cam_data["camera_c2w"][:, :3, 3], dtype=torch.float32).to(device)
    cam_forward = torch.tensor(cam_data["camera_c2w"][:, :3, 2], dtype=torch.float32).to(device)
    targets = torch.tensor(cam_data["contributions"], dtype=torch.float32).to(device)
    g_pos_all = torch.tensor(g_data["means"], dtype=torch.float32).to(device)

    num_cams, num_gaussians = targets.shape
    
    # --- 2. WandB Config Setup ---
    config = {
        "num_gaussians": num_gaussians,
        "num_lobes": 5,
        "learning_rate": 0.0005,
        "batch_size": 2000,
        "culling_threshold": 0.001,
        "epochs": 1000
    }
    
    # Gebruik wandb.init om de config op te slaan
    run = wandb.init(project="sg_optimiser", config=config)
    # Gebruik c (of config) als shortcut naar de waarden
    c = wandb.config

    # Gebruik de waarden uit de config
    model = MultiSGVisibility(c.num_gaussians, c.num_lobes, device)
    optimizer = optim.Adam([
        {'params': model.axis_raw, 'lr': c.learning_rate * 10},
        {'params': model.sharpness_raw, 'lr': c.learning_rate/8},
        {'params': model.amplitude_raw, 'lr': c.learning_rate/8}
    ])


    penalty_weight = 300.0 


    # --- 3. Training Loop ---
    for epoch in range(c.epochs):
        indices = torch.randperm(c.num_gaussians, device=device)
        epoch_loss = 0
        
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

        for i in range(0, c.num_gaussians, c.batch_size):
            optimizer.zero_grad()
            
            idx = indices[i : i + c.batch_size]
            g_pos_batch = g_pos_all[idx]
            target_batch = targets[:, idx]
            
            preds, sharpness, amplitude = model(idx, cam_pos, cam_forward, g_pos_batch)
            
            target_binary = (target_batch > c.culling_threshold).float()

            # 2. Clamp preds for numerical stability (BCE dislikes exactly 0 or 1)
            preds_clamped = torch.clamp(preds, 1e-7, 1.0 - 1e-7)

            # 3. Calculate weighted BCE
            # We use 'none' first so we can apply your penalty_weight manually
            bce_loss_map = torch.nn.functional.binary_cross_entropy(preds_clamped, target_binary, reduction='none')

            # 4. Apply penalty to False Negatives (where target is 1 but model predicted low)
            # This is equivalent to your 'diff > 0' logic
            loss_map = torch.where(target_binary > 0.5, penalty_weight * bce_loss_map, bce_loss_map)

            loss = loss_map.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()

            # --- Extra Metrics Berekenen ---
            with torch.no_grad():
                # Gebruik culling_threshold uit config
                is_visible_gt = (target_batch > c.culling_threshold)
                is_visible_pred = (preds > c.culling_threshold)

                total_tp += (is_visible_pred & is_visible_gt).sum().item()
                total_fp += (is_visible_pred & ~is_visible_gt).sum().item()
                total_fn += (~is_visible_pred & is_visible_gt).sum().item()
                total_tn += (~is_visible_pred & ~is_visible_gt).sum().item()

        # --- Logging stats ---
        n_batches = (c.num_gaussians / c.batch_size)
        avg_loss = epoch_loss / n_batches
        
        recall = total_tp / (total_tp + total_fn + 1e-7)
        precision = total_tp / (total_tp + total_fp + 1e-7)

        if total_fn > 1000:
            penalty_weight += 1
        else:
            penalty_weight = 100
        
        wandb.log({
            "Train/Loss": avg_loss,
            "Metrics/Recall": recall,
            "Metrics/Precision": precision,
            "Metrics/False_Negatives": total_fn,
            "Metrics/False_Positives": total_fp,
            "Params/Avg_Sharpness": sharpness.mean().item(),
            "Params/Avg_Amplitude": amplitude.mean().item(),
            "epoch": epoch,
            "penalty_weight": penalty_weight
        })
        
        if epoch % 10 == 0:
            print(f"E {epoch:3} | Loss: {avg_loss:.5f} | Recall: {recall:.2f} | FN: {total_fn} | FP: {total_fp}")

    # --- 4. Opslaan ---
    with torch.no_grad():
        results = {
            # Richting van de lobes (genormaliseerd naar lengte 1)
            "axis": torch.nn.functional.normalize(model.axis_raw, dim=-1).cpu().numpy(),
            
            # Sharpness met softplus + 1.0 voor stabiliteit (geen negatieve waarden of 0)
            "sharpness": (torch.nn.functional.softplus(model.sharpness_raw)).cpu().numpy(),
            
            # Amplitude met softplus
            "amplitude": torch.nn.functional.softplus(model.amplitude_raw).cpu().numpy(),
            "num_lobes": model.num_lobes
        }
    
    # Zorg dat de map bestaat
    os.makedirs("./npz_files", exist_ok=True)
    np.savez("./npz_files/trained_sg_culling.npz", **results)
    
    wandb.finish()

if __name__ == "__main__":
    train()
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
NPZ_ROOT = SCRIPT_DIR / "npz_files"

class MultiSGVisibility(nn.Module):
    def __init__(self, num_gaussians, num_lobes):
        super().__init__()
        self.num_lobes = num_lobes
                
        self.axis_raw = nn.Parameter(torch.randn(num_gaussians, num_lobes, 3, dtype=torch.float32))
        self.sharpness_raw = nn.Parameter(torch.full((num_gaussians, num_lobes), 5.0, dtype=torch.float32))
        self.amplitude_raw = nn.Parameter(torch.randn(num_gaussians, num_lobes, dtype=torch.float32) * 0.1)

    def forward(self, idx_gpu, cam_pos, g_pos_batch): 
        vec = g_pos_batch.unsqueeze(0) - cam_pos.unsqueeze(1)
        dist = torch.norm(vec, dim=-1, keepdim=True) + 1e-7
        view_dirs = vec / dist 

        axis = torch.nn.functional.normalize(self.axis_raw[idx_gpu], dim=-1)
        sharpness = torch.nn.functional.softplus(self.sharpness_raw[idx_gpu]) 
        
        amplitude = self.amplitude_raw[idx_gpu]

        dot = torch.einsum('cbj,blj->cbl', view_dirs, axis)
        exponent = sharpness * (dot - 1.0)
        logits = torch.sum(amplitude * torch.exp(exponent), dim=-1)
                
        return logits, sharpness, amplitude

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

    if num_lobes >= 5:
        optimal_batch = 120000
    else:
        optimal_batch = 150000
        
    config = {
        "num_lobes": num_lobes,
        "lr": 0.001,
        "batch_size": optimal_batch,      
        "cams_per_batch": 50,  
        "epochs": 750,
        "threshold": threshold,
        "fn_weight": 3.0
    }
    
    t_str = str(threshold).replace('.', '_')
    run_name = f"SG_{model_root.name}_l{config['num_lobes']}_t{t_str}_max_precision"
    
    run = wandb.init(project="sg_FINAL_62", config=config, name=run_name, reinit=True)
    c = wandb.config

    # --- Data Laden ---
    cam_data = np.load(model_root / "camera_data.npz")
    g_data = np.load(model_root / "gaussians_atlas.npz")
    
    cam_pos = torch.tensor(cam_data["camera_positions"], dtype=torch.float32)
    targets = torch.tensor(cam_data["contributions"], dtype=torch.float32) 
    g_pos_all = torch.tensor(g_data["means"], dtype=torch.float32)

    num_gaussians = g_pos_all.shape[0]
    num_total_cams = cam_pos.shape[0]

    model = MultiSGVisibility(num_gaussians, c.num_lobes)
    model = model.to(device)
    
    optimizer = optim.Adam([
        {"params": model.axis_raw, "lr": c.lr * 0.8},
        {"params": model.sharpness_raw, "lr": c.lr * 1.5},
        {"params": model.amplitude_raw, "lr": c.lr * 1.5}
    ], lr=c.lr)


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=50)

    # --- Training Loop ---
    for epoch in range(c.epochs):
        # 1. FIX: Shuffle de camera's één keer PER EPOCH, niet per batch, of fixeer de cams per epoch
        cam_idx = torch.randperm(num_total_cams, device='cpu')[:c.cams_per_batch]
        batch_cam_pos = cam_pos[cam_idx].to(device)
        
        indices = torch.randperm(num_gaussians, device='cpu')
        epoch_loss = 0
        total_fn, total_fp, total_pixels = 0, 0, 0
        num_batches = 0

        model.train()
        for i in range(0, num_gaussians, c.batch_size):
            optimizer.zero_grad()
            
            idx = indices[i : i + c.batch_size]
            g_pos_batch = g_pos_all[idx].to(device)
            
            # Gebruik de vaste camera-set voor deze epoch stap
            target_batch = (targets[cam_idx][:, idx] > c.threshold).float().to(device)
            idx_gpu = idx.to(device)
            
            logits, sharpness, amplitude = model(idx_gpu, batch_cam_pos, g_pos_batch)
            
            weights = torch.ones_like(target_batch)
            weights[target_batch == 1.0] = c.fn_weight

            # Gebruik BCE met Logits rechtstreeks voor stabiele gradients
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, 
                target_batch, 
                weight=weights
            )

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1

            # 2. FIX: Bereken metrics voor ELKE batch, niet alleen de laatste!
            with torch.no_grad():
                preds = torch.sigmoid(logits)
                binary_preds = (preds > 0.5).float()
                total_fn += ((target_batch == 1.0) & (binary_preds == 0.0)).sum().item()
                total_fp += ((target_batch == 0.0) & (binary_preds == 1.0)).sum().item()
                total_pixels += target_batch.numel()

        # Netjes middelen over het daadwerkelijke aantal batches
        avg_loss = epoch_loss / num_batches
        scheduler.step(avg_loss)
        
        with torch.no_grad():
            current_vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            
            # Bereken absolute accuratesse
            total_accuracy = 1.0 - ((total_fn + total_fp) / total_pixels)
            
            # --- NIEUW: Berekening van percentages %FN en %FP ---
            pct_fn = (total_fn / total_pixels) * 100.0
            pct_fp = (total_fp / total_pixels) * 100.0

            # --- COMPACT LOGGING ---
            log_dict = {
                "loss": avg_loss,
                "vram/current_gb": current_vram,
                "metrics/total_accuracy": total_accuracy,
                "metrics/pct_false_negatives": pct_fn, 
                "metrics/pct_false_positives": pct_fp,  
            }
            
        wandb.log(log_dict)
        
        if epoch % 10 == 0 or epoch == c.epochs - 1:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Acc: {total_accuracy:.4f} | VRAM: {current_vram:.2f}GB")

    # --- Opslaan ---
    t_folder = f"threshold_{t_str}"
    l_folder = f"lobes_{c.num_lobes}"
    base_dir = model_root / "SG" / t_folder / l_folder
    base_dir.mkdir(parents=True, exist_ok=True)
    filename = f"sg_l{c.num_lobes}_t{t_str}.npz"
    full_path = base_dir / filename
    
    with torch.no_grad():
        final_axis = torch.nn.functional.normalize(model.axis_raw, dim=-1).cpu().numpy()
        final_sharpness = torch.nn.functional.softplus(model.sharpness_raw).cpu().numpy()
        final_amplitude = model.amplitude_raw.cpu().numpy()

        np.savez(full_path, 
                 axis=final_axis,
                 sharpness=final_sharpness,
                 amplitude=final_amplitude,
                 num_lobes=model.num_lobes,
                 threshold=c.threshold)

    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak VRAM usage: {peak_vram:.2f} GB for {c.num_lobes} lobes")
    
    wandb.finish()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"Running on: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda:0") 
    else:
        print("CUDA niet gevonden, switched naar CPU")
        device = torch.device("cpu")

    thresholds = [0.0]
    lobes_options = [1, 2, 3,5,7,9]

    for model_root in _iter_model_roots():
        for t in thresholds:
            for l in lobes_options:
                
                # --- FLEXIBELE MAP-CHECK ---
                t_str_floated = str(float(t)).replace('.', '_')  # bvb. "0_0"
                t_str_int = str(int(t))                          # bvb. "0"
                
                folder_option_1 = model_root / "SG" / f"threshold_{t_str_floated}" / f"lobes_{l}"
                folder_option_2 = model_root / "SG" / f"threshold_{t_str_int}" / f"lobes_{l}"

                # Als de map al bestaat van een eerdere succesvolle run, skippen we direct
                if folder_option_1.is_dir() or folder_option_2.is_dir():
                    actual_folder = folder_option_1 if folder_option_1.is_dir() else folder_option_2
                    print(f"Skipping: {model_root.name} | Threshold: {t} | Lobes: {l} (Bestaat al: {actual_folder.relative_to(model_root)})")
                    continue
                # ----------------------------

                # --- EENMALIGE RUN PER CONFIGURATIE ---
                print(f"\n--- START SG TRAINING | Model: {model_root.name} | Threshold: {t} | Lobes: {l} ---")
                try:
                    train(threshold=t, num_lobes=l, model_root=model_root)
                    # Als train() succesvol afrondt, gaan we gewoon door naar de volgende stap in de for-loops
                    print(f"--- SUCCES | Model: {model_root.name} | Threshold: {t} | Lobes: {l} succesvol afgerond ---")
                    
                except RuntimeError as e:
                    print(f"\n[CRASH] RuntimeError opgevangen bij Model: {model_root.name}, Threshold: {t}, Lobes: {l}")
                    print(f"Foutmelding: {e}")
                    print("Gecrasht! Systeem skipt deze configuratie en gaat door naar de volgende...")
                    if wandb.run is not None:
                        wandb.finish(exit_code=1)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"\n[CRASH] Algemene fout opgevangen bij Model: {model_root.name}, Threshold: {t}, Lobes: {l}")
                    print(f"Foutmelding: {e}")
                    print("Gecrasht! Systeem skipt deze configuratie en gaat door naar de volgende...")
                    if wandb.run is not None:
                        wandb.finish(exit_code=1)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
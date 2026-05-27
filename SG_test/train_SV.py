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
        self.sites_raw = nn.Parameter(torch.randn(num_gaussians, num_sites, 3, dtype=torch.float32) * startTemp)
        self.values_raw = nn.Parameter(torch.randn(num_gaussians, num_sites, dtype=torch.float32))

    def forward(self, idx_gpu, cam_pos, g_pos_batch):
        # A. Richting van camera naar Gaussian
        vec = g_pos_batch.unsqueeze(0) - cam_pos.unsqueeze(1) 
        omega = torch.nn.functional.normalize(vec, dim=-1) # [Cams, Batch, 3]

        # B. Slicen op de GPU
        sites = self.sites_raw[idx_gpu] # [Batch, Sites, 3]

        # C. Bereken logits
        logits = torch.einsum('cbj,blj->cbl', omega, sites) 
        
        # D. Softmax gewichten
        weights = torch.nn.functional.softmax(logits, dim=-1)

        # E. Gebruik de geleerde waarden
        c_k = torch.sigmoid(self.values_raw[idx_gpu]) # [Batch, Sites]
        
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

    if sites >= 5:
        optimal_batch = 120000
    else:
        optimal_batch = 150000

    config = {
        "num_sites": sites,
        "lr": 0.001,
        "batch_size": optimal_batch,     
        "cams_per_batch": 50,   
        "epochs": 750,
        "threshold": treshhold, 
        "startTemp": 5.0
    }
    
    t_str = str(treshhold).replace('.', '_')
    run_name = f"SV_{model_root.name}_s{config['num_sites']}_t{t_str}_temp{str(config['startTemp']).replace('.', '_')}"
    run = wandb.init(project="sv_FINAL_62", config=config, name=run_name, reinit=True)
    c = wandb.config

    # --- Data Laden ---
    cam_data = np.load(model_root / "camera_data.npz")
    g_data = np.load(model_root / "gaussians_atlas.npz")

    cam_pos = torch.tensor(cam_data["camera_positions"], dtype=torch.float32)
    g_pos_all = torch.tensor(g_data["means"], dtype=torch.float32)
    targets = torch.tensor(cam_data["contributions"], dtype=torch.float32)

    num_gaussians = g_pos_all.shape[0]
    num_total_cams = cam_pos.shape[0]
    
    model = SphericalVoronoi(num_gaussians, c.num_sites, c.startTemp)
    model = model.to(device)
    
    optimizer = optim.Adam([
        {"params": model.sites_raw, "lr": c.lr * 1.0},
        {"params": model.values_raw, "lr": c.lr * 1.5}
    ], lr=c.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=50)

    # --- Training Loop ---
    for epoch in range(c.epochs):
        # 1. FIX: Kies de camera-set ÉÉN keer per epoch (niet per sub-batch). 
        # Hierdoor blijft het doelwit stabiel tijdens het afreizen van alle Gaussians.
        cam_idx = torch.randperm(num_total_cams, device='cpu')[:c.cams_per_batch]
        batch_cam_pos = cam_pos[cam_idx].to(device)
        
        indices = torch.randperm(num_gaussians, device='cpu')
        epoch_loss = 0
        total_fn, total_fp, total_pixels = 0, 0, 0
        num_batches = 0  # FIX: Tellers bijhouden voor exacte middeling

        for i in range(0, num_gaussians, c.batch_size):
            optimizer.zero_grad()
            
            idx = indices[i : i + c.batch_size]
            g_pos_batch = g_pos_all[idx].to(device)
            
            # Gebruik de vaste camera_idx voor deze sub-batch targets
            target_batch = (targets[cam_idx][:, idx] > c.threshold).float().to(device)
            idx_gpu = idx.to(device)
            
            # Forward pass (model geeft direct kansen [0, 1] terug)
            preds = model(idx_gpu, batch_cam_pos, g_pos_batch) 

            weights = torch.ones_like(target_batch)
            weights[target_batch == 1.0] = 3.0

            # BCE met handmatige clamp (behouden zoals gevraagd)
            loss = torch.nn.functional.binary_cross_entropy(
                preds.clamp(1e-7, 1.0 - 1e-7), 
                target_batch, 
                weight=weights
            )
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1

            # 2. FIX: Bereken metrics voor álle batches, zodat je accuracy representatief is
            with torch.no_grad():
                binary_preds = (preds > 0.5).float()
                total_fn += ((target_batch == 1.0) & (binary_preds == 0.0)).sum().item()
                total_fp += ((target_batch == 0.0) & (binary_preds == 1.0)).sum().item()
                total_pixels += target_batch.numel()

        # 3. FIX: Altijd nauwkeurig delen door het daadwerkelijk gedraaide aantal batches
        avg_loss = epoch_loss / num_batches
        scheduler.step(avg_loss)
        
        with torch.no_grad():
            current_vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            total_accuracy = 1.0 - ((total_fn + total_fp) / total_pixels)
            pct_fn = (total_fn / total_pixels) * 100.0
            pct_fp = (total_fp / total_pixels) * 100.0
            
            # --- COMPACT LOGGING (Alleen loss, lr, vram en accuracy) ---
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
    t_folder = f"threshold_{str(c.threshold).replace('.', '_')}"
    s_folder = f"sites_{c.num_sites}"
    base_dir = model_root / "SV" / t_folder / s_folder
    base_dir.mkdir(parents=True, exist_ok=True)
    
    t_str = str(c.threshold).replace('.', '_')
    temp_str = str(c.startTemp).replace('.', '_')
    filename = f"sv_s{c.num_sites}_t{t_str}_temp{temp_str}.npz"
    full_path = base_dir / filename

    with torch.no_grad():
        values_final = torch.sigmoid(model.values_raw).cpu().numpy()
        sites_final = model.sites_raw.cpu().numpy()
    
    np.savez(full_path, 
             sites=sites_final,
             values=values_final,
             num_sites=c.num_sites,
             threshold=c.threshold,
             startTemp=c.startTemp)

    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak VRAM usage: {peak_vram:.2f} GB for {c.num_sites} sites")
        
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
    sites_options = [1, 2, 3,5,7,9]

    for model_root in _iter_model_roots():
        for t in thresholds:
            for s in sites_options:
                
                # --- FLEXIBELE MAP-CHECK ---
                t_str_floated = str(float(t)).replace('.', '_')  # bvb. "0_0"
                t_str_int = str(int(t))                          # bvb. "0"
                
                folder_option_1 = model_root / "SV" / f"threshold_{t_str_floated}" / f"sites_{s}"
                folder_option_2 = model_root / "SV" / f"threshold_{t_str_int}" / f"sites_{s}"

                # Als de map al bestaat van een eerdere succesvolle run, skippen we direct
                if folder_option_1.is_dir() or folder_option_2.is_dir():
                    actual_folder = folder_option_1 if folder_option_1.is_dir() else folder_option_2
                    print(f"Skipping: {model_root.name} | Threshold: {t} | Sites: {s} (Bestaat al: {actual_folder.relative_to(model_root)})")
                    continue
                # ----------------------------

                # --- WHILE LOOP VOOR RETRIES ---
                while True:
                    print(f"\n--- START SV TRAINING | Model: {model_root.name} | Threshold: {t} | Sites: {s} ---")
                    try:
                        train(t, s, model_root)
                        # Als train() zonder errors finisht, breken we uit de infinite while loop
                        print(f"--- SUCCES | Model: {model_root.name} | Threshold: {t} | Sites: {s} succesvol afgerond ---")
                        break
                        
                    except RuntimeError as e:
                        print(f"\n[CRASH] RuntimeError opgevangen bij Model: {model_root.name}, Threshold: {t}, Sites: {s}")
                        print(f"Foutmelding: {e}")
                        print("Systeem probeert het direct opnieuw...")
                        if wandb.run is not None:
                            wandb.finish(exit_code=1)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        print(f"\n[CRASH] Algemene fout opgevangen bij Model: {model_root.name}, Threshold: {t}, Sites: {s}")
                        print(f"Foutmelding: {e}")
                        print("Systeem probeert het direct opnieuw...")
                        if wandb.run is not None:
                            wandb.finish(exit_code=1)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
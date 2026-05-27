import numpy as np
import torch
import gc

def inspect_contributions(file_path="npz_files\drums\camera_data.npz"):
    print("--- Memory-Efficient Contribution Deep Dive ---")
    
    # Gebruik mmap_mode om het bestand niet direct volledig in RAM te laden
    cam_data = np.load(file_path, mmap_mode='r')
    raw_contributions = cam_data["contributions"]
    
    num_cams, num_gaussians = raw_contributions.shape
    total_elements = num_cams * num_gaussians
    thresholds = [0.0, 0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05]
    
    # Accumulatoren
    total_active = 0
    total_zeros = 0  # <--- NIEUW: Teller voor exacte nullen
    counts_under = {t: 0 for t in thresholds}
    all_positive_samples = [] 
    global_max = 0.0  

    chunk_size = 5000  
    
    for i in range(0, num_gaussians, chunk_size):
        end = min(i + chunk_size, num_gaussians)
        
        # Laad strip
        batch = torch.from_numpy(raw_contributions[:, i:end].astype(np.float32))
        
        # Bereken lokale max van deze batch en update de globale max
        current_max = torch.max(batch).item()
        if current_max > global_max:
            global_max = current_max
        
        # Tel exacte nullen
        total_zeros += (batch == 0).sum().item() # <--- NIEUW: Telt nullen in deze batch
        
        positive_mask = batch > 0
        pos_values = batch[positive_mask]
        
        total_active += pos_values.numel()
        
        for t in thresholds:
            counts_under[t] += (pos_values < t).sum().item()
            
        if len(all_positive_samples) < 1_000_000:
            all_positive_samples.append(pos_values[:10000])

        del batch, positive_mask, pos_values
        if i % 50000 == 0:
            print(f"Progress: {i/num_gaussians*100:.1f}%")

    # Bereken percentages voor de algemene statistieken
    perc_zeros = (total_zeros / total_elements) * 100
    perc_active = (total_active / total_elements) * 100

    print("-" * 40)
    print(f"Dataset Stats:")
    print(f"  Totaal aantal labels:      {total_elements:,}")
    print(f"  Exact nul (0.0):           {total_zeros:,} ({perc_zeros:.2f}%)") # <--- NIEUW
    print(f"  Actieve (niet-nul) labels: {total_active:,} ({perc_active:.2f}%)") # <--- Nu met %
    print(f"  Maximale waarde gevonden:  {global_max:.4f}")
    print("-" * 40)

    if total_active > 0:
        print("Verdeling van de ACTIEVE labels (onder drempelwaarden):")
        for t in thresholds:
            perc_under = (counts_under[t] / total_active) * 100
            print(f"  Onder {t:.4f}: {perc_under:5.2f}% ({counts_under[t]:,} van de {total_active:,})")
        
        approx_median = torch.cat(all_positive_samples).median().item()
        print("-" * 40)
        print(f"Benaderde Mediaan (van actieve labels): {approx_median:.6f}")
    else:
        print("Geen positieve bijdrages gevonden!")

    gc.collect()

if __name__ == "__main__":
    inspect_contributions()
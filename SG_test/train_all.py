import torch
import torch.nn as nn
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from train import train_multi_sg_visibility 

def benchmark_performance(num_lobes, device):
    """Meet hoe snel de GPU de 317k punten kan filteren."""
    sg_model = np.load(f"npz_files/trained_sg_{num_lobes}_lobes.npz")
    axis = torch.tensor(sg_model["axis"]).to(device)
    sharpness = torch.tensor(sg_model["sharpness"]).to(device)
    amplitude = torch.tensor(sg_model["amplitude"]).to(device)
    view_dir = torch.randn(3, device=device) # Simuleer een kijkrichting
    
    # Warmsup
    torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(100): # Doe 100 metingen voor een goed gemiddelde
        axis_norm = torch.nn.functional.normalize(axis, dim=-1)
        dot = torch.einsum('d,gld->gl', view_dir, axis_norm)
        preds = torch.sum(amplitude * torch.exp(sharpness * (dot - 1)), dim=-1)
        _ = preds > 0.5
    
    torch.cuda.synchronize()
    avg_time = (time.time() - start_time) / 100 * 1000 # In milliseconden
    
    # Geheugengebruik (Axis=3f, Sharp=1f, Amp=1f) -> 5 floats per lob * 4 bytes
    vram_mb = (len(axis) * num_lobes * 5 * 4) / (1024 * 1024)
    
    return avg_time, vram_mb

def evaluate_metrics(num_lobes, threshold=0.5):
    cam_data = np.load("npz_files/camera_data.npz")
    sg_model = np.load(f"npz_files/trained_sg_{num_lobes}_lobes.npz")
    
    # Gemiddelde over meerdere camera's voor een eerlijk beeld
    total_acc, total_fn, total_fp = [], [], []
    for c_idx in [0, 10, 50]:
        view_dir = cam_data["camera_c2w"][c_idx, :3, 2]
        view_dir /= np.linalg.norm(view_dir)
        gt = cam_data["contributions"][c_idx] > 0
        
        dot = np.einsum('d,gld->gl', view_dir, sg_model["axis"])
        preds = np.sum(sg_model["amplitude"] * np.exp(sg_model["sharpness"] * (dot - 1)), axis=1)
        
        pred_vis = preds > threshold
        total_acc.append(np.mean(pred_vis == gt) * 100)
        total_fn.append(np.sum(~pred_vis & gt))
        total_fp.append(np.sum(pred_vis & ~gt))
        
    return np.mean(total_acc), np.mean(total_fn), np.mean(total_fp)

def run_all():
    lobe_counts = [1, 2, 3, 4, 5]
    metrics = {"acc": [], "fn": [], "fp": [], "ms": [], "mb": []}
    device = torch.device("cuda")

    for n in lobe_counts:
        # 1. Train (als bestand nog niet bestaat)
        fname = f"npz_files/trained_sg_{n}_lobes.npz"
        if not os.path.exists(fname):
            train_multi_sg_visibility(num_lobes=n)
            os.rename("npz_files/trained_multi_sg_atlas.npz", fname)
        
        # 2. Meet kwaliteit
        acc, fn, fp = evaluate_metrics(n)
        # 3. Meet performance
        ms, mb = benchmark_performance(n, device)
        
        metrics["acc"].append(acc)
        metrics["fn"].append(fn)
        metrics["fp"].append(fp)
        metrics["ms"].append(ms)
        metrics["mb"].append(mb)

    # --- PLOTTING ---
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Kwaliteit (Accuracy & Gaten)
    axs[0,0].plot(lobe_counts, metrics["acc"], 'b-o', label='Accuracy %')
    axs[0,0].set_title("Algemene Nauwkeurigheid")
    axs[0,0].set_ylabel("%")
    
    ax_fn = axs[0,0].twinx()
    ax_fn.plot(lobe_counts, metrics["fn"], 'r--s', label='Gaten (FN)')
    ax_fn.set_ylabel("Aantal Gaten")
    axs[0,0].legend(loc='upper left')

    # 2. Overhead (Te veel Gaussians)
    axs[0,1].bar(lobe_counts, metrics["fp"], color='orange', alpha=0.7)
    axs[0,1].set_title("Ruis (False Positives)")
    axs[0,1].set_ylabel("Aantal onnodige Gaussians")

    # 3. Rekentijd (MS)
    axs[1,0].plot(lobe_counts, metrics["ms"], 'g-^')
    axs[1,0].set_title("Inference Tijd op GPU (3050)")
    axs[1,0].set_ylabel("ms per frame")

    # 4. Geheugen (MB)
    axs[1,1].bar(lobe_counts, metrics["mb"], color='purple')
    axs[1,1].set_title("Atlas Bestandsgrootte")
    axs[1,1].set_ylabel("MB op VRAM")

    plt.tight_layout()
    plt.savefig("benchmark_vis_performance.png")
    plt.show()

if __name__ == "__main__":
    run_all()
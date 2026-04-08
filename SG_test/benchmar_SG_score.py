import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def evaluate_lobes_over_thresholds():
    # 1. Data laden
    cam_data = np.load("npz_files/camera_data.npz")
    cam_idx = 10 # We pakken een representatieve camera voor de test
    
    view_dir = cam_data["camera_c2w"][cam_idx, :3, 2]
    view_dir /= np.linalg.norm(view_dir)
    gt_visibility = cam_data["contributions"][cam_idx] > 0
    
    lobe_configs = [1, 2, 3, 4, 5]
    thresholds = np.linspace(0.01, 0.99, 50) # 50 stappen tussen 0 en 1
    
    plt.figure(figsize=(12, 8))
    
    for n in lobe_configs:
        fname = f"npz_files/trained_sg_{n}_lobes.npz"
        if not os.path.exists(fname):
            print(f"Skipping {n} lobes: {fname} niet gevonden.")
            continue
            
        sg_model = np.load(fname)
        axis = sg_model["axis"]
        sharpness = sg_model["sharpness"]
        amplitude = sg_model["amplitude"]

        # SG Activatie berekenen voor deze configuratie
        dot = np.einsum('d,gld->gl', view_dir, axis)
        activations = amplitude * np.exp(sharpness * (dot - 1))
        sg_preds = np.sum(activations, axis=1)

        # Bereken accuracy voor elke drempel
        acc_list = []
        for t in thresholds:
            pred_vis = sg_preds > t
            accuracy = np.mean(pred_vis == gt_visibility) * 100
            acc_list.append(accuracy)
            
        # Voeg lijn toe aan de grafiek
        plt.plot(thresholds, acc_list, label=f'{n} Lobben', linewidth=2)

    plt.title("Gevoeligheidsanalyse: Accuracy vs Threshold per aantal Lobben", fontsize=14)
    plt.xlabel("Threshold (Drempelwaarde)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axvline(x=0.5, color='r', linestyle=':', label='Standaard 0.5 drempel')
    
    plt.savefig("threshold_sensitivity_comparison.png")
    plt.show()

if __name__ == "__main__":
    evaluate_lobes_over_thresholds()
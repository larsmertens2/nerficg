import torch

import torch.nn as nn

import torch.optim as optim

import numpy as np

import os



def train_multi_sg_visibility(num_lobes=3):

    device = torch.device("cuda")

    print(f"🚀 Training Multi-SG ({num_lobes} lobes) op: {torch.cuda.get_device_name(0)}")



    # 1. Data laden

    if not os.path.exists("npz_files/camera_data.npz"):

        print("Fout: npz_files/camera_data.npz niet gevonden!")

        return



    cam_data = np.load("npz_files/camera_data.npz")

    targets_raw = torch.tensor(cam_data["contributions"], dtype=torch.float32).to(device)

    binary_targets = (targets_raw > 0).float()

   

    num_cams, num_gaussians = binary_targets.shape

   

    view_dirs = torch.tensor(cam_data["camera_c2w"][:, :3, 2], dtype=torch.float32).to(device)

    view_dirs = torch.nn.functional.normalize(view_dirs, dim=-1)



    # 2. Parameters initialiseren voor N lobben

    # [Gaussians, Lobes, 3] voor de richtingen

    axis = nn.Parameter(torch.randn(num_gaussians, num_lobes, 3, device=device))

    # [Gaussians, Lobes] voor de scherpte en amplitude

    sharpness = nn.Parameter(torch.ones(num_gaussians, num_lobes, device=device) * 10.0)

    amplitude = nn.Parameter(torch.ones(num_gaussians, num_lobes, device=device) * (1.0 / num_lobes))



    # Stabiele optimizer instellingen

    optimizer = optim.Adam([

        {'params': axis, 'lr': 0.005},

        {'params': sharpness, 'lr': 0.015},

        {'params': amplitude, 'lr': 0.005}

    ])



    # Batching om binnen 4GB VRAM te blijven

    batch_size = 1500



    print(f"Start training voor {num_gaussians} Gaussians met {num_lobes} lobben per punt...")



    for step in range(1001):

        total_loss = 0

       

        for i in range(0, num_gaussians, batch_size):

            optimizer.zero_grad()

            end = min(i + batch_size, num_gaussians)

           

            b_targets = binary_targets[:, i:end] # [Cams, Batch]

            b_axis = axis[i:end]                 # [Batch, Lobes, 3]

            b_sharp = sharpness[i:end]           # [Batch, Lobes]

            b_amp = amplitude[i:end]             # [Batch, Lobes]



            # Multi-SG Berekening

            # Normaliseer assen

            axis_norm = torch.nn.functional.normalize(b_axis, dim=-1)

           

            # Dot product tussen alle camera's en alle lobben in de batch

            # cd: cams, bld: batch*lobes -> uitkomst cbl: cams*batch*lobes

            dot = torch.einsum('cd,bld->cbl', view_dirs, axis_norm)

           

            # Bereken de bijdrage per lob en sommeer ze (dim=-1)

            # Formule: Sum( Amp * exp( Sharp * (dot - 1) ) )

            preds = torch.sum(b_amp * torch.exp(b_sharp * (dot - 1)), dim=-1)

           

            # MSE Loss vergeleken met binaire target

            loss = torch.nn.functional.mse_loss(preds, b_targets)

            loss.backward()

            optimizer.step()

           

            total_loss += loss.item()



        # Constraints: Voorkom dat waarden onrealistisch worden

        with torch.no_grad():

            amplitude.clamp_(0.0, 1.0)

            sharpness.clamp_(0.1, 150.0)



        if step % 20 == 0:

            avg_loss = total_loss / (num_gaussians / batch_size)

            print(f"Stap {step:3d} | Gemiddelde Loss: {avg_loss:.6f}")



    # 3. Opslaan

    print(f"💾 Opslaan naar trained_multi_sg_atlas.npz...")

    np.savez("npz_files/trained_multi_sg_atlas.npz",

             axis=axis.detach().cpu().numpy(),

             sharpness=sharpness.detach().cpu().numpy(),

             amplitude=amplitude.detach().cpu().numpy(),

             num_lobes=num_lobes)

    print("✅ Training succesvol voltooid.")



if __name__ == "__main__":

    train_multi_sg_visibility(num_lobes=3)
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_middle_check(vol_id, output_dir_str):
    output_dir = Path(output_dir_str)
    ct_dir = output_dir / "ct"
    label_dir = output_dir / "labels"
    simus_dir = output_dir / "simus"
    
    # Get sorted list of files for this volunteer
    files = sorted([f for f in os.listdir(ct_dir)
                    if f.startswith(vol_id) and f.endswith(".npy")])
    if not files:
        print(f"No files found for {vol_id}")
        return
        
    n_files = len(files)
    print(f"Volunteer {vol_id}: total saved frames = {n_files}")
    
    # Pick 4 frames evenly spaced from 20% to 80% of the sweep to see the middle
    idxs = [int(i) for i in np.linspace(int(n_files * 0.2), int(n_files * 0.8), 4)]
    
    fig, axes = plt.subplots(4, 3, figsize=(9, 12))
    
    for i, idx in enumerate(idxs):
        fname = files[idx]
        ct = np.load(ct_dir / fname)
        lbl = np.load(label_dir / fname)
        us = np.load(simus_dir / fname)
        
        # Calculate non-zero percent
        ct_non_zero = (ct != 0).mean() * 100.0
        
        axes[i, 0].imshow(ct, cmap="gray")
        axes[i, 0].set_title(f"CT Slice {fname} ({ct_non_zero:.1f}% inside)")
        axes[i, 1].imshow(lbl, cmap="gray")
        axes[i, 1].set_title("Bone Mask")
        axes[i, 2].imshow(us, cmap="gray")
        axes[i, 2].set_title("US Frame")
        
        for ax in axes[i]:
            ax.axis("off")
            
    fig.suptitle(f"{vol_id} — Middle Sweep Alignment Check", fontsize=13)
    fig.tight_layout()
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)
    out_path = diag_dir / f"{vol_id}_middle_check.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved middle check image to {out_path}")

for v in ["URS02", "URS03"]:
    plot_middle_check(v, "./data/cavalcanti_processed")

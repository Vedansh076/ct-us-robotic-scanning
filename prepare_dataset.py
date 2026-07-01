import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

def preprocess_ultrabones():
    data_root = Path("data/UltraBones100k")
    output_root = Path("dataset")
    
    # 1. Clear previous dataset directory to avoid stale files
    if output_root.exists():
        print("[+] Clearing old dataset folder...")
        shutil.rmtree(output_root)
        
    # Create clean directories
    for folder in ["ct", "simus", "labels", "poses"]:
        (output_root / folder).mkdir(parents=True, exist_ok=True)
        
    print("[+] Preprocessing all 14 specimens from UltraBones100k...")
    
    train_specimens = [f"specimen{i:02d}" for i in range(1, 11)]
    val_specimens = [f"specimen{i:02d}" for i in range(11, 15)]
    
    total_train = 0
    total_val = 0
    
    # Loop over all specimens
    for idx in range(1, 15):
        specimen_name = f"specimen{idx:02d}"
        specimen_dir = data_root / specimen_name / "ultrasound_records" / "tibia"
        
        if not specimen_dir.exists():
            print(f"    [SKIP] Tibia sweeps for {specimen_name} not found.")
            continue
            
        print(f"    Processing {specimen_name}...")
        sample_idx = 0
        
        # Loop over sweeps within the tibia directory
        for sweep_dir in sorted(specimen_dir.iterdir()):
            if not sweep_dir.is_dir():
                continue
                
            us_dir = sweep_dir / "UltrasoundImages"
            lbl_dir = sweep_dir / "Labels"
            lbl_full_dir = sweep_dir / "Labels_full"
            
            if not (us_dir.exists() and lbl_dir.exists() and lbl_full_dir.exists()):
                continue
                
            # Iterate through images, subsampling every 5th frame
            img_paths = sorted(list(us_dir.glob("*.png")))
            subsampled_paths = img_paths[::5]  # Subsample every 5th frame
            
            for img_path in subsampled_paths:
                stem = img_path.stem
                
                # Check for corresponding label files
                lbl_path = lbl_dir / f"{stem}_label.png"
                lbl_full_path = lbl_full_dir / f"{stem}_label.png"
                
                # Fallbacks if files are named identically to the images
                if not lbl_path.exists():
                    lbl_path = lbl_dir / f"{stem}.png"
                if not lbl_full_path.exists():
                    lbl_full_path = lbl_full_dir / f"{stem}.png"
                    
                if not (lbl_path.exists() and lbl_full_path.exists()):
                    continue
                    
                try:
                    # Load and resize
                    us_img = Image.open(img_path).convert("L").resize((256, 256), Image.BILINEAR)
                    lbl_img = Image.open(lbl_path).convert("L").resize((256, 256), Image.NEAREST)
                    lbl_full_img = Image.open(lbl_full_path).convert("L").resize((256, 256), Image.NEAREST)
                except Exception as exc:
                    continue
                    
                # Process to numpy arrays
                # 1. SimUS target: normalize by converting [0, 255] to [0, 220]
                us_arr = np.array(us_img, dtype=np.float32) * (220.0 / 255.0)
                
                # 2. CT source: convert binary mask to soft tissue range [-200, 300] HU
                # So background becomes -200 HU, foreground (bone) becomes 300 HU
                lbl_full_bin = (np.array(lbl_full_img, dtype=np.float32) > 127.0).astype(np.float32)
                ct_arr = lbl_full_bin * 500.0 - 200.0
                
                # 3. Label: visible bone boundary mask in [0, 1]
                lbl_bin = (np.array(lbl_img, dtype=np.float32) > 127.0).astype(np.float32)
                
                # Save filename matching subject prefix for dataset filter
                filename = f"{specimen_name}_{sample_idx:05d}.npy"
                
                np.save(output_root / "ct" / filename, ct_arr)
                np.save(output_root / "simus" / filename, us_arr)
                np.save(output_root / "labels" / filename, lbl_bin)
                
                # Save a dummy pose dictionary to prevent dataloader failure
                pose_dict = {
                    "subject_id": specimen_name,
                    "center": np.zeros(3, dtype=np.float32),
                    "quaternion": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                }
                np.save(output_root / "poses" / filename, pose_dict)
                
                sample_idx += 1
                
        if specimen_name in train_specimens:
            total_train += sample_idx
        else:
            total_val += sample_idx
            
        print(f"      Created {sample_idx} samples for {specimen_name}.")
        
    print(f"\n[+] Preprocessing complete! Created:")
    print(f"    Total train samples (specimen01-10): {total_train}")
    print(f"    Total val samples (specimen11-14)  : {total_val}")

if __name__ == "__main__":
    preprocess_ultrabones()

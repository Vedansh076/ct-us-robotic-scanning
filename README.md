# CT-to-Ultrasound Robotic Scanning Simulation

A simulation environment in PyBullet to navigate a virtual robotic ultrasound probe over patient skin surfaces and synthesize real-time ultrasound images from CT scans using deep learning (U-Net/Pix2Pix).

---

## 🛠️ Setup & Training on Other Devices

### 1. Clone the Repository
Clone the repository to your training machine:
```bash
git clone https://github.com/Vedansh076/ct-us-robotic-scanning.git
cd ct-us-robotic-scanning
```

### 2. Copy Patient Data (Large Files)
Copy the raw patient volume directories (`TCGA-QQ-A8VG`, `TCGA-QQ-ASV2`, `TCGA-QQ-ASVC`) containing the large binary NIfTI files (`CT.nii`, `SimUS.nii`) and meshes (`patient_skin.obj`) directly into the root of this cloned directory. These files are excluded from git.

### 3. Generate Training Slices
Generate the 2D training dataset from the 3D patient volumes:
```bash
python gen_data.py --subject TCGA-QQ-A8VG
python gen_data.py --subject TCGA-QQ-ASV2
python gen_data.py --subject TCGA-QQ-ASVC
```

### 4. Train the Model
Start U-Net training with the soft-tissue window bounds:
```bash
python model/train.py --data_root dataset --output_dir model/runs/exp1 --epochs 30 --batch_size 8 --lr 2e-4
```

---

## 🎮 Running the Simulation

Start the interactive PyBullet robot scanning demo:
```bash
python live_unet_demo.py --subject TCGA-QQ-A8VG
```

### Keyboard Controls:
* **Arrow Keys:** Translate probe (Forward/Backward/Left/Right).
* **R / F:** Translate probe (Up/Down).
* **J / L:** Roll rotation (Left/Right).
* **I / K:** Pitch rotation (Forward/Backward).
* **U / O:** Yaw rotation (Clockwise/Counter-Clockwise).
* **X / Y / Z:** Lock movement along X, Y, or Z axis (Z lock also disables surface snapping).
* **[ / ]:** Decrease / Increase translational scanning speed.
* **T:** Toggle between In-Plane (longitudinal) and Out-of-Plane (transverse) clinical B-mode scan views.
* **M:** Toggle between Auto sweep and Manual scanning modes.
* **P:** Toggle skin surface snapping.
* **S:** Save debug snapshot images and stats.
* **ESC / Q:** Quit the simulation.

---

## 🔄 Keeping the Repository Up-to-Date

To synchronize changes across your development and training devices, use standard Git commands:

### Pulling Updates (on your training device):
Before starting a new session or training run, download any updates pushed from other devices:
```bash
git pull origin main
```

### Pushing Updates (on your development device):
After modifying code or checking in new settings:
1. Stage the files (large data and caches are auto-ignored by `.gitignore`):
   ```bash
   git add .
   ```
2. Commit your changes:
   ```bash
   git commit -m "Describe your changes"
   ```
3. Push to GitHub:
   ```bash
   git push origin main
   ```

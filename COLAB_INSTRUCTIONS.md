# Training A2C on Google Colab

This guide explains how to run the A2C training script (`train_a2c.py`) on Google Colab for a fast, dedicated run without CPU/RAM resource conflicts.

---

## Step 1 — Upload the Zip file to Google Drive
1. Locate the file `ct-us-colab.zip` in your local project workspace:
   `e:\DELL\internship\Data\HumanSubjects\HumanSubjects\ct_us\ct-us-colab.zip` (Size: ~177 MB)
2. Upload this ZIP file directly to the root of your **Google Drive** (`My Drive`).

---

## Step 2 — Open a new Google Colab Notebook
1. Open [Google Colab](https://colab.research.google.com/).
2. Click **New Notebook**.
3. (Optional) Set the runtime type to GPU for fast PyTorch tensor operations:
   * **Runtime** ➔ **Change runtime type** ➔ Select **T4 GPU** ➔ **Save**.

---

## Step 3 — Add and Run the Colab Cells

Create 5 code cells in your Colab notebook and run them in order:

### Cell 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
*(Follow the link and authorize Colab to access your Google Drive).*

### Cell 2: Copy and Extract Project Code
```python
# Create workspace directory and extract code
!mkdir -p /content/ct_us
!unzip -q "/content/drive/MyDrive/ct-us-colab.zip" -d /content/ct_us
%cd /content/ct_us
```

### Cell 3: Install Dependencies
```python
# Install pybullet, gymnasium, stable-baselines3 and other libraries
!pip install gymnasium stable-baselines3[extra] nibabel trimesh opencv-python --quiet
```

### Cell 4: Launch Training (Runs at ~70+ FPS)
```python
# Run Strategy 2 (skip_unet=True by default) A2C training
!python train_a2c.py --timesteps 500000 --save-freq 50000
```
*Because the U-Net is skipped, this will run at maximum speed and complete in about 1.9 hours.*

### Cell 5: Save Checkpoints back to Google Drive
```python
# Copy checkpoints and tensorboard logs back to your Google Drive
!cp -r /content/ct_us/a2c_checkpoints "/content/drive/MyDrive/a2c_checkpoints_colab"
!cp -r /content/ct_us/a2c_tensorboard "/content/drive/MyDrive/a2c_tensorboard_colab"
print("Checkpoints and logs successfully backed up to Google Drive!")
```

---

## Step 4 — Download Checkpoints to Laptop
Once training finishes, you will find the `a2c_checkpoints_colab/` folder in your Google Drive. 
Download the `best_model.zip` file to your laptop, place it in the `ct_us` workspace, and you can run it in the GUI demo!

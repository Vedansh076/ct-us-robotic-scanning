import numpy as np
import matplotlib.pyplot as plt

ct_slice = np.load("dataset/ct/tcga-qq-asvc_02021.npy")
sim_slice = np.load("dataset/simus/tcga-qq-asvc_02021.npy")

fig, ax = plt.subplots(1,2, figsize=(10,5))

ax[0].imshow(ct_slice, cmap="gray")
ax[0].set_title("CT")

ax[1].imshow(sim_slice, cmap="gray")
ax[1].set_title("SimUS")

plt.show()
# import os

# mins = []
# maxs = []

# for f in os.listdir("dataset/ct")[:100]:
#     img = np.load(f"dataset/ct/{f}")
#     mins.append(img.min())
#     maxs.append(img.max())

# print("CT min:", min(mins))
# print("CT max:", max(maxs))
# mins =[]
# maxs=[]
# for f in os.listdir("dataset/simus")[:100]:
#     img = np.load(f"dataset/simus/{f}")
#     mins.append(img.min())
#     maxs.append(img.max())

# print("simus min:", min(mins))
# print("simus max:", max(maxs))

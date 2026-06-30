# Paper Analysis: SonoGym (NeurIPS 2025)
## Reference Document for AI Agents

This document contains a structured technical analysis of the paper: *"SonoGym: High Performance Simulation for Challenging Surgical Tasks with Robotic Ultrasound"* (NeurIPS 2025). It serves as a reference for agents aligning this project's simulation capabilities with the SonoGym benchmark.

---

## 1. Overview & Core Concept
* **Objective:** Establish a high-performance simulation platform for robotic ultrasound (US) control tasks, bridging the gap between simulation-driven reinforcement learning (RL) and real-world clinical execution.
* **Problem Addressed:** Real robotic ultrasound requires complex coordination of force control, acoustic contact, and real-time visual interpretation. Traditional physics-based acoustic simulators (like field II or wave-propagation solvers) are too slow for RL training, while simple geometric ray-casters lack realistic B-mode textures and acoustic artifacts (shadowing, reflection, noise).

---

## 2. Technical Components & Methodology

### A. Generative Ultrasound Simulation (Domain Transfer)
* **Framework:** Pix2Pix (Conditional GAN) translating 2D Computed Tomography (CT) slices to ultrasound images.
* **Loss Functions:** Combination of L1 reconstruction loss (to enforce spatial alignment) and Adversarial GAN loss (to generate realistic speckle patterns and textures).
* **Training Data:** A paired ex-vivo CT-to-Ultrasound dataset collected from 7 spine specimens (Ultrabones100k).
* **Intensity Histogram Matching:** To handle unseen CT volumes (e.g., from new patient cohorts or scanner manufacturers), they apply **intensity histogram matching** between the test CT slice and the training CT images. This maps out-of-distribution Hounsfield Units (HU) into the training distribution, dramatically reducing domain transfer errors.

### B. Robotic Environment & Physics (PyBullet/IsaacLab)
* **Probe Navigation:** 3D collision mechanics simulating probe contact with soft tissue (torso mesh).
* **Force Correction:** Contact force feedback to ensure the probe remains in contact with the skin surface without causing tissue damage.
* **Slicing Mechanics:** Dynamic 2D slice extraction at the intersection point of the probe tip. The slice plane is aligned with the probe's orientation vectors, representing a clinical B-mode scan.

### C. Benchmark Control Tasks (Markov Decision Processes)
The paper defines three primary tasks structured as OpenAI Gym/Gymnasium environments:

1. **Ultrasound Navigation (3 DoF):**
   * *Action Space:* Delta movements along the surface $(\Delta x, \Delta y, \Delta \alpha)$ to move the probe.
   * *State Space:* Current US image observation and relative position.
   * *Goal:* Guide the probe along the patient skin to find and center a specific target anatomy (e.g., a specific lumbar vertebra vertebra).
2. **Bone Surface Reconstruction (4 DoF):**
   * *Action Space:* Coordinate movements of the probe.
   * *Goal:* Maximize the reconstruction coverage of the underlying bone surface. Formulated as a Submodular Markov Decision Process (MDP).
3. **Ultrasound-guided Surgery (6 DoF):**
   * *Action / State Space:* Full DoF drill manipulation guided by live US tracking.
   * *Safety Constraints:* State-wise safety constraints to prevent drilling critical structures (e.g., nerves, vessels), solved using Safe PPO.

### D. Benchmark Baselines
* **Reinforcement Learning:** PPO (Proximal Policy Optimization) and A2C.
* **Imitation Learning:** ACT (Action Chunking with Transformers) and Diffusion Policy.

---

## 3. Direct Implications for our Project

To align our `ct_us` project with the SonoGym benchmark, we must focus on:
1. **Clinical B-Mode Slicing (Stage 1):** Slicing must be orthogonal to the skin face, using registration-aware transforms (inverse affine, mesh scaling, and body orientation) to ensure sub-voxel accuracy. (Implemented).
2. **Histogram Matching (Stage 2):** Preprocess the extracted CT slice to match the training intensity histogram, reducing domain shift. (Implemented).
3. **Gym Environment Wrapping (Stage 3):** Wrap the PyBullet robot controller, contact force feedback, and slice extraction in a Gymnasium interface to define states, actions, and rewards matching the Navigation task.

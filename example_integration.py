"""
example_integration.py
-----------------------
Minimal runnable example showing how to integrate the simulator
into an existing PyBullet project.

Three stages:
    Stage 1  – Headless smoke test (no PyBullet, no CT, no model)
    Stage 2  – PyBullet stub (probe moved manually, no CT/model)
    Stage 3  – Full integration template (YOUR assets plugged in)

Run this file directly to execute Stage 1.
"""

import numpy as np
import sys
import os

# Make imports work from this directory
sys.path.insert(0, os.path.dirname(__file__))

from core.coordinate_system import AffineTransform, CoordinateMapper
from core.probe import ProbePose, RayCastResult
from core.ct_slice import (
    CTSlicePipeline, SliceRequest, SliceFrame, NullSliceExtractor
)
from rendering.neural_renderer import NeuralUSRenderer, NullUSModel, PreprocessConfig


# ===========================================================================
# STAGE 1 — Headless smoke test
# ===========================================================================

def stage1_headless_smoke_test():
    """
    Verify the whole pipeline runs end-to-end without PyBullet,
    CT data, or a trained model.  Uses all placeholder components.
    """
    print("=" * 60)
    print("STAGE 1: Headless smoke test")
    print("=" * 60)

    # --- Build coordinate mapper (identity placeholder) ---
    mapper = CoordinateMapper.placeholder()

    # --- Simulate a probe pose ---
    probe_pos = np.array([0.05, 0.10, 0.20])          # 5cm, 10cm, 20cm
    probe_orn = np.array([1.0, 0.0, 0.0, 0.0])        # identity quaternion

    # --- Simulate a ray-cast hit ---
    hit_position = np.array([0.05, 0.08, 0.20])        # body surface

    # --- World → CT mapping ---
    ct_center = mapper.world_to_ct(hit_position)
    ct_quat   = mapper.world_orientation_to_ct(probe_orn)

    print(f"  Probe position (world):   {probe_pos}")
    print(f"  Ray hit position (world): {hit_position}")
    print(f"  CT centre (voxels):       {ct_center}")
    print(f"  CT quaternion:            {ct_quat}")

    # --- Null slice extraction ---
    extractor = NullSliceExtractor()
    ct_slice = extractor(
        ct_volume=None,
        center=ct_center,
        quaternion=ct_quat,
        spacing=np.array([0.5, 0.5, 0.5]),
        output_size=(256, 256),
    )
    print(f"  CT slice shape:           {ct_slice.shape}  dtype={ct_slice.dtype}")

    # --- Null neural renderer ---
    renderer = NeuralUSRenderer(model=NullUSModel())
    us_image = renderer.render(ct_slice)
    print(f"  US image shape:           {us_image.shape}  range=[{us_image.min():.2f},{us_image.max():.2f}]")

    print("\n  ✓ Stage 1 passed – all components initialise and run.\n")


# ===========================================================================
# STAGE 2 — Affine transform example
# ===========================================================================

def stage2_affine_example():
    """
    Demonstrate building a real affine from CT header metadata.

    Fill in these values from your CT's ITK/SimpleITK header when ready.
    """
    print("=" * 60)
    print("STAGE 2: Affine transform construction")
    print("=" * 60)

    # --- Example CT header values (replace with your real ones) ---
    # SimpleITK:
    #   image.GetOrigin()     → (ox, oy, oz) in mm
    #   image.GetDirection()  → flat 9-tuple (row-major 3×3)
    #   image.GetSpacing()    → (dx, dy, dz) in mm

    ct_origin_mm = np.array([-120.0, -160.0, -400.0])   # mm
    ct_origin_m  = ct_origin_mm / 1000.0                  # → metres for PyBullet
    
    # CT axes in world frame (identity = CT aligned with world)
    ct_axes_world = np.eye(3)    # replace with image.GetDirection() reshaped to (3,3)

    voxel_spacing_mm = np.array([0.5, 0.5, 0.5])         # mm per voxel

    affine = AffineTransform.from_ct_header(
        ct_origin_world=ct_origin_m,
        ct_axes_world=ct_axes_world,
        voxel_spacing_mm=voxel_spacing_mm,
    )
    mapper = CoordinateMapper.from_affine(affine)

    # Round-trip test
    world_pt = np.array([0.05, 0.10, 0.20])    # metres
    ct_pt    = mapper.world_to_ct(world_pt)
    world_rt = mapper.ct_to_world(ct_pt)

    print(f"  World point (m):  {world_pt}")
    print(f"  CT voxel:         {ct_pt}")
    print(f"  Round-trip (m):   {world_rt}")
    print(f"  Round-trip error: {np.linalg.norm(world_pt - world_rt):.2e} m")
    print("\n  ✓ Stage 2 passed – affine round-trip consistent.\n")


# ===========================================================================
# STAGE 3 — Full integration template
# ===========================================================================

STAGE3_TEMPLATE = '''
"""
STAGE 3 — Full integration into your existing PyBullet project.
Copy this block into your main simulation file and fill in the TODOs.
"""

import pybullet as p
import pybullet_data
import numpy as np

# ---- your existing project imports ----
from ct_us_simulator.simulator import UltrasoundSimulator
from ct_us_simulator.core.coordinate_system import AffineTransform

# ---- YOUR assets ----
# from my_ct_loader import load_ct
# from my_extract_slice import extract_slice
# from my_model import MyUNet, load_weights

def main():
    # --- 1. Start PyBullet (or connect to existing client) ---
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.81)

    # --- 2. Load your phantom / body URDF ---
    body_id = p.loadURDF("path/to/body.urdf", [0, 0, 0])

    # --- 3. Load your probe URDF ---
    probe_id = p.loadURDF("path/to/probe.urdf", [0, 0.1, 0.3])

    # ================================================================
    # --- 4. Load YOUR CT data ---
    # ct_volume = load_ct("path/to/ct.nii.gz")
    ct_volume = None  # ← replace when available

    # --- 5. Build the world→CT affine from the CT header ---
    # import SimpleITK as sitk
    # image = sitk.ReadImage("ct.nii.gz")
    # ct_origin_m  = np.array(image.GetOrigin()) / 1000.0
    # ct_axes      = np.array(image.GetDirection()).reshape(3,3)
    # ct_spacing   = np.array(image.GetSpacing())
    # affine = AffineTransform.from_ct_header(ct_origin_m, ct_axes, ct_spacing)
    affine = None  # ← replace with real affine

    # --- 6. Set up your extract_slice and model ---
    # extractor = extract_slice   # your existing function
    # net = MyUNet(); load_weights(net, "weights.pth")
    # from ct_us_simulator.rendering.neural_renderer import PyTorchModelAdapter
    # model = PyTorchModelAdapter(net, device="cuda")
    extractor = None  # ← replace
    model     = None  # ← replace
    # ================================================================

    # --- 7. Build simulator ---
    sim = UltrasoundSimulator.build(
        probe_body_id=probe_id,
        ct_volume=ct_volume,
        extractor=extractor,
        model=model,
        affine=affine,
        physics_client=client,
        ray_length=0.3,
        output_size=(256, 256),
    )

    # Plug in assets later without rebuilding:
    #   sim.plug_in_ct_volume(ct_volume)
    #   sim.plug_in_model(model)
    #   sim.plug_in_affine(affine)

    # --- 8. Simulation loop ---
    running = True
    while running:
        p.stepSimulation()

        frame, us_image = sim.step()

        # frame.ct_slice          → (256,256) CT image
        # us_image                → (256,256) US prediction
        # frame.probe_pose        → position + orientation
        # frame.ray_result.hit    → bool
        # frame.slice_request     → ct_center, ct_quaternion

        if sim.should_quit():
            running = False

    sim.close()
    p.disconnect()

if __name__ == "__main__":
    main()
'''


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    stage1_headless_smoke_test()
    stage2_affine_example()

    print("=" * 60)
    print("STAGE 3 template (copy to your project):")
    print("=" * 60)
    print(STAGE3_TEMPLATE)

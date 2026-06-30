---
name: slicer_mesh_registration
description: Guides precise registration of 3D Slicer OBJ meshes to CT volume coordinates in PyBullet simulations, compensating for LPS/RAS conversions, centering offsets, and non-isotropic mesh scaling.
---

# Slicer Mesh Registration and CT Slice Alignment

This skill outlines the process and mathematical transformations required to register a 3D surface mesh exported from 3D Slicer to its corresponding CT volume, and to perform registration-aware 2D slice extraction during robotic scanning in PyBullet.

---

## 1. Coordinate Space Alignments

There are three primary coordinate spaces involved:
1. **NIfTI Voxel Space:** Integer indices $(i, j, k)$ referencing the CT volume.
2. **Scanner Physical Space (RAS):** Millimeters $(x_{\text{mm}}, y_{\text{mm}}, z_{\text{mm}})$ relative to the scanner origin. Right-Anterior-Superior (RAS) is used by NIfTI affines.
3. **Slicer Export Space (LPS):** Millimeters relative to the Slicer origin. 3D Slicer OBJ exports typically use Left-Posterior-Superior (LPS) orientation.
4. **PyBullet World Space:** Meters $(x_{\text{world}}, y_{\text{world}}, z_{\text{world}})$.

### LPS to RAS Conversion
To align Slicer OBJ coordinates with NIfTI's RAS space:
$$x_{\text{RAS}} = -x_{\text{LPS}}$$
$$y_{\text{RAS}} = -y_{\text{LPS}}$$
$$z_{\text{RAS}} = z_{\text{LPS}}$$

---

## 2. Mesh Preprocessing (`prepare_slicer_mesh.py`)

To place the mesh in PyBullet, we scale it to meters, center it horizontally, and place the lowest Z coordinate at $z=0$.

### Steps:
1. **Load NIfTI Affine:** Extract `affine` ($M$) and its inverse `inv_affine` ($M^{-1}$).
2. **Convert LPS to RAS:** Negate X and Y coordinates of OBJ vertices.
3. **Scale to Meters:** Divide vertices by $1000.0$.
4. **Compute Centering Offset:**
   $$\text{offset} = \begin{bmatrix} x_{\text{center}} \\ y_{\text{center}} \\ z_{\text{min}} \end{bmatrix}$$
5. **Center Vertices:** $P_{\text{centered}} = P_{\text{meters}} - \text{offset}$.
6. **Save Files:**
   * `patient_skin.obj` (centered mesh in meters).
   * `registration_meta.json` containing the affine, inverse affine, shape, voxel spacing, and centering offset.

---

## 3. Mathematical Transforms

### World to Voxel (Find Slice Center Voxel)
Given a raycast hit point $P_{\text{world}}$ on the patient body in PyBullet:
1. **Undo PyBullet Placement:**
   $$P_{\text{mesh}} = R_{\text{body}}^T (P_{\text{world}} - T_{\text{body}})$$
2. **Undo PyBullet Scaling:**
   $$P_{\text{unscaled}} = P_{\text{mesh}} \oslash \text{scale}_{\text{mesh}}$$
3. **Undo Centering & Convert to mm:**
   $$P_{\text{mm}} = (P_{\text{unscaled}} + \text{offset}_{\text{centering}}) \times 1000.0$$
4. **Transform to Voxel via Inverse Affine:**
   $$\begin{bmatrix} i \\ j \\ k \\ 1 \end{bmatrix} = M^{-1} \begin{bmatrix} P_{\text{mm}} \\ 1 \end{bmatrix}$$
5. **Clamp:** Clamp $i, j, k$ to volume shape bounds.

### Direction Vector Transform (Registration-Aware Slicing)
When extracting a 2D slice, the probe's local axes $u_{\text{world}}, v_{\text{world}}$ must be converted to voxel space increments:
1. **Scale step by pixel spacing:**
   $$u_{\text{world\_step}} = u_{\text{world}} \times \frac{\text{pixel\_spacing}}{1000.0}$$
2. **Transform direction through inverse registration chain:**
   $$u_{\text{voxel}} = M^{-1}_{3 \times 3} \left( \left( R_{\text{body}}^T u_{\text{world\_step}} \oslash \text{scale}_{\text{mesh}} \right) \times 1000.0 \right)$$
   $$v_{\text{voxel}} = M^{-1}_{3 \times 3} \left( \left( R_{\text{body}}^T v_{\text{world\_step}} \oslash \text{scale}_{\text{mesh}} \right) \times 1000.0 \right)$$
3. **Grid offsets:** The offsets relative to the slice center are:
   $$\text{offsets}_{\text{voxel}}(x, y) = x \cdot u_{\text{voxel}} + y \cdot v_{\text{voxel}}$$

---

## 4. Verification Checklist

1. **Affine Round-trip Error:** Verify that converting a voxel to world space and back yields a sub-voxel error ($< 10^{-5}$ voxels).
2. **Slice Aspect Ratio:** Ensure that scaling the mesh (e.g., `--mesh-scale 2,4,1`) does not stretch the extracted slice plane. The slice plane must remain isotropic when specified by isotropic `pixel_spacing`.
3. **Boundary Check:** Check if the probe hitting the extreme vertices of the mesh matches the corresponding CT volume boundaries.

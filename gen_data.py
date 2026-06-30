from pathlib import Path
import argparse

import nibabel as nib
import numpy as np
from scipy.spatial.transform import Rotation as R

from extract_slice import extract_slice

np.random.seed(42)
def random_pose(center, translation_range=15):
    offset = np.random.uniform(
        -translation_range,
        translation_range,
        size=3,
    )
    new_center = center + offset

    roll = np.random.uniform(-30, 30)
    pitch = np.random.uniform(-30, 30)
    yaw = np.random.uniform(-180, 180)

    quaternion = R.from_euler(
        "xyz",
        [roll, pitch, yaw],
        degrees=True,
    ).as_quat()

    return new_center, quaternion


def load_subject(subject_dir):
    subject_dir = Path(subject_dir)
    ct_path = subject_dir / "CT.nii"
    simus_path = subject_dir / "SimUS.nii"

    if not ct_path.exists() or not simus_path.exists():
        raise FileNotFoundError(f"Missing CT.nii or SimUS.nii in {subject_dir}")

    ct_img = nib.load(ct_path)
    simus_img = nib.load(simus_path)

    ct_volume = ct_img.get_fdata()
    simus_volume = simus_img.get_fdata()

    spacing = np.array(ct_img.header.get_zooms()[:3], dtype=np.float32)
    center = (np.array(ct_volume.shape, dtype=np.float32) - 1) / 2

    return ct_volume, simus_volume, spacing, center


def save_sample(output_dir, subject_id, sample_index, ct_slice, simus_slice, center, quaternion):
    filename = f"{subject_id}_{sample_index:05d}.npy"

    np.save(output_dir / "ct" / filename, ct_slice.astype(np.float32))
    np.save(output_dir / "simus" / filename, simus_slice.astype(np.float32))
    np.save(
        output_dir / "poses" / filename,
        {
            "subject_id": subject_id,
            "center": np.asarray(center, dtype=np.float32),
            "quaternion": np.asarray(quaternion, dtype=np.float32),
        },
    )


def generate_subject_dataset(
    subject_dir,
    output_dir,
    samples_per_subject=3000,
    translation_range=15,
    size=256,
    pixel_spacing=0.35,
    empty_std_threshold=5,
    max_attempts_multiplier=20,
):
    subject_dir = Path(subject_dir)
    output_dir = Path(output_dir)
    subject_id = subject_dir.name.lower()

    try:
        ct_volume, simus_volume, spacing, center = load_subject(subject_dir)
    except Exception as exc:
        print(f"[SKIP] {subject_dir.name}: {exc}")
        return 0

    saved = 0
    attempts = 0
    max_attempts = samples_per_subject * max_attempts_multiplier

    print(f"[START] {subject_id}: generating {samples_per_subject} samples")

    while saved < samples_per_subject and attempts < max_attempts:
        attempts += 1

        try:
            pose_center, quaternion = random_pose(center, translation_range=translation_range)

            ct_slice = extract_slice(
                ct_volume,
                center=pose_center,
                quaternion=quaternion,
                spacing=spacing,
                size=size,
                pixel_spacing=pixel_spacing,
            )

            simus_slice = extract_slice(
                simus_volume,
                center=pose_center,
                quaternion=quaternion,
                spacing=spacing,
                size=size,
                pixel_spacing=pixel_spacing,
            )

            if np.std(simus_slice) < empty_std_threshold:
                continue

            save_sample(
                output_dir=output_dir,
                subject_id=subject_id,
                sample_index=saved,
                ct_slice=ct_slice,
                simus_slice=simus_slice,
                center=pose_center,
                quaternion=quaternion,
            )

            saved += 1

            if saved % 100 == 0:
                print(f"[{subject_id}] Saved {saved}/{samples_per_subject}")

        except Exception as exc:
            print(f"[WARN] {subject_id}: failed attempt {attempts}: {exc}")
            continue

    if saved < samples_per_subject:
        print(
            f"[WARN] {subject_id}: saved {saved}/{samples_per_subject} "
            f"after {attempts} attempts"
        )
    else:
        print(f"[DONE] {subject_id}: saved {saved} samples")

    return saved


def find_subject_dirs(input_root):
    input_root = Path(input_root)

    if (input_root / "CT.nii").exists() and (input_root / "SimUS.nii").exists():
        return [input_root]

    subject_dirs = []
    for path in sorted(input_root.iterdir()):
        if path.is_dir() and (path / "CT.nii").exists() and (path / "SimUS.nii").exists():
            subject_dirs.append(path)

    return subject_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Generate paired oblique CT and SimUS slice datasets."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path.cwd(),
        help="Folder containing subject folders, or one subject folder with CT.nii and SimUS.nii.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset"),
        help="Output dataset folder.",
    )
    parser.add_argument(
        "--samples-per-subject",
        type=int,
        default=3000,
        help="Number of valid samples to save per subject.",
    )
    parser.add_argument(
        "--translation-range",
        type=float,
        default=15,
        help="Random translation range in voxels.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Output slice size in pixels.",
    )
    parser.add_argument(
        "--pixel-spacing",
        type=float,
        default=0.35,
        help="Output slice pixel spacing in mm.",
    )
    parser.add_argument(
        "--empty-std-threshold",
        type=float,
        default=5,
        help="Reject SimUS slices with standard deviation below this value.",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    (output_dir / "ct").mkdir(parents=True, exist_ok=True)
    (output_dir / "simus").mkdir(parents=True, exist_ok=True)
    (output_dir / "poses").mkdir(parents=True, exist_ok=True)

    subject_dirs = find_subject_dirs(args.input_root)
    if not subject_dirs:
        print(f"No valid subject folders found in {args.input_root}")
        return

    total_saved = 0
    for subject_dir in subject_dirs:
        try:
            total_saved += generate_subject_dataset(
                subject_dir=subject_dir,
                output_dir=output_dir,
                samples_per_subject=args.samples_per_subject,
                translation_range=args.translation_range,
                size=args.size,
                pixel_spacing=args.pixel_spacing,
                empty_std_threshold=args.empty_std_threshold,
            )
        except Exception as exc:
            print(f"[SKIP] {subject_dir.name}: unexpected error: {exc}")
            continue

    print(f"Dataset generation complete. Saved {total_saved} samples total.")


if __name__ == "__main__":
    main()

import os
import pydicom

dicom_dir = "data/Cavalcanti/computed_tomography/URS01_CT_raw/URS1_CT_raw/CT_Seq._7_LWS SN140_IQ75 0,60 Br60 Q3 Torso/"
if not os.path.exists(dicom_dir):
    # Try finding the correct path recursively
    print(f"Path {dicom_dir} does not exist. Searching...")
    for root, dirs, files in os.walk("data/Cavalcanti/computed_tomography/"):
        if "Torso" in root:
            dicom_dir = root
            print(f"Found: {dicom_dir}")
            break

if os.path.exists(dicom_dir):
    files = [f for f in sorted(os.listdir(dicom_dir)) if os.path.isfile(os.path.join(dicom_dir, f))]
    print(f"Files found in {dicom_dir}: {len(files)}")
    if files:
        first_file = os.path.join(dicom_dir, files[0])
        print(f"Inspecting first file: {first_file}")
        try:
            ds = pydicom.dcmread(first_file, force=True)
            print("Successfully read file with force=True")
            print("Has ImagePositionPatient:", hasattr(ds, "ImagePositionPatient"))
            print("Has pixel_array:", hasattr(ds, "pixel_array"))
            print("Attributes present:", list(ds.keys())[:15])
            if hasattr(ds, "ImagePositionPatient"):
                print("ImagePositionPatient value:", ds.ImagePositionPatient)
            if hasattr(ds, "pixel_array"):
                print("pixel_array shape:", ds.pixel_array.shape)
        except Exception as e:
            print(f"Error reading file: {e}")
else:
    print("Could not find any directory containing CT data.")

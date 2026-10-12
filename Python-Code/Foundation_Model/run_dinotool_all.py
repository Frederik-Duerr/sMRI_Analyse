
from pathlib import Path
import subprocess
import sys

root = Path(
    r"C:\Users\Johan\OneDrive - ucp.pt\Desktop\Privat\Portugal"
    r"\UCP\Semester 3\Biomedical Project\MRI Project\Pre_processed"
)

output_root = Path(
    r"C:\Users\Johan\OneDrive - ucp.pt\Desktop\Privat\Portugal"
    r"\UCP\Semester 3\Biomedical Project\MRI Project\Features"
    r"\dinov2_features"
)

extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Find patient directories containing images directly.
patient_dirs = sorted(
    folder for folder in root.rglob("*")
    if folder.is_dir()
    and any(
        file.is_file() and file.suffix.lower() in extensions
        for file in folder.iterdir()
    )
)

print(f"Found {len(patient_dirs)} image directories.")

output_root.mkdir(parents=True, exist_ok=True)

for i, folder in enumerate(patient_dirs, start=1):
    relative = folder.relative_to(root)
    output_name = "_".join(relative.parts)

    output_base = output_root / output_name
    output_file = Path(str(output_base) + ".parquet")

    # Skip completed patients when rerunning the script.
    if output_file.exists():
        print(f"[{i}/{len(patient_dirs)}] Skipping {relative}")
        continue
   #  r"C:\Users\Johan\anaconda3\envs\dinotool\Scripts\dinotool.exe"
  
    command = [
        r"C:\Users\Johan\anaconda3\envs\dinotool\Scripts\dinotool.exe",
        str(folder),
        "-o", str(output_base),
        "--model-name", "dinov2_vitb14",
        "--save-features", "frame",
        "--input-size", "224", "224",
        "--batch-size", "10",
        "--no-vis",
    ]
    print(f"\n[{i}/{len(patient_dirs)}] Processing {relative}")

    subprocess.run(command, check=True)

print("\nAll patient directories processed.")
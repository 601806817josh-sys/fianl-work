from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.datasets.ct_dataset import CTDataset


def main():
    data_dir = Path("data/raw")
    print(f"Data directory exists: {data_dir.exists()} ({data_dir})")

    nii_files = list(data_dir.rglob("*.nii")) if data_dir.exists() else []
    nii_gz_files = list(data_dir.rglob("*.nii.gz")) if data_dir.exists() else []
    all_files = sorted(nii_files + nii_gz_files)

    if not data_dir.exists() or len(all_files) == 0:
        print("No CT files found. Expected .nii or .nii.gz files in data/raw. Exiting.")
        return

    dataset = CTDataset(data_dir)
    print(f"Dataset length: {len(dataset)}")

    sample = dataset[0]
    print(f"First sample tensor shape: {sample.shape}")
    print(f"First sample tensor dtype: {sample.dtype}")


if __name__ == "__main__":
    main()

from pathlib import Path

import kagglehub
import shutil

from dataset_sources import ALL


def copy_dataset_to_raw(dataset_name: str, dataset_id: str,extra_path:str):
    root_project = Path(__file__).resolve().parents[1]
    download_path = Path(kagglehub.dataset_download(dataset_id))
    target_path = ( root_project / "data" / "raw" / dataset_name)
    target_path.mkdir(parents=True,exist_ok=True)

    if target_path.exists():
        shutil.rmtree(target_path)
    final_path = download_path / extra_path
    print(final_path)
    shutil.copytree(final_path,target_path)

    print(f"Copied {dataset_name} to {target_path}")

def main():
    raw_data_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    for source in ALL:
        if source._is_downloaded(raw_data_dir):
            print(f"Skipping {source.NAME}: already downloaded")
            continue
        copy_dataset_to_raw(source.NAME, source.KAGGLE_ID, source.KAGGLE_SUBPATH)

if __name__ == "__main__":
    main()
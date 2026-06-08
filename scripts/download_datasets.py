from pathlib import Path

import kagglehub
import shutil

DATASETS = {
    "trashnet":{
        "id": "feyzazkefe/trashnet",
        "path" : "dataset-resized"
    },
    "household_garbage":{
        "id": "alistairking/recyclable-and-household-waste-classification",
        "path" : "images/images"
    },
    "garbage_dataset":{
        "id": "sumn2u/garbage-classification-v2",
        "path" : "original"
    },
    "tricascade": {
        "id": "salam035/waste-datasets",
        "path": "Dataset_ThirdStage/Dataset"
    }
}


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
    for ds,ds_info in DATASETS.items():
        copy_dataset_to_raw(ds,ds_info["id"],ds_info["path"])

if __name__ == "__main__":
    main()
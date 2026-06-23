from pathlib import Path
from PIL import Image
import shutil

LABELS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

for label in LABELS:
    project_root = Path(__file__).resolve().parents[1]
    raw_data_dir = project_root / "data" / "raw" / "custom" / "Trash"
    label_dir = raw_data_dir / label

    if not label_dir.exists():
        print(f"Skipping missing folder: {label_dir}")
        continue

    image_paths = [
        path for path in label_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]

    image_paths = sorted(image_paths)

    temp_dir = label_dir / "_renamed_temp"
    temp_dir.mkdir(exist_ok=True)

    counter = 1

    for image_path in image_paths:
        new_name = f"{label}_{counter:04d}.jpg"
        new_path = temp_dir / new_name

        try:
            # Open image and convert to RGB JPEG
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img.save(new_path, "JPEG", quality=95)

            counter += 1

        except Exception as error:
            print(f"Could not process {image_path}: {error}")

    # Move original files to backup folder
    backup_dir = label_dir / "_original_backup"
    backup_dir.mkdir(exist_ok=True)

    for image_path in image_paths:
        shutil.move(str(image_path), str(backup_dir / image_path.name))

    # Move renamed files back into label folder
    for new_image_path in temp_dir.iterdir():
        shutil.move(str(new_image_path), str(label_dir / new_image_path.name))

    temp_dir.rmdir()

    print(f"{label}: renamed {counter - 1} images")

print("Done.")
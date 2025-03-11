import cv2
import shutil
from pathlib import Path

# Define source and destination directories
source = Path("/cluster/home/pettdalh/tdt4265_project/data")
dest = Path("/cluster/home/pettdalh/tdt4265_project/cleaned_data")

# Delete the cleaned_data folder if it exists
if dest.exists() and dest.is_dir():
    shutil.rmtree(dest)

# Copy entire data folder to cleaned_data
shutil.copytree(source, dest)

# Define the images and labels directories in the copied folder
images_dir = [dest / "lidar_data" / "train" / "images", dest / "lidar_data" / "valid" / "images"]
labels_dir = [dest / "lidar_data" / "train" / "labels", dest / "lidar_data" / "valid" / "labels"]

# Iterate through all PNG files in the images folder
for i in range(len(images_dir)):
    for img_path in images_dir[i].glob("*.png"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read {img_path}, deleting image and label...")
            # Delete the image file
            img_path.unlink()
            # Delete the corresponding label file (assumes same filename stem with .txt extension)
            label_file = labels_dir[i] / f"{img_path.stem}.txt"
            if label_file.exists():
                label_file.unlink()


# Delete the test files without corresponding label
images_dir = dest / "lidar_data" / "test" / "images"

for img_path in images_dir.glob("*.png"):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to read {img_path}, deleting image and label...")
        # Delete the image file
        img_path.unlink()



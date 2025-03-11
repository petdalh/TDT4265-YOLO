import albumentations as A
import cv2
import os
import random

# Define paths
image_folder = "/cluster/home/pettdalh/tdt4265_project/cleaned_data/lidar_data/train/images/"
label_folder = image_folder.replace('/images/', '/labels/')

# Get list of image files
image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

# Number of samples to augment
num_samples = 100

# Randomly select samples
random_samples = random.sample(image_files, min(num_samples, len(image_files)))

print("Selected samples:", random_samples)

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=1),
    A.RandomBrightnessContrast(p=0.9),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Process each sample
for img in random_samples:
    # Construct paths
    image_path = os.path.join(image_folder, img)
    label_path = image_path.replace('/images/', '/labels/').replace('.png', '.txt')

    # Read image
    image = cv2.imread(image_path)

    # Read labels
    with open(label_path, 'r') as f:
        label_lines = f.readlines()

    # Skip if no labels
    if not label_lines:
        print(f"Skipping {img} (no labels found)")
        continue

    # Parse labels
    bboxes = []  # stores [x_center, y_center, width, height]
    class_labels = []

    for line in label_lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        bboxes.append([x_center, y_center, width, height])
        class_labels.append(int(class_id))

    # Apply augmentation
    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    augmented_image = augmented["image"]
    augmented_label = augmented["bboxes"]
    augmented_class_labels = augmented['class_labels']

    # Save augmented image and labels
    image_file_name = img.replace('.png', '_augmented.png')
    label_file_name = img.replace('.png', '_augmented.txt')

    # Save augmented image
    cv2.imwrite(os.path.join(image_folder, image_file_name), augmented_image)

    # Save augmented labels
    with open(os.path.join(label_folder, label_file_name), "w") as file:
        for i, [x, y, w, h] in enumerate(augmented_label):
            file.write(f"{int(augmented_class_labels[i])} {x} {y} {w} {h}\n")

    print(f"Saved augmented data: {image_file_name}, {label_file_name}")
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
num_samples = 100  # Adjust as needed
random_samples = random.sample(image_files, min(num_samples, len(image_files)))

print(f"Selected {len(random_samples)} samples for augmentation")

# Define separate augmentation pipelines
augmentation_profiles = {
    "flip": A.Compose([
        A.HorizontalFlip(p=1)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
    
    "color": A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
    
    "weather": A.Compose([
        A.RandomSnow(
            brightness_coeff=1.8,  # Reduced from 2.5 (less brightness distortion)
            snow_point_lower=0.1,  # Reduced from 0.3 (less coverage)
            snow_point_upper=0.2,  # Reduced from 0.5
            p=0.7  # Not applied to every image
        ),
        
        # Subtle fog effect
        A.RandomFog(
            fog_coef_lower=0.1,  # Very light fog
            fog_coef_upper=0.2,  # Max 20% fog intensity
            alpha_coef=0.08,     # Controls how dense the fog appears
            p=0.5               # Only applied half the time
        ),
        
        # Optional: Light rain streaks (very subtle)
        A.RandomRain(
            slant_lower=-5, 
            slant_upper=5,
            drop_length=10,      # Shorter streaks
            drop_width=1,        # Thinner streaks
            drop_color=(200,200,200),  # Lighter color
            blur_value=1,        # Minimal blur
            rain_type='drizzle', # Light rain type
            p=0.3               # Low probability
        )
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
    
    "geometric": A.Compose([
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=1),
        A.Rotate(limit=15, p=0.5)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
    
    "sensor": A.Compose([
        A.GaussianBlur(blur_limit=(3, 5), p=1),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.5)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
}

def load_labels(label_path):
    with open(label_path, 'r') as f:
        label_lines = f.readlines()
    
    bboxes = []
    class_labels = []
    
    for line in label_lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        bboxes.append([x_center, y_center, width, height])
        class_labels.append(int(class_id))
    
    return bboxes, class_labels

def save_augmented_data(image, bboxes, class_labels, original_path, suffix):
    # Generate new filenames
    base_name = os.path.splitext(os.path.basename(original_path))[0]
    image_file = f"{base_name}_{suffix}.png"
    label_file = f"{base_name}_{suffix}.txt"
    
    # Save image
    cv2.imwrite(os.path.join(image_folder, image_file), image)
    
    # Save labels
    with open(os.path.join(label_folder, label_file), 'w') as f:
        for cls, bbox in zip(class_labels, bboxes):
            f.write(f"{cls} {' '.join(map(str, bbox))}\n")
    
    return image_file, label_file

# Process each sample with each augmentation type
for img_file in random_samples:
    # Load original image
    image_path = os.path.join(image_folder, img_file)
    label_path = os.path.join(label_folder, img_file.replace('.png', '.txt'))
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        continue
    
    # Load labels
    try:
        bboxes, class_labels = load_labels(label_path)
    except:
        print(f"Warning: Could not read labels for {img_file}")
        continue
    
    # Apply each augmentation profile
    for aug_name, transform in augmentation_profiles.items():
        try:
            augmented = transform(
                image=image.copy(),  # Work with a copy
                bboxes=bboxes.copy(),
                class_labels=class_labels.copy()
            )
            
            # Save results
            new_img, new_label = save_augmented_data(
                augmented['image'],
                augmented['bboxes'],
                augmented['class_labels'],
                image_path,
                aug_name
            )
            
            print(f"Created {aug_name} variant: {new_img}")
            
        except Exception as e:
            print(f"Error applying {aug_name} to {img_file}: {str(e)}")

print("Augmentation complete!")
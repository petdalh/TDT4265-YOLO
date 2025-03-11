def visualize_sample(img_path, label_path):
    """Visualize an image with its bounding boxes"""
    # Load image
    img = plt.imread(img_path)
    img_height, img_width = img.shape[:2]
    print(img_height, img_width)
    # Plot image
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    
    # Load and plot bounding boxes
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                values = line.strip().split()
                class_id = int(values[0])
                x_center, y_center, width, height = [float(x) for x in values[1:5]]
                
                # Convert YOLO format to pixel coordinates
                img_height, img_width = img.shape[:2]
                x1 = int((x_center - width/2) * img_width)
                y1 = int((y_center - height/2) * img_height)
                x2 = int((x_center + width/2) * img_width)
                y2 = int((y_center + height/2) * img_height)
                
                # Draw rectangle
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='r', linewidth=2)
                plt.gca().add_patch(rect)
                plt.text(x1, y1, 'Pole', color='white', bbox=dict(facecolor='red', alpha=0.5))
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()


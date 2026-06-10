import os
import shutil
import random

# 📂 SOURCE (PlantVillage dataset)
SOURCE_DIR = r"C:\Users\Praveen Reddy\Downloads\archive (1)\PlantVillage"

# 📂 DESTINATION
BASE_DIR = "dataset"

train_dir = os.path.join(BASE_DIR, "train")
test_dir = os.path.join(BASE_DIR, "test")
valid_dir = os.path.join(BASE_DIR, "valid")

# Create folders
for folder in [train_dir, test_dir, valid_dir]:
    os.makedirs(folder, exist_ok=True)

# Split ratio
train_ratio = 0.7
test_ratio = 0.2
valid_ratio = 0.1

# Supported image formats
valid_extensions = ('.jpg', '.jpeg', '.png')

# Loop through each class folder
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    print(f"Processing: {class_name}")

    # Get only image files
    images = [
        img for img in os.listdir(class_path)
        if img.lower().endswith(valid_extensions)
    ]

    random.shuffle(images)

    total = len(images)
    train_split = int(total * train_ratio)
    test_split = int(total * (train_ratio + test_ratio))

    train_imgs = images[:train_split]
    test_imgs = images[train_split:test_split]
    valid_imgs = images[test_split:]

    # Create class folders
    for folder in [train_dir, test_dir, valid_dir]:
        os.makedirs(os.path.join(folder, class_name), exist_ok=True)

    # Copy files safely
    def copy_images(image_list, dest_folder):
        for img in image_list:
            src = os.path.join(class_path, img)
            dst = os.path.join(dest_folder, class_name, img)

            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"Skipping file: {img}")

    copy_images(train_imgs, train_dir)
    copy_images(test_imgs, test_dir)
    copy_images(valid_imgs, valid_dir)

print("✅ Dataset successfully split with ALL classes!")
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def augment_images(input_folder, output_folder, augment_count):
    os.makedirs(output_folder, exist_ok=True)

    # Load images
    images = []
    filenames = []
    for f in os.listdir(input_folder):
        path = os.path.join(input_folder, f)
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
            filenames.append(f)

    print(f"Loaded {len(images)} images from {input_folder}")

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    total_augmented = 0
    idx = 0
    while total_augmented < augment_count:
        img = images[idx % len(images)]
        x = np.expand_dims(img, 0)  # Add batch dimension
        aug_iter = datagen.flow(x, batch_size=1)

        aug_img = next(aug_iter)[0].astype(np.uint8)
        save_path = os.path.join(output_folder, f"aug_{total_augmented}.png")
        cv2.imwrite(save_path, aug_img)

        total_augmented += 1
        idx += 1

    print(f"Saved {total_augmented} augmented images to {output_folder}")

# ---- User Inputs ----
input_folder = r"C:\Users\frhan\Desktop\03_Farhan\Projects\Python\Project_College\Brain_Tumor_Datasets copy\train\no"
output_folder = r"C:\Users\frhan\Desktop\03_Farhan\Projects\Python\Project_College\Brain_Tumor_Datasets copy\train\unlabel"
augment_count = 1274  # Number of augmented images you want

augment_images(input_folder, output_folder, augment_count)
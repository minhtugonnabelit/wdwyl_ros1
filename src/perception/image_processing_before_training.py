from PIL import Image, ImageOps
import imgaug.augmenters as iaa
import os
import numpy as np
from tqdm import tqdm

# Define the directory paths
source_dir = '/home/anh/workspace/test_image_contour/src/data_image/train_resize'
dest_dir = '/home/anh/workspace/test_image_contour/src/data_image/train_flip'

# Create the destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Define standard size (you can adjust it based on your requirements)
standard_size = (640, 480)

# Define a simple augmentation sequence
aug = iaa.Sequential([
    # iaa.Fliplr(0.5),  # horizontal flips with a 50% probability

    iaa.Crop(percent=(0, 0.1)),  # random crops

    # Scale the image to standard size, maintaining the aspect ratio
    # iaa.Resize({"height": standard_size[1], "width": standard_size[0]}),

    # Adjust the brightness of images (50-150% of the original value)
    # iaa.Multiply((0.5, 1.5)),

    # Improve or worsen the contrast of images.
    # iaa.LinearContrast((0.75, 1.5)),

    # Apply gamma contrast
    # iaa.GammaContrast((0.5, 2.0)),

    # Apply Gaussian blur with a sigma of 0 to 3.0
    # iaa.GaussianBlur(sigma=(0, 3.0)),

    # Add Gaussian noise to images.
    # iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
])

# Process and augment images
for filename in tqdm(os.listdir(source_dir)):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add other file types if needed
        # Open and augment the image
        with Image.open(os.path.join(source_dir, filename)) as img:
            img_aug = aug.augment_image(np.array(img))  # Convert PIL Image to numpy array for augmentation
            # Convert back to PIL Image and save
            img_processed = Image.fromarray(img_aug)
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_flip{ext}"
            img_processed.save(os.path.join(dest_dir, new_filename))

            # ratio = min(standard_size[0] / img.size[0], standard_size[1] / img.size[1])
            # new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))

            # # Resize the image with the new size
            # img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
            

            # img_new = Image.new("RGB", standard_size, (0, 0, 0))
            # img_new.paste(img_resized, ((standard_size[0] - new_size[0]) // 2,
            #                             (standard_size[1] - new_size[1]) // 2))
            
            # name, ext = os.path.splitext(filename)
            # new_filename = f"{name}_resize{ext}"
            # img_new.save(os.path.join(dest_dir, new_filename))
    else:
        continue

print("Image preprocessing and augmentation completed.")

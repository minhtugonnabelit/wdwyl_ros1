import numpy as np
import cv2
import os
from tqdm import tqdm  # Optional, provides a progress bar

def compute_stats(data_dir):
    sum_pixels = np.zeros(3)  # Sum of pixel values for each channel (RGB)
    sum_sq_pixels = np.zeros(3)  # Sum of squares of pixel values for each channel (RGB)
    count = 0  # Total number of pixels accumulated

    # Walk through the directory
    for root, dirs, files in os.walk(data_dir):
        for file in tqdm(files):  # tqdm is optional but helps visualize progress
            if file.endswith(".jpg"):
                path = os.path.join(root, file)
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
                    img = img / 255.0  # Normalize to [0, 1]

                    sum_pixels += img.sum(axis=(0, 1))
                    sum_sq_pixels += (img ** 2).sum(axis=(0, 1))
                    count += img.shape[0] * img.shape[1]

    mean_rgb = sum_pixels / count
    std_rgb = np.sqrt((sum_sq_pixels / count) - (mean_rgb ** 2))

    return mean_rgb, std_rgb

# Specify the directory containing the training data
data_directory = 'data_dir/train'
mean_rgb, std_rgb = compute_stats(data_directory)
print("Mean of RGB channels:", mean_rgb)
print("Standard deviation of RGB channels:", std_rgb)

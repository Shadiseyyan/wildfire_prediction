import numpy as np
import os

print("--- DIAGNOSTIC SCRIPT STARTING ---")

# 1. Check exactly where Python is looking
current_folder = os.getcwd()
print(f"Current working directory: {current_folder}")

target_path = os.path.join("trainandtar", "train_target.npy")
print(f"Looking for dataset at: {target_path}")

# 2. Verify the file actually exists before trying to load it
if not os.path.exists(target_path):
    print("❌ ERROR: File not found! Make sure the 'trainandtar' folder is in the current directory.")
else:
    print("✅ File found! Loading data...")
    
    # 3. Load and inspect the data
    Y = np.load(target_path)
    
    print(f"Shape of dataset: {Y.shape}")
    print(f"Maximum value in dataset: {np.max(Y)}")
    print(f"Minimum value in dataset: {np.min(Y)}")
    
    # Count the fire pixels
    total_pixels = Y.size
    fire_pixels = np.sum(Y > 0.5)
    
    print(f"Total pixels: {total_pixels:,}")
    print(f"Pixels categorized as 'Fire' (> 0.5): {fire_pixels:,}")

print("--- DIAGNOSTIC SCRIPT FINISHED ---")
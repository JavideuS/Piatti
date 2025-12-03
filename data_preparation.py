import kagglehub
import os
import shutil
import random

def split_dataset(source_dir, train_ratio=0.8, train_dir="data/train", val_dir="data/val", copy=True):
    categories = [d for d in os.listdir(source_dir)
              if os.path.isdir(os.path.join(source_dir, d))]

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for category in categories:
        category_path = os.path.join(source_dir, category)

        if not os.path.exists(category_path):
            print(f"Warning: {category} folder not found, skipping...")
            continue

        # Get all files in the category folder
        files = [f for f in os.listdir(category_path)
                if os.path.isfile(os.path.join(category_path, f))]

        if len(files) == 0:
            print(f"Warning: {category} folder is empty, skipping...")
            continue

        # Shuffle files randomly
        random.shuffle(files)

        # Calculate split point (90% for train)
        split_idx = int(len(files) * 0.9)
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        # Create category subfolders in train and test
        train_category_dir = os.path.join(train_dir, category)
        test_category_dir = os.path.join(val_dir, category)
        os.makedirs(train_category_dir, exist_ok=True)
        os.makedirs(test_category_dir, exist_ok=True)

        # Move files to train
        for file in train_files:
            src = os.path.join(category_path, file)
            dst = os.path.join(train_category_dir, file)
            if copy:
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)

        # Move files to test
        for file in test_files:
            src = os.path.join(category_path, file)
            dst = os.path.join(test_category_dir, file)
            if copy:
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)

        if not copy:
            # Remove the original category folder if empty
            if not os.listdir(category_path):
                os.rmdir(category_path)

        print(f"{category}: {len(train_files)} files to train, {len(test_files)} files to test")


def download():
    target_dir = "data/raw"

    # Download dataset
    tmp_path = kagglehub.dataset_download("jiayuanchengala/aid-scene-classification-datasets")
    tmp_path = os.path.join(tmp_path, "AID")

    # 3. Remove target dir if it exists (optional)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    # 4. Move instead of copying
    shutil.move(tmp_path, target_dir)

if __name__ == "__main__":
    download()
    split_dataset("data/raw/", train_ratio=0.9, train_dir="data/train", val_dir="data/test")
    # It can also do manual train/val/test validation
    # However we will follow manual separation of test (to keep test set consistent/unseen)
    # And then use skearn sepration for train/val split
    # split_dataset("data/train/", train_dir="data/train/train", val_dir="data/train/val", copy=False)


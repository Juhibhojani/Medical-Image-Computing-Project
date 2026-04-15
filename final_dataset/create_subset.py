import os
import shutil
import random

SEED = 42
random.seed(SEED)

base_dir = './dataset_600'

source_train_dir = os.path.join(base_dir, 'train')
train_50_dir = os.path.join(base_dir, 'train_50')
train_100_dir = os.path.join(base_dir, 'train_100')

# creating new directories
os.makedirs(train_50_dir, exist_ok=True)
os.makedirs(train_100_dir, exist_ok=True)


def create_nested_subsets(src_dir, dir_50, dir_100):
    classes = os.listdir(src_dir)

    for cls in classes:
        class_path = os.path.join(src_dir, cls)

        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        # taking top 100/50 images from this
        if len(images)<100:
            print(f"Warning for {cls}, less than 100")
            
        subset_100 = images[:min(100, len(images))]
        subset_50 = subset_100[:min(50, len(subset_100))]

        # Create folders
        os.makedirs(os.path.join(dir_50, cls), exist_ok=True)
        os.makedirs(os.path.join(dir_100, cls), exist_ok=True)

        # Copy 100 subset
        for img in subset_100:
            shutil.copy2(
                os.path.join(class_path, img),
                os.path.join(dir_100, cls, img)
            )

        # Copy 50 subset (nested)
        for img in subset_50:
            shutil.copy2(
                os.path.join(class_path, img),
                os.path.join(dir_50, cls, img)
            )

        print(f"{cls}: 50 -> {len(subset_50)}, 100 -> {len(subset_100)}")


# Run
create_nested_subsets(source_train_dir, train_50_dir, train_100_dir)
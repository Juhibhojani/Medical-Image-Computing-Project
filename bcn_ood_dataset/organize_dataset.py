import json
import os
import pandas as pd
import shutil
import random
import numpy as np
from sklearn.model_selection import train_test_split

# seeting seed everywhere
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

df = pd.read_csv("./metadata.csv")

df.loc[
    df["diagnosis_2"] == "Benign soft tissue proliferations - Vascular",
    "diagnosis_3"
] = "Benign soft tissue proliferations - Vascular (diagnosis_2)"


train,test = train_test_split(df["lesion_id"].unique(),random_state=42)

test_df = df[df['lesion_id'].isin(set(test))]
train_df = df[df["lesion_id"].isin(set(train))]

mapping = {
    # direct
    "Nevus": "nv",
    "Basal cell carcinoma": "bcc",
    "Dermatofibroma": "df",
    "Solar or actinic keratosis": "akiec",

    # selected
    "Melanoma, NOS": "mel",
    "Seborrheic keratosis": "bkl",
    "Benign soft tissue proliferations - Vascular (diagnosis_2)": "vasc"
}

train_df["dx"] = train_df["diagnosis_3"].map(mapping)

test_df["dx"] = test_df["diagnosis_3"].map(mapping)

train_df = train_df.copy()

train_df = train_df.dropna(subset=["dx"])

test_df = test_df.dropna(subset=["dx"])

# limiting training set
def limit_samples_per_class(df, max_per_class=300):
    result = []
    class_distribution = {}

    for label in sorted(df['dx'].unique()):  
        class_subset = df[df['dx'] == label]

        if len(class_subset) > max_per_class:
            # sampling in a fixed manner
            class_subset = class_subset.sample(
                n=max_per_class,
                random_state=SEED
            )
            class_distribution[label] = 0
        else:
            class_distribution[label] = max_per_class-len(class_subset)
            

        result.append(class_subset)

    return pd.concat(result).reset_index(drop=True),class_distribution

# limiting only training samples
train_df,class_distribution = limit_samples_per_class(train_df, max_per_class=200)

# checking the train distribution
print("Train distribution:\n", train_df['dx'].value_counts())
print("Train size:", len(train_df))
print("Test size:", len(test_df))

# organizing based on format required
source_dir = './ood_dataset'
output_dir = './dataset_200'

# Create folders
for split in ['train', 'test']:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

def organize_images(df, split):
    for _, row in df.iterrows():
        img_name = row['isic_id'] + '.jpg'
        label = row['dx']

        class_dir = os.path.join(output_dir, split, label)
        os.makedirs(class_dir, exist_ok=True)

        src = os.path.join(source_dir, img_name)
        dst = os.path.join(class_dir, img_name)

        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"Missing: {src}")

# Copy images
organize_images(train_df, 'train')
organize_images(test_df, 'test')
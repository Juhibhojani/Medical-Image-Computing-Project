import json
import os
import pandas as pd
import shutil
import random
import numpy as np
from sklearn.model_selection import train_test_split

# following approach as given in the referenced code

# seeting seed everywhere
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# load csv
data_pd = pd.read_csv('/home/micml/25m0764/MIC_project/final_dataset/HAM10000_metadata')

# Count how many images per lesion - we only want to keep one image per lesion, so as not have problem in train and test
df_count = data_pd.groupby('lesion_id').count()

# keeping only one image per lesion
df_count = df_count[df_count['dx'] == 1]
df_count.reset_index(inplace=True)

# marking duplicates
def duplicates(x):
    unique = set(df_count['lesion_id'])
    if x in unique:
        return 'no' 
    else:
        return 'duplicates'

# marking duplicates 
data_pd['is_duplicate'] = data_pd['lesion_id'].apply(duplicates)

df_count = data_pd[data_pd['is_duplicate'] == 'no']

# stratified splitting
train, test_df = train_test_split(df_count, test_size=0.15, stratify=df_count['dx'])

# identifying if it is in train or test 
def identify_trainOrtest(x):
    test_data = set(test_df['image_id'])
    if str(x) in test_data:
        return 'test'
    else:
        return 'train'

# creating train test split 
data_pd['train_test_split'] = data_pd['image_id'].apply(identify_trainOrtest)
train_df = data_pd[data_pd['train_test_split'] == 'train']
test_df = data_pd[data_pd['train_test_split'] == 'test']

print(train_df.dx.value_counts())
print(test_df.dx.value_counts())

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
train_df,class_distribution = limit_samples_per_class(train_df, max_per_class=600)

# checking the train distribution
print("Train distribution:\n", train_df['dx'].value_counts())
print("Train size:", len(train_df))
print("Test size:", len(test_df))

#creating json just to be safe
def make_image_label_json(df):
    return df[['image_id', 'dx']].to_dict(orient='records')

train_json = make_image_label_json(train_df)
test_json = make_image_label_json(test_df)

with open("classes_distribution.json","w") as f:
    json.dump(class_distribution,f,indent=4)

with open("train_600.json", "w") as f:
    json.dump(train_json, f, indent=4)

with open("test_600.json", "w") as f:
    json.dump(test_json, f, indent=4)

# organizing based on format required
source_dir = './images'
output_dir = './dataset_600'

# Create folders
for split in ['train', 'test']:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

def organize_images(df, split):
    for _, row in df.iterrows():
        img_name = row['image_id'] + '.jpg'
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
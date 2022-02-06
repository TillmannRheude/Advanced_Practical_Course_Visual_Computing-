""" 
    Description

    This file is responsible for adding together all body parts to be able to train VoxelMorph on "all" parts.
    IMPORTANT: This file only works if all images and segmentations for the body parts have already been 
    created with preprocessing.py for the current variant and input data.

"""


""" Imports """
import os 
import shutil

import  glob
from argparse import ArgumentParser



""" Arguments """
parser = ArgumentParser()
parser.add_argument("-m", "--method", type=str, default="padding") #Parameter: [padding, scaling]
args = parser.parse_args()

# Create the root folder
if os.path.isdir(f"data_{args.method}/dataset_all"):
    shutil.rmtree(f"data_{args.method}/dataset_all")

# Get variant and body part
padding_or_scaling_path = f"data_{args.method}/*/"
all_body_parts = glob.glob(padding_or_scaling_path)

# Make the directories for all data
os.makedirs(f"data_{args.method}/dataset_all/dataset_bones/test/images")
os.makedirs(f"data_{args.method}/dataset_all/dataset_bones/test/segmentations")
os.makedirs(f"data_{args.method}/dataset_all/dataset_bones/train/images")
os.makedirs(f"data_{args.method}/dataset_all/dataset_bones/train/segmentations")
os.makedirs(f"data_{args.method}/dataset_all/dataset_muscles/test/images")
os.makedirs(f"data_{args.method}/dataset_all/dataset_muscles/test/segmentations")
os.makedirs(f"data_{args.method}/dataset_all/dataset_muscles/train/images")
os.makedirs(f"data_{args.method}/dataset_all/dataset_muscles/train/segmentations")
os.makedirs(f"data_{args.method}/dataset_all/dataset_original/test/images")
os.makedirs(f"data_{args.method}/dataset_all/dataset_original/test/segmentations")
os.makedirs(f"data_{args.method}/dataset_all/dataset_original/train/images")
os.makedirs(f"data_{args.method}/dataset_all/dataset_original/train/segmentations")

# Iterate and put the files together in the directories
for body_part in all_body_parts: 
    all_variants = glob.glob(body_part + "*/")
    for variant in all_variants:
        train_test = glob.glob(variant + "*/")
        for split in train_test: 
            images_segmentations = glob.glob(split  + "*/")
            for file_seg_or_img in images_segmentations: 
                files = glob.glob(file_seg_or_img + "/*.png")
                print(files)
                
                for data in files: 
                    
                    if "dataset_feet_flat" in data:
                        new_path = data.replace("dataset_feet_flat", "dataset_all")
                        new_path = new_path.replace("R_", "ff_R_")
                    if "dataset_hands_flat" in data:
                        new_path = data.replace("dataset_hands_flat", "dataset_all")
                        new_path = new_path.replace("R_", "hf_R_")
                    if "dataset_hands_bent" in data:
                        new_path = data.replace("dataset_hands_bent", "dataset_all")
                        new_path = new_path.replace("R_", "hb_R_")
                    print(data)
                    print(new_path)
                    shutil.copyfile(data, new_path)

""" 
Description

UPDATE: THIS FILE IS OUTDATED. The augmentation is done in model.py

This file is responsible for doing data augmentation for the training, testing and auxillary data. 

Input: Training images

Rotates, Mirrors and/or contrasts the image(s). 

Output: Augmented images

Parameters:
    method (-m):
        padding             (default)
        scaling
    folder (-f):
        train               (default)
        test
    auxinf (-ai)
        False               (default)
        True            
"""

# TODO: Bei Rotation und Scaling werden am oberen Rand teile der Bilder abgeschnitten, d.h. Bild müsste erweitert werden

""" Imports """
import os 
import numpy as np
import cv2

from matplotlib import image
from matplotlib import pyplot as plt
from argparse import ArgumentParser


""" Arguments """
parser = ArgumentParser()
parser.add_argument("-m", "--method", type=str, default="padding") #Parameter: [padding, scaling]
parser.add_argument("-f", "--folder", type=str, default="train") #Parameter: [train, test]
parser.add_argument("-ai", "--auxinf", type=str, default="False") #Parameter: [True, False]

args = parser.parse_args()


""" Functions """
def load_img(img_path):
    img = image.imread(img_path)
    # img = Image.open(img_path)

    return img

def save_image(np_array, save_path):
    #np_array = np_array.astype(np.uint8)
    #im = Image.fromarray(np_array)
    #im.save(save_path)
    plt.imsave(save_path, np_array, cmap = "gray")

def mirror(picture):
    """ Mirror the image vertically. 

    Args:
        picture ([np_array]): Array of the loaded image

    Returns:
        [np_array]: Array of the mirrored image
    """
    picture = np.fliplr(picture)
    picture = (picture - np.min(picture))/np.ptp(picture)
    return picture

def rotation(picture):
    """ Rotate the image. 

    Args:
        picture ([np_array]): Array of the loaded image

    Returns:
        [np_array]: Array of the rotated image
    """
    image_center = tuple(np.array(picture.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, 20, 1.0)
    picture = cv2.warpAffine(picture, rot_mat, picture.shape[1::-1], borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    picture = (picture - np.min(picture))/np.ptp(picture)
    return picture

def contrast(picture):
    """ Make the contrast of the image higher/lower. 

    Args:
        picture ([np_array]): Array of the loaded image

    Returns:
        [np_array]: Array of the contrast-changed image.
    """
    alpha = 2.5
    beta = 0
    picture[:,:,3] = picture[:,:,3] * alpha + beta

    picture = (picture - np.min(picture))/np.ptp(picture)

    return picture


""" Main """
def main():

    if args.method == "padding":
        root = "data_padding/"
    elif args.method == "scaling":
        root = "data_scaling/"

    if "True" == args.auxinf:
        if "train" == args.folder:
            print("train images of auxinf used for augmentation")
            root_list = [
                root + "dataset_hands_flat/dataset_bones/train/segmentations/", root + "dataset_hands_flat/dataset_muscles/train/segmentations/", root + "dataset_hands_flat/dataset_original/train/segmentations/",
                root + "dataset_hands_bent/dataset_bones/train/segmentations/", root + "dataset_hands_bent/dataset_muscles/train/segmentations/", root + "dataset_hands_bent/dataset_original/train/segmentations/",
                root + "dataset_feet_flat/dataset_bones/train/segmentations/", root + "dataset_feet_flat/dataset_muscles/train/segmentations/", root + "dataset_feet_flat/dataset_original/train/segmentations/"
            ]
        elif "test" == args.folder:
            print("test images of auxinf used for augmentation")
            root_list = [
                root + "dataset_hands_flat/dataset_bones/test/segmentations/", root + "dataset_hands_flat/dataset_muscles/test/segmentations/", root + "dataset_hands_flat/dataset_original/test/segmentations/",
                root + "dataset_hands_bent/dataset_bones/test/segmentations/", root + "dataset_hands_bent/dataset_muscles/test/segmentations/", root + "dataset_hands_bent/dataset_original/test/segmentations/",
                root + "dataset_feet_flat/dataset_bones/test/segmentations/", root + "dataset_feet_flat/dataset_muscles/test/segmentations/", root + "dataset_feet_flat/dataset_original/test/segmentations/"
            ]
    else:
        if "train" == args.folder:
            print("train images of xrays used for augmentation")
            root_list = [
                root + "dataset_hands_flat/dataset_bones/train/images/", root + "dataset_hands_flat/dataset_muscles/train/images/", root + "dataset_hands_flat/dataset_original/train/images/",
                root + "dataset_hands_bent/dataset_bones/train/images/", root + "dataset_hands_bent/dataset_muscles/train/images/", root + "dataset_hands_bent/dataset_original/train/images/",
                root + "dataset_feet_flat/dataset_bones/train/images/", root + "dataset_feet_flat/dataset_muscles/train/images/", root + "dataset_feet_flat/dataset_original/train/images/"
            ]
        elif "test" == args.folder:
            print("test images of xrays used for augmentation")
            root_list = [
                root + "dataset_hands_flat/dataset_bones/test/images/", root + "dataset_hands_flat/dataset_muscles/test/images/", root + "dataset_hands_flat/dataset_original/test/images/",
                root + "dataset_hands_bent/dataset_bones/test/images/", root + "dataset_hands_bent/dataset_muscles/test/images/", root + "dataset_hands_bent/dataset_original/test/images/",
                root + "dataset_feet_flat/dataset_bones/test/images/", root + "dataset_feet_flat/dataset_muscles/test/images/", root + "dataset_feet_flat/dataset_original/test/images/"
            ]    

    for picture_path in root_list:
        pair_done = False
        # picture_path = root + "dataset_hands_flat/dataset_bones/train/"

        for filename in sorted(os.listdir(picture_path)):
            # filename = R_01_1_l_Hand_Bones_padding.png
            
            if filename == ".gitignore":
                continue 
            
            picture = load_img(picture_path + filename)
            mirrored_picture = mirror(picture)
            picture = load_img(picture_path + filename)
            contrasted_picture = contrast(picture)
            picture = load_img(picture_path + filename)
            rotated_picture = rotation(picture)

            filename = filename.split("_1_")

            save_image(mirrored_picture, picture_path + f"{filename[0]}_{2}_{filename[1]}")
            save_image(contrasted_picture, picture_path + f"{filename[0]}_{3}_{filename[1]}")
            save_image(rotated_picture, picture_path + f"{filename[0]}_{4}_{filename[1]}")
            print(f"{filename[0]}_{4}_{filename[1]}")

            if pair_done: 
                pair_done = False
            else:
                pair_done = True



if __name__ == "__main__":
    main()
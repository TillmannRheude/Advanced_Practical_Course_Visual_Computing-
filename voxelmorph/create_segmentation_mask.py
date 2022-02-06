""" 
    Description

    This file is responsible for adding together the input data for VoxelMorph.

"""

""" Imports """
import os 
import numpy as np

from utils import load_img, save_image



""" Functions """
def create_one_coherent_segmentation_mask(path_list_of_images):
    """ Create one coherent segmentation mask out of many small segmentation masks. This is useful for missing segmentation masks
    even if smaller segmentations are available (e.g. for feet data).

    Args:
        path_list_of_images ([list of strings]): List containing all paths to the smaller segmentation masks. 

    Returns:
        [np_array]: Array with the information collected into one image. 
    """
    # get shapes of one image which should be the shape of all images
    height, width = load_img(path_list_of_images[0]).shape

    # create empty (final) segmentation mask
    segmentation_mask = np.zeros((height, width))

    for i, path in enumerate(path_list_of_images):
        print(path)
        actual_seg_information = load_img(path)
        segmentation_mask[actual_seg_information == 1] = actual_seg_information[actual_seg_information == 1]

    return segmentation_mask


def main():
    
    # Define the folders which contain all the mask part images
    folder_list = ["dataset/Feet_flat", "dataset/Hands_bent", "dataset/Hands_flat"]
    save_name_list = ["feetBones.png", "handBones.png", "handBones.png"]

    # Define names of masks which should be put together
    mask_parts = ["mt", "mc", "pd", "pm", "pp", "carpal"]

    for f in range(len(folder_list)): 
        # Define folders analog to these ones
        folders = os.listdir(folder_list[f])
        name_for_saving = save_name_list[f]
        folder_path = folder_list[f] + "/"

        # remove gitignore from the list
        for i in range(len(folders)): 
            if folders[i] == ".gitignore": 
                folders.remove(".gitignore")
                break

        # iterate through all folders inside
        for x in range(len(folders)):
            path_list_of_parts = []

            sub_folder_path = folder_path + folders[x] + "/"

            path_list_of_images = os.listdir(sub_folder_path)
            for i in range(len(path_list_of_images)):
                path_list_of_images[i] = sub_folder_path + path_list_of_images[i]

            # if one image is found which is not in mask_parts, it should be removed from path list
            print(path_list_of_images)
            print(len(path_list_of_images))
            for i in range(len(path_list_of_images)):
                print(path_list_of_images[i])
                contains_parts = any(mask_part in path_list_of_images[i] for mask_part in mask_parts)
                
                if contains_parts:
                    path_list_of_parts.append(path_list_of_images[i])
                    # break
            
            path_list_of_images = path_list_of_parts
            # Create the coherent segmentation mask out of all the mask parts in path_list_of_images
            segmentation_mask = create_one_coherent_segmentation_mask(path_list_of_images)
            # Save the coherent segmentation mask
            save_image(segmentation_mask, sub_folder_path + name_for_saving)


if __name__ == "__main__":
    main()
"""
Removing unnecessary files with a specific name(s) from specific directories to clean the projects directories.

!!!!! IMPORTANT: Please note that you have to insert the names of the image files you want to delete and the correspoinding path to the image directories. (marked with TODO)
"""
import os
import shutil

paths = [ 
    # TODO: please select the path to all directories which should be included in the removing process
    
    # paths for files
    #"dataset/Hands_flat/", "dataset/Feet_flat/", "dataset/Hands_bent/"
    #"data_scaling/dataset_hands_flat/dataset_bones/", "data_scaling/dataset_hands_bent/dataset_bones/", "data_scaling/dataset_feet_flat/dataset_bones/",
    #"data_scaling/dataset_hands_flat/dataset_muscles/", "data_scaling/dataset_hands_bent/dataset_muscles/", "data_scaling/dataset_feet_flat/dataset_muscles/",
    #"data_scaling/dataset_hands_flat/dataset_original/", "data_scaling/dataset_hands_bent/dataset_original/", "data_scaling/dataset_feet_flat/dataset_original/",
    #"data_padding/dataset_hands_flat/dataset_bones/", "data_padding/dataset_hands_bent/dataset_bones/", "data_padding/dataset_feet_flat/dataset_bones/",
    #"data_padding/dataset_hands_flat/dataset_muscles/", "data_padding/dataset_hands_bent/dataset_muscles/", "data_padding/dataset_feet_flat/dataset_muscles/",
    #"data_padding/dataset_hands_flat/dataset_original/", "data_padding/dataset_hands_bent/dataset_original/", "data_padding/dataset_feet_flat/dataset_original/"
    
    # paths for directories
    "data_padding/dataset_hands_flat/", "data_scaling/dataset_hands_flat/", "data_padding/dataset_hands_bent/", "data_scaling/dataset_hands_bent/", "data_padding/dataset_feet_flat/", "data_scaling/dataset_feet_flat/"
]

for path in paths:

    input_paths = [ f.path for f in os.scandir(path) if f.is_dir() ]

    # adding directory names of each image folder to the path
    for input_path in input_paths: 
        output_path = input_path + "/"

        # removing all files which match the target filename
        for file_name in os.listdir(input_path):
            # TODO: please replace string with the name of the file you want to delete in each image folder
            if "val" in file_name:
                # for deleting files/images
                #os.remove(output_path + file_name)

                # for deleting directories
                shutil. rmtree(output_path + file_name)

                print(output_path + file_name + " successfully removed")
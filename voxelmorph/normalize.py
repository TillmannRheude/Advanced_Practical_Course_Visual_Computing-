""" 
    Description

    This file is a helper file to normalize the segmentations.
    
    Use Case: after drawing the segmentations with the tablets we needed this code to set the pixels to either zero or one.

"""


""" Imports """
import cv2 as cv
import os
import numpy as np

from PIL import Image 

# Selection of target paths with all files we want to nomralize
# TODO: as the case may be you can insert here the path of the target data that you wont to nomralize
paths = [ 
    "dataset/Feet_flat/", "dataset/Hands_bent/", "dataset/Hands_flat/"
]

for path in paths:

    input_paths = [ f.path for f in os.scandir(path) if f.is_dir() ]

    for input_path in input_paths: 
        output_path = input_path + "/"

        for file_name in os.listdir(input_path):

            if file_name.endswith(".mitk"):
                continue

            #TODO if necessary adapt file name of the target image file(s)
            if "skin" in file_name or "mc" in file_name or "pd" in file_name or "pp" in file_name or "pm" in file_name or "carpal" in file_name: 

                file_path = os.path.join(input_path, file_name)
                # img = cv.imread(file_path, 0) # read as gray image

                print(file_path)
                img = Image.open(file_path)
                img = np.asarray(img)

                # get non-zero pixel indices
                rows, cols = np.nonzero(img) 
                indices = list(zip(rows, cols))

                # for debuging:
                # print(np.unique(img))

                img_normalized = img.copy()
                for r, c in indices:
                    # set all non zero pixels to white
                    img_normalized[r, c] = 255
                save_img_path = os.path.join(output_path, file_name)
                cv.imwrite(save_img_path, img_normalized)

    print(f"Done with path {path}")
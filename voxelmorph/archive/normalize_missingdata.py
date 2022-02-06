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

#TODO insert here the path of the target data that you wont to nomralize
paths = ["hands_bent_no_processing/"]

for path in paths:
    input_paths = [ f.path for f in os.scandir(path) if f.is_dir() ]

    for input_path in input_paths: 
        output_path = input_path + "/"
        
        for file_name in os.listdir(input_path):
            if file_name.endswith(".mitk"):
                continue
            
            #TODO if necessary adapt file name of the target image file
            if "Left_bent" in file_name: 

                file_path = os.path.join(input_path, file_name)
                # img = cv.imread(file_path, 0) # read as gray image

                print(file_path)
                img = Image.open(file_path)
                img = np.asarray(img)

                # get non-zero pixel indices
                rows, cols = np.nonzero(img) 
                indices = list(zip(rows, cols))

                #print(np.unique(img))
                #print(img)
                #print(indices)
                
                img_normalized = img.copy()
                for r, c in indices:
                    # multiply all non zero pixels
                    img_normalized[r, c] = (img_normalized[r, c] / 255) * 255 
                save_img_path = os.path.join(output_path, file_name)
                cv.imwrite(save_img_path, img_normalized)
                
            else:
                continue
                file_path = os.path.join(input_path, file_name)
                # img = cv.imread(file_path, 0) # read as gray image

                print(file_path)
                img = Image.open(file_path)
                img = np.asarray(img)

                # get non-zero pixel indices
                rows, cols = np.nonzero(img) 
                indices = list(zip(rows, cols))

                print(np.unique(img))

                img_normalized = img.copy()
                for r, c in indices:
                    # set all non zero pixels to white
                    img_normalized[r, c] = 255 
                save_img_path = os.path.join(output_path, file_name)
                cv.imwrite(save_img_path, img_normalized)

    print(f"Done with path {path}")
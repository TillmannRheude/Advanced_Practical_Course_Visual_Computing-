"""
    Description

    Preprocessing prepares the data for voxelmorph. 

    Input: xray images and segmentation masks from the dataset directory

    Cleans the image according to the scope of the segemtation mask:
        1. original = no cleaning, the original image is used
        2. muscles = removes the background noise, edges of the xray and the letters L or R, only the hand remains
        3. bones = removes in addition to muscles also the skin and soft tissue of the Hand or Feet, only the bones of the hand remains

    Voxelmorph needs quadratic images which all have the same size. We support two methods to acoplish this: scaling and padding.
        Scaling:    The images are downscaled (nearest neighbour) to get the size of the smallest image.
                    In case the lagest edge is the hight, only the width is padded to get a quadratic image and vice versa.
        Padding:    The images are padded in hight and width according to the lagest edge of all images.

    Note:   For Validating our model outputs we also have to preprocess the segmentations of the single bones. 
            For this purpose the auxinf parameter must be set to True.

    Output: images ready for model traing with voxelmorph

    Parameters:
        input_data (-i):
            hf = hands-flat    (default)
            hb = hands-bent 
            ff = feet-flat
        method (-m):
            padding             (default)
            scaling
        auxinf (-ai)
            False               (default)
            True
"""


""" Imports """
import os 
import numpy as np
import cv2 as cv
import math
import shutil

from matplotlib import image
from matplotlib import pyplot as plt
from argparse import ArgumentParser


""" Arguments """
parser = ArgumentParser()
parser.add_argument("-i", "--input_data", type=str, default="hf") #Parameter: [hf, hb, ff] ... hf = hands-flat, hb = hands-bent, ff = feet-flat
parser.add_argument("-m", "--method", type=str, default="padding") #Parameter: [padding, scaling]
parser.add_argument("-ai", "--auxinf", type=bool, default=False) #Parameter: [True, False]
args = parser.parse_args()


segmentation_names = ["mc", "pd", "pm", "pp", "mt"]


""" Functions """
def load_img(img_path):
    img = image.imread(img_path)
    return img

def save_image(np_array, save_path):
    plt.imsave(save_path, np_array, cmap = "gray")

def show_only_segmented_information(segmentation_mask, xray_image):
    # if segmenation mask is larger than the xray image the xray image will be adjusted
    if segmentation_mask.shape[0] > xray_image.shape[0]:
        height, width = xray_image.shape
        xray_image_new = np.zeros_like(segmentation_mask)
        new_height, new_width = xray_image_new.shape[0] - height, xray_image_new.shape[1] - width
        xray_image_new[:height, new_width:] = xray_image
        xray_image = xray_image_new    
    # perform segmentation by appling the segmentation mask onto the xray image
    xray_image_clean = np.zeros_like(xray_image)
    xray_image_clean[segmentation_mask == 1] = xray_image[segmentation_mask == 1]

    return xray_image_clean

def preproess_image(filename, path, method, largest_width, largest_height, smallest_width, smallest_height):
    """
    Depending of the chosen method (padding/scaling) this function modifys and saves each images with a square
    shape and uniform size.

    Input:
    ------
        filename (str): name of the image file
        path (str): path to the directory of the images
        method (str): padding or scaling
        largest_width (int): largest image width (for padding)
        largest_hight (int): largest image hight (for padding)
        smallest_width (int): smallest image width (for scaling)
        smallest_hight (int): smallest image hight (for scaling)

    """
    if filename == ".gitignore" or filename == "test" or filename == "train":
        return
    
    if filename == "Thumbs.db":
        os.remove(path + filename)
        return

    # change the data path depending on test or train data (depending on the numbers we specified).
    # We choose that image 02 and 06 are our testing data.
    if "02" in filename or "06" in filename:
        data_path = "test" 
    else:
        data_path = "train"

    cleaned_image = load_img(path + filename)
    if cleaned_image.ndim == 3: height, width, _ = cleaned_image.shape
    if cleaned_image.ndim == 2: height, width = cleaned_image.shape

    # Padding 
    if method == "padding":
        # get largest edge of the largest image
        if largest_width > largest_height: 
            largest_edge = largest_width 
        else: 
            largest_edge = largest_height

        # use largest edge to padd with the right amout of zero/black pixels
        if cleaned_image.shape[1] < largest_edge or cleaned_image.shape[0] < largest_edge:
            cleaned_image_new = np.zeros((largest_edge, largest_edge))

            padding_width = largest_edge - width
            padding_height = largest_edge - height

            if cleaned_image.ndim == 3: cleaned_image = cleaned_image[:,:,0]

            cleaned_image_new[int(math.ceil(padding_height / 2)):(largest_edge - int(math.floor(padding_height/2))), int(math.ceil(padding_width/2)):(largest_edge - int(math.floor(padding_width/2)))] = cleaned_image

    # Scaling
    if method == "scaling":                    
        if height > width:
            # padd only the width of the segmented image to get an quadratic image
            padding_width = height - width
            padded_image_new = np.zeros((height, height))
            if cleaned_image.ndim == 3: cleaned_image = cleaned_image[:,:,0]
            padded_image_new[:, int(math.ceil(padding_width/2)):(height - int(math.floor(padding_width/2)))] = cleaned_image
        else:
            # padd only the hight of the segmented image to get an quadratic image
            padding_height = width - height
            padded_image_new = np.zeros((width, width))
            if cleaned_image.ndim == 3: cleaned_image = cleaned_image[:,:,0]
            padded_image_new[int(math.ceil(padding_height/2)):(width - int(math.floor(padding_height/2))), :] = cleaned_image
        
        # get lagest edge of the smallest image for resizing
        if smallest_width < smallest_height:
            largest_edge = smallest_height
        else:
            largest_edge = smallest_width
        
        # resizing the quadratic image
        cleaned_image_new = cv.resize(padded_image_new, (largest_edge, largest_edge),
                                      0, 0, interpolation=cv.INTER_NEAREST)
    
    # Save preprocessed image
    if segmentation_names[0] in filename or segmentation_names[1] in filename or segmentation_names[2] in filename or segmentation_names[3] in filename or segmentation_names[4] in filename:
        save_image(cleaned_image_new, path + data_path + "/segmentations" + "/" + filename)
        print("Image saved under ", path + data_path + "/segmentations" + "/" + filename)
        os.remove(path + filename)
    else:
        save_image(cleaned_image_new, path + data_path + "/images" + "/" + filename)
        print("Image saved under ", path + data_path + "/images" + "/" + filename)
        os.remove(path + filename)


""" Main """
def main():

    print(f"The input_data chosen is called: {args.input_data}")
    print(f"The method to preprocess the data is called: {args.method}")

    # Definition of our directory structures
    method = args.method
    name_dict = dict()
    name_list = [{"bones" : ["dataset_hands_flat/dataset_bones/", "handBones.png", f"Hand_Bones_{method}.png"],
                    "muscles" : ["dataset_hands_flat/dataset_muscles/", "skin.png", f"Hand_Muscles_{method}.png"],
                    "original" : ["dataset_hands_flat/dataset_original/", "NONE", f"Hand_Original_{method}.png"]},

                {"bones" : ["dataset_hands_bent/dataset_bones/", "handBones.png", f"Hand_Bones_{method}.png"],
                    "muscles" : ["dataset_hands_bent/dataset_muscles/", "skin.png", f"Hand_Muscles_{method}.png"],
                    "original" : ["dataset_hands_bent/dataset_original/", "NONE", f"Hand_Original_{method}.png"]},

                {"bones" : ["dataset_feet_flat/dataset_bones/", "feetBones.png", f"Feet_Bones_{method}.png"],
                    "muscles" : ["dataset_feet_flat/dataset_muscles/", "skin.png", f"Feet_Muscles_{method}.png"],
                    "original" : ["dataset_feet_flat/dataset_original/", "NONE", f"Feet_Original_{method}.png"]}]
    root_list = ["dataset/Hands_flat", "dataset/Hands_bent", "dataset/Feet_flat"]
    out_dir = "data_padding/" if method == "padding" else "data_scaling/"
    file_left = "Left_flat.png"
    file_right = "Right_flat.png"

    # Define root_list depending on which input_data
    if "hf" == args.input_data:
        root_list = ["dataset/Hands_flat"]
    elif "hb" == args.input_data:
        root_list = ["dataset/Hands_bent"]
    elif "ff" == args.input_data:
        root_list = ["dataset/Feet_flat"]

    # Choose data with cmd line to preprocess
    for r in range(0, len(root_list)):
        rootdir = root_list[r]
        name_dict = name_list[r]

        if "ff" == args.input_data:
            rootdir = root_list[0]
            name_dict = name_list[2]
        elif "hb" == args.input_data:
            rootdir = root_list[0]
            name_dict = name_list[1]
        elif "hf" == args.input_data:
            rootdir = root_list[0]
            name_dict = name_list[0]

        
        # Generate for each data image type of bones, muscles and original
        for key in name_dict:
            print(f"Currently we care about: {key}")
            seg_size_info = dict()
            # Get information to save new image
            # Subdirectory of actual segmenation scope (bones, muscles, original)
            segmentation_inf = out_dir + name_dict[key][0]
            # Name of segmentation mask (saved in root_list/subdirectory)
            segmentation_mask_name = name_dict[key][1]
            # Name of preprocessed image (with mode scale or padding)
            name = name_dict[key][2]
            # Get original image
            if "hands_bent" in segmentation_inf: 
                file_left = "Left_bent.png"
                file_right = "Right_bent.png"

            # Delete directories if they already exist
            if os.path.isdir(segmentation_inf + "train"): shutil.rmtree(segmentation_inf + "train")
            if os.path.isdir(segmentation_inf + "test"): shutil.rmtree(segmentation_inf + "test")

            # Create new directories
            if os.path.isdir(segmentation_inf + "train") == False:
                os.makedirs(segmentation_inf + "train")
                os.makedirs(segmentation_inf + "train/images")
                os.makedirs(segmentation_inf + "train/segmentations")

                os.makedirs(segmentation_inf + "test")
                os.makedirs(segmentation_inf + "test/images")
                os.makedirs(segmentation_inf + "test/segmentations")

            # Variable defintions
            largest_width, largest_height = 0, 0
            smallest_width, smallest_height = np.inf, np.inf
            xray_found = False 
            segmentation_found = False
            
            # Iterate through every image directory ("R_XY_ ...") in the current directory (Hands_flat/Hands_bent/Feet_flat)
            for subdir, dirs, files in os.walk(rootdir):

                if (".gitignore" in subdir):
                    continue

                if subdir.endswith("l") or subdir.endswith("d"):
                    # Iterate through each directory of actual directory of root_list
                    for file in files:

                        if (".gitignore" == file):
                            continue

                        # If we have the segmentation mask for a left Hand/Feet load the image of the left Hand/Feet
                        if file == file_left and subdir.endswith("l"):
                            xray_image_name = os.path.join(subdir,file)
                            xray_image = load_img(os.path.join(subdir,file))
                            xray_found = True
                        # If we have the segmentation mask for a right Hand/Feet load the image of the right Hand/Feet
                        if file == file_right and (subdir.endswith("d")): # d stands for right_rotated
                            xray_image_name = os.path.join(subdir,file)
                            xray_image = load_img(os.path.join(subdir,file))
                            xray_found = True
                        # Load segmentation mask
                        if file == segmentation_mask_name and segmentation_mask_name != "NONE":
                            segmentation_mask_id = os.path.join(subdir,file)
                            segmentation_mask = load_img(os.path.join(subdir,file))
                            segmentation_found = True
                        # if original xray and segmenation mask exist ...
                        if xray_found and (segmentation_found or segmentation_mask_name == "NONE"):

                            # Use segmentation mask to create segmented image ("xray_image_cleaned")
                            if segmentation_found:
                               # Check dimension of the mask and scale it down to 2 dimensions
                                if segmentation_mask.ndim > 2:
                                    segmentation_mask = segmentation_mask[:,:,0]
                                xray_image_clean = show_only_segmented_information(segmentation_mask, xray_image)

                            # Else if no segmentation mask exists, we later want to save original image
                            elif segmentation_mask_name == "NONE":
                                #save_image(xray_image, subdir + "/" + name)
                                xray_image_clean = xray_image

                            # get pixels where xray_image_clean is 0 along axis 1
                            if args.auxinf and os.path.basename(subdir) not in seg_size_info:
                                info = (np.where(~np.all(xray_image_clean == 0, axis=1))[0][0],
                                        np.where(~np.all(xray_image_clean == 0, axis=1))[0][-1])
                                seg_size_info[os.path.basename(subdir)] = info # xray_image_clean[info[0]:info[1]]

                            xray_image_clean = xray_image_clean[~np.all(xray_image_clean == 0, axis=1)]

                            if method == "padding":
                                # Get lagest width and hight within all images
                                if xray_image_clean.shape[1] > largest_width:
                                    largest_width = xray_image_clean.shape[1]
                                if xray_image_clean.shape[0] > largest_height:
                                    largest_height = xray_image_clean.shape[0]

                            if method == "scaling":
                                # Get smallest width and hight within all images
                                if xray_image_clean.shape[1] < smallest_width:
                                    smallest_width = xray_image_clean.shape[1]
                                if xray_image_clean.shape[0] < smallest_height:
                                    smallest_height = xray_image_clean.shape[0]

                            # Save segmented image to corresponding directory of preproecessed images (e.g. data_feet_flat/dataset_bones/image)
                            dirname = os.path.basename(subdir)
                            save_image(xray_image_clean, segmentation_inf + dirname + "_" + name)

                            #reset booleans for next iteration
                            xray_found = False
                            segmentation_found = False

                        if args.auxinf:
                            if (segmentation_names[0] in file or segmentation_names[1] in file or segmentation_names[2] in file or
                                    segmentation_names[3] in file or segmentation_names[4] in file):
                                dirname = os.path.basename(subdir)
                                new_filename = segmentation_inf + dirname + "_" + file
                                shutil.copy(os.path.join(rootdir, subdir.split("/")[-1], file),
                                            new_filename)

                    if xray_found == False or segmentation_found == False:
                        #reset booleans for next iteration
                        xray_found = False
                        segmentation_found = False

            # For each segmented image 
            print(f"Do preprocess with mode {method}")
            print(os.listdir(segmentation_inf))
            for filename in os.listdir(segmentation_inf):
                if (segmentation_names[0] in filename or segmentation_names[1] in filename or segmentation_names[2] in filename or
                        segmentation_names[3] in filename or segmentation_names[4] in filename):
                    new_filename = segmentation_inf + filename
                    auxinf_image = load_img(os.path.join(segmentation_inf, filename))

                    image_name = filename[:16] if "rotated" in filename else filename[:8]

                    # Resize image, so that the segmentation has the same size as the segmented image
                    info_size = seg_size_info[image_name]
                    auxinf_image = auxinf_image[info_size[0]:info_size[1]]
                    save_image(auxinf_image, new_filename)

                preproess_image(filename, segmentation_inf, method, largest_width, largest_height, smallest_width, smallest_height)




if __name__ == "__main__":
    main()
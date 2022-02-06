""" 
    Description

    This file is responsible for different experiments needed for the presentation and evaluation of VoxelMorph.

"""


""" Imports """
import os
import pandas as pd 
from datetime import datetime


# #################################################################################################
# ************************************* Evaluation Runs *******************************************
# #################################################################################################
def create_arg_path(variant, body_part, input_data):
    """Combine variant {padding, scaling}, body_part {bones, muscle, original} and input_data {hf, hb, ff} to get data for evaluation."""
    path_variant = f"data_{variant}"
    path_body_part = f"dataset_{body_part}"
    path_input_data = f"dataset_{input_data}"

    arg_path = f"{path_variant}/{path_body_part}/{path_input_data}"

    return arg_path


def convert_body_part(body_part):
    """Convert body_part name to acronym used in this project."""
    if body_part == "hands_flat":
        new_body_part = "hf"
    elif body_part == "hands_bent":
        new_body_part = "hb"
    elif body_part == "feet_flat":
        new_body_part = "ff"
    elif body_part == "all":
        new_body_part = "all"
    return new_body_part


def evaluation_loop(gpu, variants, body_parts, input_data, vised_versions):
    # Hyperparameter
    modelname = f"model"
    nb_epochs = 20
    # Loop for all the evaluation / experiment stuff
    for variant in variants: 
        for body_part in body_parts:
            for data in input_data: 
                for vised_version in vised_versions: 
                    
                    # get all necessary arguments
                    arg_variant = variant
                    arg_path = create_arg_path(variant, body_part, data)
                    arg_body_part = body_part
                    arg_input_data = data
                    # Check if semi-supervised or unsupervised model is used
                    if vised_version == "semi-supervised": 
                        auxinf = True
                    else: 
                        auxinf = False

                    # Get current modelpath
                    modelpath = f"outputs/evaluations/{arg_variant}/{body_part}/{data}/{vised_version}/{DATE_TODAY}/"

                    # Print for ourselves outputs/evaluations/padding/hands_flat/muscles/unsupervised/2022-01-19/
                    print("-" * 30)
                    print(f"Current path in evaluation: {modelpath}")
                    print("-" * 30)

                    arg_body_part = convert_body_part(body_part)

                    # Call the file with the correct arguments
                    if auxinf: 
                        os.system(f"python3 main.py --size=512 --gpus={gpu} --variant={arg_variant} --crossvalidation=True --auxinf={auxinf} --modelpath={modelpath} --modelname={modelname} --epoch={nb_epochs} --path={arg_path} --body_part={arg_body_part} --input_data={arg_input_data}")
                    else: 
                        os.system(f"python3 main.py --size=512 --gpus={gpu} --variant={arg_variant} --crossvalidation=True --modelpath={modelpath} --modelname={modelname} --epoch={nb_epochs} --path={arg_path} --body_part={arg_body_part} --input_data={arg_input_data}")
                    # Create all necessary statistic-csv-files ... 
                    cv1_statistics = pd.read_csv(f"{modelpath}cv1/statistics.csv")
                    cv2_statistics = pd.read_csv(f"{modelpath}cv2/statistics.csv")
                    cv3_statistics = pd.read_csv(f"{modelpath}cv3/statistics.csv")
                    cv4_statistics = pd.read_csv(f"{modelpath}cv4/statistics.csv")

                    all_euclideans = pd.concat([cv1_statistics["Euclidean distance"], 
                                                cv2_statistics["Euclidean distance"], 
                                                cv3_statistics["Euclidean distance"], 
                                                cv4_statistics["Euclidean distance"]], axis = 1)
                    all_euclideans.index = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

                    all_dices = pd.concat([cv1_statistics["Dice score"], 
                                            cv2_statistics["Dice score"], 
                                            cv3_statistics["Dice score"], 
                                            cv4_statistics["Dice score"]], axis = 1)
                    all_dices.index = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

                    all_ious = pd.concat([cv1_statistics["IoU"], 
                                            cv2_statistics["IoU"], 
                                            cv3_statistics["IoU"], 
                                            cv4_statistics["IoU"]], axis = 1)
                    all_ious.index = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

                    euclidean_statistics = all_euclideans.apply(pd.DataFrame.describe, axis = 1)
                    dice_statistics = all_dices.apply(pd.DataFrame.describe, axis = 1)
                    iou_statistics = all_ious.apply(pd.DataFrame.describe, axis = 1)

                    presentation_data = [["Euclidean distance", euclidean_statistics.loc["mean", "mean"]],
                            ["Dice score", dice_statistics.loc["mean", "mean"]],
                            ["IoU", iou_statistics.loc["mean", "mean"]]
                            ]
                    presentation_statistics = pd.DataFrame(presentation_data, columns = ["Metric", "Mean"])

                    presentation_statistics.to_csv(f"{modelpath}presentation_statistics.csv", encoding = "utf-8")

if __name__ == '__main__':

    # #################################################################################################
    # ************************************ Hyper Parameters *******************************************
    # #################################################################################################
    variants = ["padding", "scaling"]
    body_parts = ["all", "hands_flat", "hands_bent", "feet_flat"]
    input_data = ["muscles", "bones", "original"]

    vised_versions = ["unsupervised", "semi-supervised"]

    cv_splits = ["cv1", "cv2", "cv3", "cv4"]

    # #################################################################################################
    # ************************************ Folder Structure *******************************************
    # #################################################################################################
    # Get todays date
    DATE_TODAY = datetime.today().strftime('%Y-%m-%d')

    # Create the folder for all the subfolders of our evaluations
    if not os.path.isdir('outputs/evaluations'): os.mkdir("outputs/evaluations")

    # Create the nested folder structure
    for variant in variants: 
        if not os.path.isdir(f"outputs/evaluations/{variant}"): os.mkdir(f"outputs/evaluations/{variant}")

        for body_part in body_parts: 
            if not os.path.isdir(f"outputs/evaluations/{variant}/{body_part}"): os.mkdir(f"outputs/evaluations/{variant}/{body_part}")

            for data in input_data: 
                if not os.path.isdir(f"outputs/evaluations/{variant}/{body_part}/{data}"): os.mkdir(f"outputs/evaluations/{variant}/{body_part}/{data}")

                for vised_version in vised_versions: 

                    if not os.path.isdir(f"outputs/evaluations/{variant}/{body_part}/{data}/{vised_version}/{DATE_TODAY}"): 
                        os.makedirs(f"outputs/evaluations/{variant}/{body_part}/{data}/{vised_version}/{DATE_TODAY}")

                    for cv_split in cv_splits: 
                        if not os.path.isdir(f"outputs/evaluations/{variant}/{body_part}/{data}/{vised_version}/{DATE_TODAY}/{cv_split}"): 
                            os.mkdir(f"outputs/evaluations/{variant}/{body_part}/{data}/{vised_version}/{DATE_TODAY}/{cv_split}")
                            os.mkdir(f"outputs/evaluations/{variant}/{body_part}/{data}/{vised_version}/{DATE_TODAY}/{cv_split}/figures")
                            os.mkdir(f"outputs/evaluations/{variant}/{body_part}/{data}/{vised_version}/{DATE_TODAY}/{cv_split}/images")
                            os.mkdir(f"outputs/evaluations/{variant}/{body_part}/{data}/{vised_version}/{DATE_TODAY}/{cv_split}/segmentations")

    # exit() 
    #evaluation_loop(gpu = 0, variants = [variants[0]], body_parts = body_parts, input_data = [input_data[0]], vised_versions = vised_versions)
    #evaluation_loop(gpu = 1, variants = [variants[0]], body_parts = body_parts, input_data = [input_data[1]], vised_versions = vised_versions)
    #evaluation_loop(gpu = 2, variants = [variants[0]], body_parts = body_parts, input_data = [input_data[2]], vised_versions = vised_versions)
    #evaluation_loop(gpu = 3, variants = [variants[1]], body_parts = body_parts, input_data = [input_data[0]], vised_versions = vised_versions)
    #evaluation_loop(gpu = 4, variants = [variants[1]], body_parts = body_parts, input_data = [input_data[1]], vised_versions = vised_versions)
    evaluation_loop(gpu = 5, variants = [variants[1]], body_parts = body_parts, input_data = [input_data[2]], vised_versions = vised_versions)

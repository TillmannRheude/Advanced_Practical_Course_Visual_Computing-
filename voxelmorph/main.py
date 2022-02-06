""" 
    Description

    Main function for training (and testing afterwards) of VoxelMorph.

    The following code is inspired by and partly copied from: 
    1. Example: 
        https://colab.research.google.com/drive/1WiqyF7dCdnNBIANEY80Pxw_mVz4fyV-S?usp=sharing
    2. GitHub: 
        https://github.com/voxelmorph/voxelmorph 
    3. Paper: 
        VoxelMorph: A Learning Framework for Deformable Medical Image Registration
        Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca
        IEEE TMI: Transactions on Medical Imaging. 2019. eprint arXiv:1809.05231

    Input: Training and Testing data 

    Folder structure:

    _dataset (e.g. data_scaling)
    _/____dataset_1 (e.g. dataset_hands_flat)
    _/____/____dataset_variant_1 (e.g. dataset_bones)
    _/____/____/____test
    _/____/____/____/____images
    _/____/____/____/____/____files.png
    _/____/____/____/____segmentations
    _/____/____/____/____/____files.png
    _/____/____/____train
    _/____/____/____/____images
    _/____/____/____/____/____files.png
    _/____/____/____/____segmentations
    _/____/____/____/____/____files.png
    _/____/____dataset_variant_2
    _/____/____/____...
    _/____dataset_2
    _/____/____...

    _outputs
    _/____trained_models
    _/____/____figures
    _/____/____/____...
    _/____/____images
    _/____/____/____...
    _/____/____segmentations
    _/____/____/____...
    _/____/____checkpoint
    _/____/____testmodel.ckpt.index
    _/____/____...
    _/____training_cv
    _/____/____...


    Parameters:
        variant (-v):
            padding                             (default)
            scaling
        body_part (-b):
            hf                                  (default)
            hb
            ff
        input_data (-i):
            bones = bones data                 (default)
            original = original (MRT) data
            muscle = muscle data
        image size (-s, --sizes)                Hint: May use multiple GPU for larger sizes than 256
            256                                 (default)
            ... 
        gpus (-g):
            "4, 5, 6, 7"                        (default)
            "0"
            ...
        --loss_functions (-lf):
            ["MSE", "L2"]                       (default)
            ["MSE"]
            ...
        loss_weights (-lw):
            [1, 0.05, 0.01]                     (default)
            [1, 0.05, 0.05]
            ...
        epoch (-e):
            10                                  (default)
            ...
        batch size for training (-bt, --batch_size_train)
            4                                   (default)
            ... 
        batch size for validation (-bv)         Hint: validation batch size must equal gpu count
            2                                   (default)
            ... 
        crossvalidation (-c):
            False                               (default)
            True = start crossvalidation mode
        cross_val_split (-cs):                  Hint: number of splits for cross validation
            4                                   (default)
            ...
        inference (-inf)
            False = train a new model           (default)
            True = load model for testing
        model_path (-m)
            outputs/trained_models/             (default)
            ...
        model name (-mn)
            testmodel                           (default)
            ...
        auxiliary information (-ai, --auxinf)
            False                               (default)
            True = include bone segmentation in training                                                  
        evaluation mode (--evaluation) 
            False                               (default)
            True = start evaluation mode
"""


""" Imports """
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" # hide all these warnings and informations of TensorFlow

import tensorflow as tf 
import voxelmorph as vxm
import model

from argparse import ArgumentParser

tf.get_logger().setLevel("ERROR") # hide all these warnings and informations of TensorFlow


""" Arguments """
parser = ArgumentParser()
# Paths and Sizes
parser.add_argument("-v", "--variant", type=str, default="padding", 
                    help = "Decide with which data variant to train (padding or scaling)")
parser.add_argument("-b", "--body_part", type=str, default="hf", 
                    help = "Decide with which body part to train (hands_flat, hands_bend or foot)")
parser.add_argument("-i", "--input_data", type=str, default="bones", 
                    help = "Decide with which data to train (bones, original or muscle)")
parser.add_argument("-s", "--size", type=int, default=256, 
                    help = "Quadratic size of the input and output images. May use multiple GPU for larger sizes than 256.")
# GPUs
parser.add_argument("-g", "--gpus", type=str, default="4", 
                    help = "Decide which GPUs should be used for training.")
# Hyperparameter
parser.add_argument("-lf", "--loss_functions", type=list, default=["MSE", "L2"], 
                    help = "Define the loss functions for the training.")
parser.add_argument("-lw", "--loss_weights", type=list, default=[1, 0.05, 0.01], 
                    help = "Define the balancing between the loss functions.")
parser.add_argument("-e", "--epoch", type=int, default=10, 
                    help = "Define how many epoch during training.")
parser.add_argument("-bt", "--batch_size_train", type=int, default=1,
                    help = "Batch size for training set.")
parser.add_argument("-bv", "--batch_size_val", type=int, default=1, 
                    help = "Batch size for validation set.")
# Cross Validation
parser.add_argument("-c", "--crossvalidation", type=bool, default=False, 
                    help = "Decide if training in crossvalidation mode or normal mode.")
parser.add_argument("-cs", "--cross_val_split", type=int, default=4, 
                    help = "Define how many crossvalidation splits.")
# Inference
parser.add_argument("-inf", "--inference", type=bool, default=False, 
                    help = "Decide if training or testing a model. If testing, a model has to be load.")
parser.add_argument("-m", "--modelpath", type=str, default="outputs/trained_models/", 
                    help = "Path to the trained model.")
parser.add_argument("-mn", "--modelname", type=str, default="testmodel", 
                    help = "Name of the trained model which should be loaded.")
# Auxiliary Information
parser.add_argument("-ai", "--auxinf", type=bool, default=False, 
                    help = "Decision if auxiliary information should be used or not.")
# Evaluation
parser.add_argument("-eval", "--evaluation", type=bool, default=False, 
                    help = "Does the evaluation about predicted data.")

args = parser.parse_args()

# Get paths
if args.variant == "padding":
    variant_directory = "data_padding/"  
elif args.variant == "scaling":
    variant_directory = "data_scaling/"

body_part_path = "dataset_hands_flat/"
if args.body_part != "hf":
    body_part_path = "dataset_hands_bent/" if args.body_part == "hb" else "dataset_feet_flat/"
if args.body_part == "all":
    body_part_path = "dataset_all/"

input_data_path = "dataset_bones/"
if args.input_data != "bones":
    input_data_path = "dataset_original/" if args.input_data == "original" else "dataset_muscles/"

path = variant_directory + body_part_path + input_data_path
train_dir = variant_directory + body_part_path + input_data_path + "train/"
test_dir = variant_directory + body_part_path + input_data_path + "test/"


orig_seg_dir = test_dir + "segmentations/"
if args.body_part == "hf": body_part = "hands_flat"
elif args.body_part == "hb": body_part = "hands_bent"
elif args.body_part == "ff": body_part = "feet_flat"
elif args.body_part == "all": body_part = "all"
# pred_seg_dir = "outputs/" + body_part + "/" + args.input_data + "/segmentations/"
pred_seg_dir = args.modelpath + "segmentations/"

""" GPUs """
device, nb_devices = vxm.tf.utils.setup_device(args.gpus)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


print("-" * 30)
print(f"{nb_devices} GPUs are going to be used now.")
print("-" * 30)

def main():
    # Hyperparameters
    # Set epochs
    nb_epochs = args.epoch
    # Set batch size
    batchsize = args.batch_size_train
    # Set learning rate
    learning_rate = 2e-4
    # Set size to train, test and validate on
    sizes = (args.size, args.size)
    
    # Define Crossvalidation split
    cross_split = args.cross_val_split

    # #################################################################################################
    # ************************************ Model Building *********************************************
    # #################################################################################################
    vxm_model = model.Voxelmorph(learning_rate = learning_rate,
                                    sizes = sizes,
                                    auxinf = args.auxinf,
                                    loss_functions = ["MSE", "L2"])


    # #################################################################################################
    # ************************************ Model Training *********************************************
    # #################################################################################################
    if not args.inference and not args.evaluation and not args.crossvalidation:
        vxm_model.train(nb_epochs = nb_epochs, 
                        batchsize_train = batchsize,
                        train_dir = train_dir,
                        modelpath = args.modelpath,
                        modelname = args.modelname)

    if not args.inference and not args.evaluation and args.crossvalidation:
        vxm_model.train_cv(nb_epochs, 
                        batchsize_train = batchsize,
                        batchsize_val = nb_devices, # args.batch_size_val, # should be as large as the number of GPUs
                        train_dir = train_dir,
                        modelname = args.modelname,
                        test_dir = test_dir,
                        dataset_path = path,
                        dataset_input_data = args.input_data,
                        dataset_body_part = args.body_part,
                        path_original_fixed_segmentations = orig_seg_dir,
                        cross_split = args.cross_val_split,
                        modelpath = args.modelpath)


    # #################################################################################################
    # ************************************** Model Testing ********************************************
    # #################################################################################################
    if args.inference:
        vxm_model.test(test_dir = test_dir,
                    modelpath = args.modelpath,
                    modelname = args.modelname,
                    dataset_path = path,
                    dataset_input_data = args.input_data,
                    dataset_body_part = args.body_part)


    # #################################################################################################
    # ************************************ Model Evaluation *******************************************
    # #################################################################################################
    if args.evaluation: 
        evaluator = model.Evaluation(sizes = sizes,
                                    path_original_fixed_segmentations = orig_seg_dir,
                                    path_predicted_segmentations = pred_seg_dir,
                                    auxinf = args.auxinf)

        landmarks_fixed_segmentations, landmarks_predicted_segmentations = evaluator.create_landmarks()

        euclidean_distances = evaluator.euclidean_distances(landmarks_fixed_segmentations, landmarks_predicted_segmentations)
        dice_scores = evaluator.dice_score()
        iou = evaluator.intersection_over_union()
        evaluator.show_landmarks(landmarks_fixed_segmentations = landmarks_fixed_segmentations, 
                            landmarks_predicted_segmentations = landmarks_predicted_segmentations, 
                            save_dir = "outputs/example_landmarks")

        summary, summary_statistics = evaluator.summary()


if __name__ == "__main__":
    main()

""" 
    Description

    This file contains all the classes and functions for building and using VoxelMorph on 2D medical image registration data.

"""


""" Imports """
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" # hide all warnings and informations of TensorFlow

import glob
import numpy as np 
import tensorflow as tf 
tf.get_logger().setLevel("ERROR") # hide all warnings and informations of TensorFlow

import gryds
import voxelmorph as vxm

import pandas as pd

from matplotlib import pyplot as plt

from utils import normalize, flow, slices, plot_loss
from PIL import Image 
from sklearn.model_selection import KFold
import cv2



# #################################################################################################
# ******************************************* VARIABLES *******************************************
# #################################################################################################
COLOR_DICT = ["b", "g", "r", "c", "m", "y", "k"]



# #################################################################################################
# ******************************************* CLASSES *********************************************
# #################################################################################################
class Voxelmorph():
    """ 
        Class for building, training and testing of VoxelMorph. VoxelMorph is used for 2D medical
        image registration in this case. The class is optimized for the folder structure described
        in main.py.
        We adjusted VoxelMorph to fit our project about unsupervised medical image registration.

        Original Paper:
            VoxelMorph: A Learning Framework for Deformable Medical Image Registration
            Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca
            IEEE TMI: Transactions on Medical Imaging. 2019. eprint arXiv:1809.05231
        
        Original GitHub:
            https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, 
                learning_rate = 1e-4,
                sizes = (256, 256),
                auxinf = False,
                loss_functions = ["MSE", "L2", "Soft_Dice"]):
        """ Initialize and build the model of VoxelMorph depending on various parameters.

        Args:
            learning_rate ([int], optional): Set the learning rate for training the network. 
                                            Defaults to 1e-4.
            sizes (tuple, optional): Size of the images which are used for the network. 
                                    Defaults to (256, 256).
            auxinf (bool, optional): Training of the unsupervised VoxelMorph model without delivered segmentations for Training or 
                                    Training of the semi-supervised VoxelMorph model with delivered segmentations for Training. +#
                                    Defaults to False.
            loss_functions (list, optional): List of loss functions for the model. For the semi-supervised model, the Dice score is 
                                            needed and automatically added to the list if it does not already exist. 
                                            Defaults to ["MSE", "L2"].
                                            Possible values MSE, L2, Dice, Soft_Dice, Tversky
        """

        self.auxinf = auxinf
        self.sizes = sizes
        self.loss_functions = loss_functions
        self.learning_rate = learning_rate

        # Build model & define Loss functions
        if not auxinf: 
            # build model
            vxm_model = self.build_model()
        if auxinf: 
            # Define Loss functions and append Dice-Loss to the list
            if "Dice" not in loss_functions or "Soft_Dice" not in loss_functions or "Tversky" not in loss_functions: 
                self.loss_functions.append("Dice")
            
            # build model
            vxm_model = self.build_model()

        self.vxm_model = vxm_model


    def build_model(self, loss_weights = [1, 0.05, 0.05]):
        """ Build the main model of VoxelMorph. 

        Args:
            (list of int) loss_weights: List for the loss balancing. 

        Returns:
            [keras model]: The model of VoxelMorph with defined parameters.
        """
        shape = self.sizes
        nb_features = [
            [shape[0], shape[0]/2, shape[0]/4, shape[0]/8, shape[0]/16], # encoder, e.g. [32, 32, 32, 32]
            [shape[0]/16, shape[0]/8, shape[0]/4, shape[0]/2, shape[0], shape[0], shape[0]] # decoder, e.g. [32, 32, 32, 32, 32, 16]
        ]

        # define loss functions to use
        losses = []
        # all possible losses: https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/tf/losses.py
        for function in self.loss_functions: 
            if function == "MSE": 
                losses.append(vxm.losses.MSE().loss)
            elif function == "L1": 
                losses.append(vxm.losses.Grad('l1').loss)
            elif function == "L2": 
                losses.append(vxm.losses.Grad('l2').loss)
            elif function == "L2_auxinf":
                losses.append(vxm.losses.Grad('l2', loss_mult = 1).loss)
            elif function == "NCC": 
                losses.append(vxm.losses.NCC().loss)
            elif function == "TukeyBiweight":
                losses.append(vxm.losses.TukeyBiweight().loss)
            elif function == "Dice":
                losses.append(vxm.losses.Dice().loss)
            elif function == "Soft_Dice":
                # This loss function was added by Tillmann, Lisa, Julia
                losses.append(vxm.losses.Soft_Dice().loss)
            elif function == "Tversky":
                # This loss function was added by Tillmann, Lisa, Julia
                losses.append(vxm.losses.Tversky().loss)

        # usually, we have to balance the losses by a hyper-parameter
        # multiple losses need multiple weights
        omega_param = loss_weights[0]
        lambda_param = loss_weights[1]
        gamma_param = loss_weights[2]
        if len(losses) == 1:
            loss_weights = [omega_param]
        elif len(losses) == 2:
            loss_weights = [omega_param, lambda_param]
        elif len(losses) == 3:
            loss_weights = [omega_param, lambda_param, gamma_param]
        
        # multiple GPU - this one works in TF 2.x
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            # if auxiliary information is delivered for training
            if self.auxinf:
                vxm_model = vxm.networks.VxmDenseSemiSupervisedSeg(inshape = shape,
                                                                nb_unet_features = nb_features, 
                                                                seg_resolution = 1,
                                                                int_steps = 0,
                                                                reg_field = "warp",
                                                                nb_labels = 19) # mc1-5, pd1-5, pm2-5, pp1-5 for hands = in total 19 files
            
            # no auxiliary information (the segmentation maps) is delivered for training
            elif not self.auxinf:
                vxm_model = vxm.networks.VxmDense(inshape = shape, 
                                                nb_unet_features = nb_features, 
                                                int_steps=0) # int_steps = 0 -> non-diffeomorphic warp, number of flow integration steps
            
        # compile the model and use Adam as optimizer
        vxm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = self.learning_rate), loss=losses, loss_weights=loss_weights)

        return vxm_model


    def train(self,
            nb_epochs, 
            batchsize_train,
            train_dir,
            modelpath,
            modelname,
            data_augmentation = ["rotate", "translate", "b_spline", "combine"]):
        """ Train the semi-supervised or unsupervised model of VoxelMorph with given parameters.

        Args:
            nb_epochs ([int]): Set the number of epochs for training.
            batchsize_train ([int]): Set the batch size for training.
            train_dir ([str]): Select the training directory.
            modelpath ([str]): Select the modelpath (for saving the model).
            modelname ([str]): Select the modelname (for saving the model).
            nb_steps_epoch ([int], optional): Set the number of steps in each epoch. 
            data_augmentation(list of strings, optional): Define which augmentation techniques are used.
                                                        Available techniques are listed in class Augmentation.
        """

        # Set batch size
        batchsize = batchsize_train

        print("-" * 30)
        print(f"Training with {str(nb_epochs)} epochs started. ")
        print("-" * 30)

        # Load the data and create a data generator depending on auxinf or no auxinf
        if not self.auxinf:
            data_train = load_data(train_dir, resize = self.sizes, data_augmentation = data_augmentation).image_data 
            train_generator = vxm_data_generator(data_train, batch_size = batchsize)  
            nb_train_samples = data_train.shape[0]
        if self.auxinf:
            data_train_ = load_data(train_dir, resize = self.sizes, data_augmentation = data_augmentation, aux_information = True) #  = ["rotate", "translate", "b_spline"],
            data_train = [data_train_.image_data, data_train_.segmentation_data]
            train_generator = vxm_data_generator(data_train, batch_size = batchsize, aux_information = True)
            nb_train_samples = data_train[0].shape[0]

        # Initalize hyperparameters
        nb_steps_epoch = nb_train_samples // batchsize

        # Save model
        checkpoint_path = modelpath + modelname + ".ckpt"

        # Create a callback that saves the model's weights at the end of all epochs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                                        save_weights_only=True, 
                                                        verbose=1, 
                                                        save_freq = 'epoch', 
                                                        period = nb_epochs,
                                                        monitor="loss",
                                                        save_best_only=True)

        # Save the weights 
        self.vxm_model.save_weights(checkpoint_path)

        print("Data loading and model building done. Fitting starts.")
        print("-" * 30)

        # start training
        hist = self.vxm_model.fit(train_generator, epochs=nb_epochs, steps_per_epoch=nb_steps_epoch, 
                            verbose=2, # verbose = 1 for progress bar.
                            callbacks=[cp_callback]
                            ) 

        # Save Loss values 
        plot_loss(hist, checkpoint_path[:-5] + f"_loss_history")
        
        print("-" * 30)
        print(f"Training of {str(nb_epochs)} epochs finished.")
        print("-" * 30)
    
    
    def train_cv(self,
                nb_epochs, 
                batchsize_train,
                batchsize_val,
                train_dir,
                modelname,
                cross_split,
                test_dir,
                dataset_path,
                dataset_input_data,
                dataset_body_part,
                path_original_fixed_segmentations,
                modelpath = "outputs/training_cv/",
                data_augmentation = ["rotate", "translate", "b_spline", "combine"]): # 
        """ Train the semi-supervised or unsupervised model of VoxelMorph with given parameters for cross validation.

        Args:
            nb_epochs ([int]): Set the number of epochs for training.
            batchsize_train ([int]): Set the batch size for training.
            batchsize_val ([int]): Set the batch size for validation.
            train_dir ([str]): Select the training directory.
            modelname ([str]): Select the modelname (for saving the model).
            cross_split ([int]): Set the number of splits (for the cross validation).
            modelpath (str, optional): Select the modelpath (for saving the model). 
                                    Defaults to "outputs/training_cv/".
            nb_steps_epoch (int, optional): Set the number of steps in each epoch. 
            data_augmentation (list of strings): Data augmentation techniques.
            test_dir ([str]): Select the test directory.
            dataset_path ([str]): Select the path to the original data.
            dataset_input_data ([str]): Select the version of the input data, e.g. bones.
            dataset_body_part ([str]): Select the version of the body part, e.g. hands flat.
        """
        
        validation_batch_size = batchsize_val # should be as large as the number of GPUs

        print("-" * 30)
        print(f"Training with crossvalidation with {str(nb_epochs)} epochs started. ")
        print("-" * 30)

        # Load the data depending on auxinf 
        if not self.auxinf:
            data_train = load_data(train_dir, resize = self.sizes, aux_information = False, data_augmentation = data_augmentation) # [26*x, 256, 256]  
            data_train_images = data_train.image_data
            # Reshape data to ensure that all image variants (/augmentations) of an image are assigned to the same split
            # Zip augmentations with its original image ([26*x, 256, 256] --> [13, 2, x, 256, 256])
            data_train_images = data_train_images.reshape((-1, data_train.nb_images, data_train.sizes[0], data_train.sizes[1])) #[26*x, 256, 256] --> [x, 26, 256, 256]
            data_train_images = data_train_images.transpose(1, 0, 2, 3) #[x, 26, 256, 256] --> [26, x, 256, 256]
            # Reshape data to ensure that the corresponding image is assigned to the same split (e.g. left hand and right hand)
            # Zip each image pair ([26, x, 256, 256] --> [13, 2, x, 256, 256])
            data_train_images = data_train_images.reshape((int(data_train_images.shape[0]/2), 2, data_train_images.shape[1], data_train.sizes[0], data_train.sizes[1]))
        if self.auxinf:
            data_train = load_data(train_dir, resize = self.sizes, aux_information = True, data_augmentation = data_augmentation)
            
            # For images
            data_train_images = data_train.image_data
            # Reshape data to ensure that all image augmentations of an image are assigned to the same split
            # Zip augmentations with its original image ([26*x, 256, 256] --> [13, 2, x, 256, 256])
            data_train_images = data_train_images.reshape((-1, data_train.nb_images, data_train.sizes[0], data_train.sizes[1])) #[26*x, 256, 256] --> [x, 26, 256, 256]
            data_train_images = data_train_images.transpose(1, 0, 2, 3) #[x, 26, 256, 256] --> [26, x, 256, 256]
            # Reshape data to ensure that the corresponding image is assigned to the same split (e.g. left hand and right hand)
            # Zip each image pair ([26, x, 256, 256] --> [13, 2, x, 256, 256])
            data_train_images = data_train_images.reshape((int(data_train.nb_images/2), 2, data_train_images.shape[1], data_train.sizes[0], data_train.sizes[1]))
            
            # For segmentations
            data_train_segmentations = data_train.segmentation_data
            # Reshape data to ensure that all segmentation image augmentations of a segmentation image are assigned to the same split
            data_train_segmentations = data_train_segmentations.transpose(2, 1, 0) #(256, 256, 3458) --> (3458, 256, 256)
            # Zip augmentations with its original segmentation image ([3458, 256, 256] --> [13, 2, 7, 19, 256, 256])
            data_train_segmentations = data_train_segmentations.reshape((-1 , data_train.nb_images, data_train.nb_bones, data_train.sizes[0], data_train.sizes[1])) #[19*2*13*x, 256, 256] --> [x, 2*13, 19, 256, 256]
            data_train_segmentations = data_train_segmentations.transpose(1, 0, 2, 3, 4) #[x, 2*13, 19, 256, 256] --> [2*13, x, 19, 256, 256]
            # Reshape data to ensure that the corresponding segmentation images are assigned to the same split (e.g. bones from the left hand and bones from the right hand)
            # Zip each image pair ([2*13, x, 19, 256, 256] --> [13, 2, x, 19, 256, 256])
            data_train_segmentations = data_train_segmentations.reshape((int(data_train.nb_images/2), 2, -1, data_train.nb_bones, data_train.sizes[0], data_train.sizes[1]))

        # Create crossvalidation type 
        kfold = KFold(n_splits = cross_split, shuffle=False)
        index_cross_validation = 1

        # To plot different crossvalidation types
        crossvalidation_loss = []
        
        for train, test in kfold.split(data_train_images): 
            print("-" * 30)
            print(f"Actual Crossvalidation Run: {index_cross_validation}/{cross_split}")
            print("-" * 30)
            print("Split: %s %s" % (train, test))
            
            if not self.auxinf:
                # Get training and validation set
                train_set = data_train_images[train]
                val_set = data_train_images[test]
                # Reshape data to ensure that all image augmentations of an image and the corresponding image (e.g. left hand and right hand) are assigned to the same split
                # Resize training set ([10, 2, x, 256, 256] --> [10*2*x, 256, 256])
                train_set = train_set.transpose(0, 2, 1, 3, 4) #([10, 2, x, 256, 256] --> [10, x, 2, 256, 256])
                train_set = train_set.reshape(train_set.shape[0] * train_set.shape[1] * train_set.shape[2], train_set.shape[3], train_set.shape[4]) #[10, x, 2, 256, 256] --> [10*x*2, 256, 256]
                # Resize validation set ([3, 2, x, 256, 256] --> [3*2*x, 256, 256])
                val_set = val_set.transpose(0, 2, 1, 3, 4) #([10, 2, x, 256, 256] --> [10, x, 2, 256, 256])
                val_set = val_set.reshape(val_set.shape[0] * val_set.shape[1] * val_set.shape[2], val_set.shape[3], val_set.shape[4]) #[10, x, 2, 256, 256] --> [10*x*2, 256, 256]
            if self.auxinf:
                # Get training and validation set of images
                train_set_images = data_train_images[train]
                val_set_images = data_train_images[test]
                # Reshape data to ensure that all image augmentations of an image and the corresponding image (e.g. left hand and right hand) are assigned to the same split
                # Resize training set ([10, 2, x, 256, 256] --> [10*2*x, 256, 256])
                train_set_images = train_set_images.transpose(0, 2, 1, 3, 4) #([10, 2, x, 256, 256] --> [10, x, 2, 256, 256])
                train_set_images = train_set_images.reshape(train_set_images.shape[0] * train_set_images.shape[1] * train_set_images.shape[2], train_set_images.shape[3], train_set_images.shape[4]) #[10, x, 2, 256, 256] --> [10*x*2, 256, 256]
                # Resize validation set ([3, 2, x, 256, 256] --> [3*2*x, 256, 256])
                val_set_images = val_set_images.transpose(0, 2, 1, 3, 4) #([10, 2, x, 256, 256] --> [10, x, 2, 256, 256])
                val_set_images = val_set_images.reshape(val_set_images.shape[0] * val_set_images.shape[1] * val_set_images.shape[2],  val_set_images.shape[3], val_set_images.shape[4]) #[10, x, 2, 256, 256] --> [10*x*2, 256, 256]

                # Get training and validation set of segmentations
                train_set_segmentations = data_train_segmentations[train]
                val_set_segmentations = data_train_segmentations[test]
                # Reshape data to ensure that all augmentations and segmentations of an image and the corresponding image (e.g. left hand and right hand) are assigned to the same split
                # Resize training set ([10, 2, x, 19, 256, 256] --> [10*2*x*19, 256, 256])
                train_set_segmentations = train_set_segmentations.transpose(0, 2, 1, 3, 4, 5) #([10, 2, x, 19, 256, 256] --> [10, x, 2, 19, 256, 256])
                train_set_segmentations = train_set_segmentations.reshape(train_set_segmentations.shape[0] * train_set_segmentations.shape[1] * train_set_segmentations.shape[2] * train_set_segmentations.shape[3],
                                                                          train_set_segmentations.shape[4], train_set_segmentations.shape[5]) #[10, x, 19, 2, 256, 256] --> [10*x*2*19, 256, 256]
                # Resize validation set ([3, 2, x, 256, 256] --> [3*2*x, 256, 256])
                val_set_segmentations = val_set_segmentations.transpose(0, 2, 1, 3, 4, 5) #([3, 2, x, 19, 256, 256] --> [3, x, 2, 19, 256, 256])
                val_set_segmentations = val_set_segmentations.reshape(val_set_segmentations.shape[0] * val_set_segmentations.shape[1] * val_set_segmentations.shape[2] * val_set_segmentations.shape[3], 
                                                                      val_set_segmentations.shape[4], val_set_segmentations.shape[5]) #[3, x, 19, 2, 256, 256] --> [3*x*2*19, 256, 256]

                train_set_segmentations = train_set_segmentations.transpose(2, 1, 0) # (3458, 256, 256) --> (256, 256, 3458)
                val_set_segmentations = val_set_segmentations.transpose(2, 1, 0) # (182, 256, 256) --> (256, 256, 182)

                train_set = [train_set_images, train_set_segmentations]
                val_set = [val_set_images, val_set_segmentations]

            # Create training and validation generator
            if not self.auxinf:
                train_generator = vxm_data_generator(train_set, batch_size = batchsize_train)
                val_generator = vxm_data_generator(val_set, batch_size = batchsize_val)
                nb_train_samples = train_set.shape[0]
                nb_val_samples = val_set.shape[0]
            if self.auxinf:
                train_generator = vxm_data_generator(train_set, batch_size = batchsize_train, aux_information = True)
                val_generator = vxm_data_generator(val_set, batch_size = batchsize_val, aux_information = True)
                nb_train_samples = train_set[0].shape[0]
                nb_val_samples = val_set[0].shape[0]

            # Initalize hyperparameters
            nb_steps_epoch = nb_train_samples // batchsize_train

            # Save model
            checkpoint_path = modelpath + f"cv{index_cross_validation}/" + modelname + f"_{index_cross_validation}" + ".ckpt"

            # Create a callback that saves the model's weights at the end of all epochs
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path, 
                                                            save_weights_only = True, 
                                                            verbose = 1, 
                                                            save_freq = 'epoch', 
                                                            period = nb_epochs,
                                                            monitor = "val_loss",
                                                            save_best_only = True)

            # Save the weights 
            self.vxm_model.save_weights(checkpoint_path)

            # start training
            hist = self.vxm_model.fit(train_generator, epochs = nb_epochs, steps_per_epoch = nb_steps_epoch, 
                                verbose = 2, # verbose = 1 for progress bar. 
                                callbacks = [cp_callback],
                                validation_data = val_generator, 
                                validation_batch_size = validation_batch_size,
                                validation_steps = 1, 
                                validation_freq = 1 # should be equal to frequency of training loss as it can't be plotted otherwise
                                ) 

            # Show the dictionary of the training history
            # print(hist.history)

            # Save Loss values
            # plot_loss(hist, checkpoint_path[:-5] + f"_loss_history_run_{index_cross_validation}")

            crossvalidation_loss.append(hist)

            print(dataset_path)
            print(dataset_input_data)
            print(dataset_body_part)

            # Do inference
            self.test(test_dir = test_dir,
                    modelpath = modelpath + f"cv{index_cross_validation}/",
                    modelname = modelname + f"_{index_cross_validation}",
                    dataset_path = dataset_path,
                    dataset_input_data = dataset_input_data,
                    dataset_body_part = dataset_body_part)

            # Do the evaluation
            evaluator = Evaluation(sizes = self.sizes,
                                    path_original_fixed_segmentations = path_original_fixed_segmentations,
                                    path_predicted_segmentations = modelpath + f"cv{index_cross_validation}/segmentations/",
                                    auxinf = self.auxinf)

            landmarks_fixed_segmentations, landmarks_predicted_segmentations = evaluator.create_landmarks()

            euclidean_distances = evaluator.euclidean_distances(landmarks_fixed_segmentations, landmarks_predicted_segmentations)
            dice_scores = evaluator.dice_score()
            iou = evaluator.intersection_over_union()

            summary, summary_statistics = evaluator.summary()

            saving_location = modelpath + f"cv{index_cross_validation}/"

            summary.to_csv(saving_location + "summary.csv", encoding='utf-8', index=False)
            summary_statistics.to_csv(saving_location + "statistics.csv", encoding='utf-8')

            index_cross_validation += 1
        
        # plot_loss_all_kfold(crossvalidation_loss, modelpath + modelname + "_all_loss_history")

        print("-" * 30)
        print(f"Training with Crossvalidation with {cross_split} splits and {str(nb_epochs)} epochs finished.")
        print("-" * 30)


    def test(self,
            test_dir,
            modelpath,
            modelname,
            dataset_path,
            dataset_input_data,
            dataset_body_part):
        """ Test the semi-supervised or unsupervised model of VoxelMorph with given parameters.

        Args:
            test_dir ([str]): Select the test directory.
            modelpath ([str]): Select the modelpath (for loading the model).
            modelname ([str]): Select the modelname (for loading the model).
            dataset_path ([str]): Select the path to the original data.
            dataset_input_data ([str]): Select the version of the input data, e.g. bones.
            dataset_body_part ([str]): Select the version of the body part, e.g. hands flat.
        """

        print("-" * 30)
        print(f"Inference is starting now.")
        print("-" * 30)

        # Load weights for the model
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.vxm_model.load_weights(modelpath + modelname + ".ckpt")

        # Load the test data
        if self.auxinf: 
            data_test_ = load_data(test_dir, resize = self.sizes, data_augmentation = [], aux_information = True)
            data_test = [data_test_.image_data, data_test_.segmentation_data]
        elif not self.auxinf: data_test = load_data(test_dir, resize = self.sizes, data_augmentation = []).image_data

        # iterate to get and iterate through every image filename in the test directory
        for i, file_name in enumerate(sorted(glob.glob(test_dir + "images/" + "*"))):
            # Set the test data generator
            if self.auxinf: test_generator = vxm_data_generator(data_test, train = False, aux_information = True)
            elif not self.auxinf: test_generator = vxm_data_generator(data_test, train = False)
                
            # only take images of the left side for saving location and skip all images of the right side
            if "_rotated_" in file_name: continue           

            print(file_name)

            # get the save location depending on the variant, path, etc.
            save_name_figure, save_name_flow = get_save_location(modelpath = modelpath, 
                                                                file_name = file_name, 
                                                                losses = self.loss_functions, 
                                                                args_input_data = dataset_input_data, 
                                                                args_body_part = dataset_body_part,
                                                                auxinf = self.auxinf)

            # get filename separately
            path_parts = file_name.split("/")[-1].split("_")
            # e.g. data_padding/dataset_hands_flat/dataset_bones/test/images/R_02_1_l_Hand_Bones_padding.png -> R_02_1_l_
            file_name = ""
            for nb_part in range(len(path_parts)):
                if path_parts[nb_part] == "Hand" or path_parts[nb_part] == "Feet":
                    break
                file_name += path_parts[nb_part] + "_"

            # Get original segmentations for the current picture as array
            original_segmentations = sorted(glob.glob(test_dir + "segmentations/" + "*")) # get list of all segmentations
            original_segmentations = [k for k in original_segmentations if not "rotated" in k] # dont take right rotated segmentations, take the left ones
            original_segmentations = [k for k in original_segmentations if file_name in k] # search for correct segmentations 
            original_segmentations = create_segmentation_array(original_segmentations, size = self.sizes)

            # Get the validation file
            test_input, _ = next(test_generator)

            # Use the trained model to predict on unseen (test) data
            test_pred = self.vxm_model.predict(test_input)

            # Only moved and fixed images are delivered (no segmentations for the training process)
            if not self.auxinf: 
                # Extract informations from the prediction
                # Get sample, prediction, flow images
                images = [img[0, :, :, 0] for img in test_input + test_pred]
                # Get difference image 
                difference_img = images[1] - images[0]
                images.append(difference_img) # append the difference image to the list of images for the final figure
                # Get predicted image
                prediction_image = images[2][..., np.newaxis]

                # Set titles for final figure
                titles = ['moving', 'fixed', 'moved', 'flow', 'difference']
                # Save the final figure
                save_figure(images, titles, title = save_name_figure)

                # Get flow and save it as a picture of vectors
                flow_figure = flow([test_pred[1].squeeze()[::3, ::3]], width=5)
                flow_figure[0].savefig(save_name_flow)

                # Warp the segmentations with the model and save them for comparison with auxinf model
                warp_model = vxm.networks.Transform((self.sizes), interp_method = "linear")

                pred_segmentations = np.zeros((1, *self.sizes, original_segmentations.shape[-1]))
                for nb_seg in range(original_segmentations.shape[-1]):
                    current_segmentation = original_segmentations[:, :, nb_seg] # [256, 256, 19] -> [256, 256]
                    current_pred_segmentation = warp_model.predict([current_segmentation[np.newaxis, ..., np.newaxis], test_pred[1]]).squeeze(axis = -1)

                    pred_segmentations[:, :, :, nb_seg] = current_pred_segmentation
                
                # Get & Save predicted segmentations
                predicted_segmentations = save_predicted_segmentations(modelpath = modelpath, 
                                                                        segmentations = pred_segmentations, 
                                                                        file_name = file_name, 
                                                                        args_path = dataset_path, 
                                                                        args_input_data = dataset_input_data,
                                                                        auxinf = self.auxinf)
                
                # Save the predicted image
                save_predicted_image(modelpath = modelpath, 
                                    image = prediction_image, 
                                    file_name = file_name[:-1], 
                                    args_path = dataset_path, 
                                    args_input_data = dataset_input_data,
                                    auxinf = self.auxinf)

                # save segmentation figure
                save_name_figure = save_name_figure[:-4] + "_segmentations.svg"
                save_segmentation_figure(modelpath = modelpath, 
                                        orig_segmentations = original_segmentations, 
                                        segmentations = predicted_segmentations,
                                        args_path = dataset_path, 
                                        args_input_data = dataset_input_data, 
                                        title = save_name_figure)

                # get next pictures
                data_test = data_test[2:,:,:]

            # Auxiliary information (segmentations) are delivered when the model was trained
            if self.auxinf:
                # Extract informations from the prediction
                # Get predicted image
                prediction_image = np.squeeze(test_pred[0], axis = 0)
                # Get the predicted flow
                prediction_flow = test_pred[1][0, :, :, 0]
                # Get difference image between moving and fixed
                difference_img = test_input[1] - test_input[0]

                # Get & Save predicted segmentations
                predicted_segmentations = save_predicted_segmentations(modelpath = modelpath, 
                                                                        segmentations = test_pred[2], 
                                                                        file_name = file_name,
                                                                        args_path = dataset_path, 
                                                                        args_input_data = dataset_input_data,
                                                                        auxinf = self.auxinf)
                # Save the predicted image
                save_predicted_image(modelpath = modelpath, 
                                    image = prediction_image, 
                                    file_name = file_name[:-1], 
                                    args_path = dataset_path, 
                                    args_input_data = dataset_input_data,
                                    auxinf = self.auxinf)
                

                # All necessary images in an array for the final figure
                images = [test_input[0], test_input[1], prediction_image, prediction_flow, difference_img] # moving and fixed image, moved image and flow picture
                # Set titles for final figure
                titles = ['moving', 'fixed', 'moved', 'flow', 'difference']
                
                # Save the final figure
                save_figure(images, titles, title = save_name_figure)
                # Get flow and save it as a picture of vectors
                flow_figure = flow([test_pred[1].squeeze()[::3, ::3]], width=15, grid=True)
                flow_figure[0].savefig(save_name_flow)
                # save segmentation figure
                save_name_figure = save_name_figure[:-4] + "_segmentations.svg"
                save_segmentation_figure(modelpath = modelpath,
                                        orig_segmentations = original_segmentations, 
                                        segmentations = predicted_segmentations,
                                        args_path = dataset_path, 
                                        args_input_data = dataset_input_data, 
                                        title = save_name_figure)

                # get next pictures
                data_test = [data_test[0][2:], data_test[1][:, :, 38:]]


class Augmentation():
    """ 
        Class for applying data augmentation.
    """
    def __init__(self, image_data, segmentation_data = None):
        """ Initalize shapes and variables.

        Args:
            image_data ([numpy array]): Shape: (nb_images, size, size), e.g. (40, 256, 256)
            segmentation_data ([numpy array]): Shape: (size, size, nb_segmentations)
        """
        # image sizes
        self.sizes = (image_data.shape[1], image_data.shape[2])
        # list of images
        self.image_data = image_data
        # number of images before augmentation
        self.nb_images = image_data.shape[0]
        # list of auxinf images (segmentations)
        self.segmentation_data = segmentation_data

        if segmentation_data is not None: 
            self.nb_segmentations = segmentation_data.shape[-1]
            self.nb_bones = int(self.nb_segmentations/self.nb_images)

    def show_example_augmentation(self, augmentation, augmentation_variant, save_dir = "outputs/example_augmentations"):
        """ Generate an example of the augmentation to be sure what the augmentation does with the picture.

        Args:
            augmentation ([numpy array]): Image which was augmented.
            augmentation_variant ([type]): Augmentation variant is important for the filename.
        """
        example_image = Image.fromarray(augmentation * 255).convert("RGB")

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        example_image.save(f"{save_dir}/example_image_{augmentation_variant}.png")

    def rotate(self, degree):
        """ Rotate the image and/or segmentation data about x degree. 

        Args:
            degree ([int]): Amount of degrees for rotating the image.

        Returns:
            [image_data]: Augmented image data.
                or 
            [image_data, segmentation_data]: Original and Augmented image data and Original and Augmented segmentation data.
        """

        # get new number of images and segmentations
        new_nb_images = self.nb_images * 2
        if self.segmentation_data is not None:
            new_nb_segmentations = self.nb_segmentations * 2

        # create new data and segmentation arrays
        augmented_image_data = np.zeros(shape = (new_nb_images, *self.sizes))
        if self.segmentation_data is not None:
            augmented_segmentation_data = np.zeros(shape = (*self.sizes, new_nb_segmentations))

        # fill the first part of the array with the original data
        augmented_image_data[0 : self.nb_images, :, :] = self.image_data 
        if self.segmentation_data is not None: 
            augmented_segmentation_data[:, :, : self.nb_segmentations] = self.segmentation_data 

        # get augmentation details
        image_center = tuple(np.array(self.image_data[0, :, :].shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, degree, 1.0)

        # fill the second part of the array with the augmented data
        # augment image data
        for nb_data in range(self.nb_images, new_nb_images):
            current_augmentation = self.image_data[nb_data - self.nb_images, :, :]
            # rotate image
            current_augmentation = cv2.warpAffine(current_augmentation, rot_mat, current_augmentation.shape[1::-1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            augmented_image_data[nb_data, :, :] = current_augmentation
            self.show_example_augmentation(augmentation = current_augmentation, augmentation_variant = "rotated")
        # augment segmentation data
        if self.segmentation_data is not None: 
            for nb_segmentation in range(self.nb_segmentations, new_nb_segmentations): 
                current_augmentation = self.segmentation_data[:, :, nb_segmentation - self.nb_segmentations]
                # rotate segmentation
                current_augmentation = cv2.warpAffine(current_augmentation, rot_mat, current_augmentation.shape[1::-1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                # convert to binary segmentation
                current_augmentation = np.where(current_augmentation > 0.1, 1.0, 0)
                augmented_segmentation_data[:, :, nb_segmentation] = current_augmentation
                self.show_example_augmentation(augmentation = current_augmentation, augmentation_variant = "rotated_segmentation")

        # return only augmented data (and not also the original images)
        if self.segmentation_data is not None: 
            return augmented_image_data[self.nb_images : new_nb_images, :, :], augmented_segmentation_data[:, :, self.nb_segmentations : new_nb_segmentations]

        return augmented_image_data[self.nb_images : new_nb_images, :, :]

    def translate(self, amount = -100):
        """ Translate the image with a certain amount. 

        Args:
            amount (int, optional): Amount of and direction of the translation.
                                    Defaults to -100.

        Returns:
            [image_data]: Augmented image data.
                or 
            [image_data, segmentation_data]: Original and Augmented image data and Original and Augmented segmentation data.
        """
        # get new number of images and segmentations
        new_nb_images = self.nb_images * 2
        if self.segmentation_data is not None:
            new_nb_segmentations = self.nb_segmentations * 2

        # create new data and segmentation arrays
        augmented_image_data = np.zeros(shape = (new_nb_images, *self.sizes))
        if self.segmentation_data is not None:
            augmented_segmentation_data = np.zeros(shape = (*self.sizes, new_nb_segmentations))

        # fill the first part of the array with the original data
        augmented_image_data[0 : self.nb_images, :, :] = self.image_data 
        if self.segmentation_data is not None: 
            augmented_segmentation_data[:, :, : self.nb_segmentations] = self.segmentation_data 

        # fill the second part of the array with the augmented data
        # augment image data
        for nb_data in range(self.nb_images, new_nb_images):
            current_augmentation = self.image_data[nb_data - self.nb_images, :, :]
            current_augmentation = np.roll(current_augmentation, amount)
            augmented_image_data[nb_data, :, :] = current_augmentation
            self.show_example_augmentation(augmentation = current_augmentation, augmentation_variant = "translated")
        # augment segmentation data
        if self.segmentation_data is not None: 
            for nb_segmentation in range(self.nb_segmentations, new_nb_segmentations): 
                current_augmentation = self.segmentation_data[:, :, nb_segmentation - self.nb_segmentations]
                current_augmentation = np.roll(current_augmentation, amount)
                # convert to binary segmentation
                current_augmentation = np.where(current_augmentation > 0.1, 1.0, 0)
                augmented_segmentation_data[:, :, nb_segmentation] = current_augmentation
                self.show_example_augmentation(augmentation = current_augmentation, augmentation_variant = "translated_segmentierung")

        # return only augmented data (and not also the original images)
        if self.segmentation_data is not None: 
            return augmented_image_data[self.nb_images : new_nb_images, :, :], augmented_segmentation_data[:, :, self.nb_segmentations : new_nb_segmentations]

        return augmented_image_data[self.nb_images : new_nb_images, :, :]

    def higher_brightness(self, level):
        """ Increase the brightness of the image and/or segmentation data about x degree. 

        Args:
            level ([int]): Amount of level for increasing  the brightness.

        Returns:
            [image_data]: Augmented image data.
                or 
            [image_data, segmentation_data]: Original and Augmented image data and Original and Augmented segmentation data.
        """
        # get new number of images and segmentations
        new_nb_images = self.nb_images * 2
        if self.segmentation_data is not None:
            new_nb_segmentations = self.nb_segmentations * 2

        # create new data and segmentation arrays
        augmented_image_data = np.zeros(shape = (new_nb_images, *self.sizes))
        if self.segmentation_data is not None:
            augmented_segmentation_data = np.zeros(shape = (*self.sizes, new_nb_segmentations))

        # fill the first part of the array with the original data
        augmented_image_data[0 : self.nb_images, :, :] = self.image_data 
        if self.segmentation_data is not None: 
            augmented_segmentation_data[:, :, : self.nb_segmentations] = self.segmentation_data 

        # fill the second part of the array with the augmented data
        # augment image data
        for nb_data in range(self.nb_images, new_nb_images):
            current_augmentation = self.image_data[nb_data - self.nb_images, :, :]
            #current_augmentation = np.where((current_augmentation < 0.5) & (current_augmentation > 0.3), current_augmentation + level, current_augmentation)
            current_augmentation = current_augmentation + level 
            current_augmentation = (current_augmentation - np.min(current_augmentation))/np.ptp(current_augmentation)

            augmented_image_data[nb_data, :, :] = current_augmentation
            
            self.show_example_augmentation(augmentation = current_augmentation, augmentation_variant = "higher_brightness")

        # augment segmentation data
        if self.segmentation_data is not None: 
            for nb_segmentation in range(self.nb_segmentations, new_nb_segmentations): 
                current_augmentation = self.segmentation_data[:, :, nb_segmentation - self.nb_segmentations]
                #current_augmentation = np.where((current_augmentation < 0.5) & (current_augmentation > 0.3), current_augmentation + level, current_augmentation)
                current_augmentation = current_augmentation + level 
                # convert to binary segmentation
                current_augmentation = np.where(current_augmentation > 0.1, 1.0, 0)
                # current_augmentation = (current_augmentation - np.min(current_augmentation))/np.ptp(current_augmentation)
                
                augmented_segmentation_data[:, :, nb_segmentation] = current_augmentation

        # return only augmented data (and not also the original images)
        if self.segmentation_data is not None: 
            return augmented_image_data[self.nb_images : new_nb_images, :, :], augmented_segmentation_data[:, :, self.nb_segmentations : new_nb_segmentations]

        return augmented_image_data[self.nb_images : new_nb_images, :, :]

    def lower_brightness(self, level):
        """ Decrease the brightness of the image and/or segmentation data about x degree. 

        Args:
            level ([int]): Amount of level for increasing  the brightness.

        Returns:
            [image_data]: Augmented image data.
                or 
            [image_data, segmentation_data]: Original and Augmented image data and Original and Augmented segmentation data.
        """
        # get new number of images and segmentations
        new_nb_images = self.nb_images * 2
        if self.segmentation_data is not None:
            new_nb_segmentations = self.nb_segmentations * 2

        # create new data and segmentation arrays
        augmented_image_data = np.zeros(shape = (new_nb_images, *self.sizes))
        if self.segmentation_data is not None:
            augmented_segmentation_data = np.zeros(shape = (*self.sizes, new_nb_segmentations))

        # fill the first part of the array with the original data
        augmented_image_data[0 : self.nb_images, :, :] = self.image_data 
        if self.segmentation_data is not None: 
            augmented_segmentation_data[:, :, : self.nb_segmentations] = self.segmentation_data 

        # fill the second part of the array with the augmented data
        # augment image data
        for nb_data in range(self.nb_images, new_nb_images):
            current_augmentation = self.image_data[nb_data - self.nb_images, :, :]
            #current_augmentation = np.where((current_augmentation > 0.8), current_augmentation - level, current_augmentation)
            current_augmentation = current_augmentation - level
            current_augmentation = (current_augmentation - np.min(current_augmentation))/np.ptp(current_augmentation)
                        
            augmented_image_data[nb_data, :, :] = current_augmentation
            
            self.show_example_augmentation(augmentation = current_augmentation, augmentation_variant = "lower_brightness")

        # augment segmentation data
        if self.segmentation_data is not None: 
            for nb_segmentation in range(self.nb_segmentations, new_nb_segmentations): 
                current_augmentation = self.segmentation_data[:, :, nb_segmentation - self.nb_segmentations]
                #current_augmentation = np.where((current_augmentation > 0.8), current_augmentation - level, current_augmentation)
                current_augmentation = current_augmentation - level
                # convert to binary segmentation
                current_augmentation = np.where(current_augmentation > 0.1, 1.0, 0)
                augmented_segmentation_data[:, :, nb_segmentation] = current_augmentation

        # return only augmented data (and not also the original images)
        if self.segmentation_data is not None: 
            return augmented_image_data[self.nb_images : new_nb_images, :, :], augmented_segmentation_data[:, :, self.nb_segmentations : new_nb_segmentations]

        return augmented_image_data[self.nb_images : new_nb_images, :, :]

    def b_spline_transform(self, gridsize = (2, 3, 3)):
        """ Transform the image data with a 2x2 grid and b splines.

            Source: https://github.com/tueimage/gryds
            Paper: K. A. J. Eppenhof, & J. P. W. Pluim (2019). 
                    Pulmonary CT Registration through Supervised Learning with Convolutional Neural Networks. 
                    IEEE Transactions on Medical Imaging, 38(5), 1097-1105.

        Returns:
            [image_data]: Augmented image data.
                or 
            [image_data, segmentation_data]: Original and Augmented image data and Original and Augmented segmentation data.
        """
        # get new number of images and segmentations
        new_nb_images = self.nb_images * 2
        if self.segmentation_data is not None:
            new_nb_segmentations = self.nb_segmentations * 2

        # create new data and segmentation arrays
        augmented_image_data = np.zeros(shape = (new_nb_images, *self.sizes))
        if self.segmentation_data is not None:
            augmented_segmentation_data = np.zeros(shape = (*self.sizes, new_nb_segmentations))

        # fill the first part of the array with the original data
        augmented_image_data[0 : self.nb_images, :, :] = self.image_data 
        if self.segmentation_data is not None: 
            augmented_segmentation_data[:, :, : self.nb_segmentations] = self.segmentation_data 

        # get augmentation details 
        # Define a random 3x3 B-spline grid for a 2D image:
        # random_grid = (np.random.rand(2, 3, 3) - 0.5) / 5
        random_grid = np.random.randint(low = 0, high = 4, size = gridsize) / 100
        # Define a B-spline transformation object
        bspline = gryds.BSplineTransformation(random_grid)

        # fill the second part of the array with the augmented data
        # augment image data
        for nb_data in range(self.nb_images, new_nb_images):
            current_augmentation = self.image_data[nb_data - self.nb_images, :, :]
            
            # Define an interpolator object for the image:
            interpolator = gryds.Interpolator(current_augmentation)
            # Transform the image using the B-spline transformation
            current_augmentation = interpolator.transform(bspline)
            
            augmented_image_data[nb_data, :, :] = current_augmentation
            self.show_example_augmentation(augmentation = current_augmentation, augmentation_variant = "bsplines")

        # augment segmentation data
        if self.segmentation_data is not None: 
            for nb_segmentation in range(self.nb_segmentations, new_nb_segmentations): 
                current_augmentation = self.segmentation_data[:, :, nb_segmentation - self.nb_segmentations]

                # Define an interpolator object for the image:
                interpolator = gryds.Interpolator(current_augmentation)
                # Transform the image using the B-spline transformation
                current_augmentation = interpolator.transform(bspline)
                # convert to binary segmentation
                current_augmentation = np.where(current_augmentation > 0.1, 1.0, 0)
                augmented_segmentation_data[:, :, nb_segmentation] = current_augmentation
                self.show_example_augmentation(augmentation = current_augmentation, augmentation_variant = "bsplines_segmentierung")

        # normalization necessary
        augmented_image_data = (augmented_image_data - np.min(augmented_image_data)) / (np.max(augmented_image_data) - np.min(augmented_image_data)) 
        if self.segmentation_data is not None: 
            augmented_segmentation_data = (augmented_segmentation_data - np.min(augmented_segmentation_data)) / (np.max(augmented_segmentation_data) - np.min(augmented_segmentation_data)) 

        # return only augmented data (and not also the original images)
        if self.segmentation_data is not None: 
            return augmented_image_data[self.nb_images : new_nb_images, :, :], augmented_segmentation_data[:, :, self.nb_segmentations : new_nb_segmentations]

        return augmented_image_data[self.nb_images : new_nb_images, :, :]

    def combine_augmentations(self, degree, amount, gridsize):
        """ Combines rotation, translation and b-spline transformation to generate an additional augmented image.

        Args:
            degree ([int]): Amount of degrees for rotating the image
            amount (int, optional): Amount of and direction of the translation.
                                    Defaults to -100.
            gridsize ([int, int, int]): grid size for b spline transfomation

        Returns:
            [image_data]: Augmented image data.
                or 
            [image_data, segmentation_data]: Original and Augmented image data and Original and Augmented segmentation data.
        """
        # get new number of images and segmentations
        new_nb_images = self.nb_images * 2
        if self.segmentation_data is not None:
            new_nb_segmentations = self.nb_segmentations * 2

        # create new data and segmentation arrays
        augmented_image_data = np.zeros(shape = (new_nb_images, *self.sizes))
        if self.segmentation_data is not None:
            augmented_segmentation_data = np.zeros(shape = (*self.sizes, new_nb_segmentations))

        # fill the first part of the array with the original data
        augmented_image_data[0 : self.nb_images, :, :] = self.image_data 
        if self.segmentation_data is not None: 
            augmented_segmentation_data[:, :, : self.nb_segmentations] = self.segmentation_data 

        # get augmentation details 
        # Define a random 3x3 B-spline grid for a 2D image:
        # random_grid = (np.random.rand(2, 3, 3) - 0.5) / 5
        random_grid = np.random.randint(low = 0, high = 4, size = gridsize) / 100
        # Define a B-spline transformation object
        bspline = gryds.BSplineTransformation(random_grid)

        # get augmentation details
        image_center = tuple(np.array(self.image_data[0, :, :].shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, degree, 1.0)

        # fill the second part of the array with the augmented data
        # augment image data
        for nb_data in range(self.nb_images, new_nb_images):
            current_augmentation = self.image_data[nb_data - self.nb_images, :, :]
            
            # Define an interpolator object for the image:
            interpolator = gryds.Interpolator(current_augmentation)
            # Transform the image using the B-spline transformation
            current_augmentation = interpolator.transform(bspline)

            # translate 
            current_augmentation = np.roll(current_augmentation, amount)

            # rotate
            current_augmentation = cv2.warpAffine(current_augmentation, rot_mat, current_augmentation.shape[1::-1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            
            augmented_image_data[nb_data, :, :] = current_augmentation
            self.show_example_augmentation(augmentation = current_augmentation, augmentation_variant = "combinations")

        # augment segmentation data
        if self.segmentation_data is not None: 
            for nb_segmentation in range(self.nb_segmentations, new_nb_segmentations): 
                current_augmentation = self.segmentation_data[:, :, nb_segmentation - self.nb_segmentations]

                # Define an interpolator object for the image:
                interpolator = gryds.Interpolator(current_augmentation)
                # Transform the image using the B-spline transformation
                current_augmentation = interpolator.transform(bspline)

                # translate 
                current_augmentation = np.roll(current_augmentation, amount)

                # rotate
                current_augmentation = cv2.warpAffine(current_augmentation, rot_mat, current_augmentation.shape[1::-1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

                # convert to binary segmentation
                current_augmentation = np.where(current_augmentation > 0.1, 1.0, 0)
                augmented_segmentation_data[:, :, nb_segmentation] = current_augmentation
                self.show_example_augmentation(augmentation = current_augmentation, augmentation_variant = "combinations_segmentierung")

        # normalization necessary
        augmented_image_data = (augmented_image_data - np.min(augmented_image_data)) / (np.max(augmented_image_data) - np.min(augmented_image_data)) 
        if self.segmentation_data is not None: 
            augmented_segmentation_data = (augmented_segmentation_data - np.min(augmented_segmentation_data)) / (np.max(augmented_segmentation_data) - np.min(augmented_segmentation_data)) 

        # return only augmented data (and not also the original images)
        if self.segmentation_data is not None: 
            return augmented_image_data[self.nb_images : new_nb_images, :, :], augmented_segmentation_data[:, :, self.nb_segmentations : new_nb_segmentations]

        return augmented_image_data[self.nb_images : new_nb_images, :, :]

    def augment(self, 
                augmentation_techniques = [], 
                comb_params = {"degree" : [5, 20, 40], "amount" : [-100, -50, -20], "gridsize" : [(2, 3, 3), (2, 4, 4), (2, 5, 5)]}
                ):
        """Performs the augmentation of the images using a selection of augmentation techniques.

        Args:
            augmentation_techniques (list of strings, optional): Define which augmentation techniques are used.
            comb_params: List of parameters for the augmentaion using rotation, translation and b-spline transformation combined
                            Default: "degree" : [5, 20, 40], "amount" : [-100, -50, -20], "gridsize" : [(2, 3, 3), (2, 4, 4), (2, 5, 5)]

        Returns:
            [image_data]: Augmented image data.
                or 
            [image_data, segmentation_data]: Original and Augmented image data and Original and Augmented segmentation data.

        """

        augmented_images = []
        augmented_segmentations = []

        if "rotate" in augmentation_techniques:
            if self.segmentation_data is not None:
                augmented_image_data, augmented_segmentation_data = self.rotate(degree = 5)
                augmented_images.append(augmented_image_data)
                augmented_segmentations.append(augmented_segmentation_data)
            else:
                augmented_image_data = self.rotate(degree = 20)
                augmented_images.append(augmented_image_data)

        if "lower_brightness" in augmentation_techniques:
            # In some experiments, this made the results worse
            if self.segmentation_data is not None:
                augmented_image_data, augmented_segmentation_data = self.lower_brightness(level = 0.2)
                augmented_images.append(augmented_image_data)
                augmented_segmentations.append(augmented_segmentation_data)
            else:
                augmented_image_data = self.lower_brightness(level = 0.2)
                augmented_images.append(augmented_image_data)

        if "higher_brightness" in augmentation_techniques:
            # In some experiments, this made the results worse
            if self.segmentation_data is not None:
                augmented_image_data, augmented_segmentation_data = self.higher_brightness(level = 0.2)
                augmented_images.append(augmented_image_data)
                augmented_segmentations.append(augmented_segmentation_data)
            else:
                augmented_image_data = self.higher_brightness(level = 0.2)
                augmented_images.append(augmented_image_data)

        if "translate" in augmentation_techniques:
            if self.segmentation_data is not None:
                augmented_image_data, augmented_segmentation_data = self.translate(amount = -100)
                augmented_images.append(augmented_image_data)
                augmented_segmentations.append(augmented_segmentation_data)
            else:
                augmented_image_data = self.translate(amount = -100)
                augmented_images.append(augmented_image_data)

        if "b_spline" in augmentation_techniques:
            if self.segmentation_data is not None:
                augmented_image_data, augmented_segmentation_data = self.b_spline_transform()
                augmented_images.append(augmented_image_data)
                augmented_segmentations.append(augmented_segmentation_data)
            else:
                augmented_image_data = self.b_spline_transform()
                augmented_images.append(augmented_image_data)

        if "combine" in augmentation_techniques:
            # number of augmentations due to a combination of augmentation techniques
            nb_combine_iterations = len(comb_params.get("degree"))

            # performs several augmentations using a combination of b-spline transformation, rotation and translation
            for i in range(nb_combine_iterations):
                current_degree = comb_params.get("degree")[i]
                current_amount = comb_params.get("amount")[i]
                current_gridsize = comb_params.get("gridsize")[i]

                if self.segmentation_data is not None:
                    augmented_image_data, augmented_segmentation_data = self.combine_augmentations(gridsize=current_gridsize,
                                                                                                    degree = current_degree,
                                                                                                    amount = current_amount
                                                                                                    )

                    augmented_images.append(augmented_image_data)
                    augmented_segmentations.append(augmented_segmentation_data)
                else:
                    augmented_image_data = self.combine_augmentations(gridsize=current_gridsize,
                                                                        degree = current_degree,
                                                                        amount = current_amount
                                                                        )
                    augmented_images.append(augmented_image_data)

        for nb_augmentation in range(len(augmented_images)):
            if self.segmentation_data is not None:
                self.segmentation_data = np.append(self.segmentation_data, augmented_segmentations[nb_augmentation], axis = 2)
            self.image_data = np.append(self.image_data, augmented_images[nb_augmentation], axis = 0)

        if self.segmentation_data is not None:
            return self.image_data, self.segmentation_data
        else:
            return self.image_data


class Evaluation():
    """ 
        Class for evaluating the predicted results of VoxelMorph. For evaluating the results, the
        segmentations of the original image data are needed. The evaluation is based on placing
        Landmarks in the middle of every segmentation and further based on using the Dice score.

        Source (partly): 
            https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python 
    """

    def __init__(self,
                path_original_fixed_segmentations,
                path_predicted_segmentations,
                sizes = (256, 256),
                auxinf = False):
        """ Initialize the evaluator with given parameters.

        Args:
            path_original_fixed_segmentations ([str]): Path to the original segmentations of the fixed image.
            path_predicted_segmentations ([str]): Path to the predicted (moved) segmentations.
            sizes (tuple, optional): Size of the segmentations. 
                                    Defaults to (256, 256).
            auxinf (bool, optional): Decision if the semi-supervised or unsupervised model should be evaluated. 
                                    Defaults to False.
        """
        # Get shapes
        self.sizes = sizes

        # Get number of segmentations
        self.nb_segmentations = 19
        
        # Get list of all segmentations from the paths
        path_original_fixed_segmentations = sorted(glob.glob(path_original_fixed_segmentations + '*.png'))
        path_predicted_segmentations = sorted(glob.glob(path_predicted_segmentations + '*.png'))

        self.predicted_segmentation_names = path_predicted_segmentations

        # Sort out auxiliary segmentations if auxinf = False
        if auxinf == False: 
            path_predicted_segmentations = [k for k in path_predicted_segmentations if not "aux" in k]
        else: 
            path_predicted_segmentations = [k for k in path_predicted_segmentations if "aux" in k]

        path_original_fixed_segmentations = [k for k in path_original_fixed_segmentations if "rotated" in k]


        # Get number of different files (1 file = nb_segmentations segmentations)
        self.nb_files = int(len(path_original_fixed_segmentations) / self.nb_segmentations)

        # Get all original fixed segmentations and all predicted segmentations
        fixed_segmentations = np.zeros((self.nb_files, *self.sizes, self.nb_segmentations))
        predicted_segmentations = np.zeros((self.nb_files, *self.sizes, self.nb_segmentations))

        counter = 0
        for nb_file in range(self.nb_files):
            for nb_segmentation in range(self.nb_segmentations):
                # get the current segmentation of the fixed segmentations
                current_segmentation = np.array(Image.open(path_original_fixed_segmentations[counter]).resize(self.sizes, Image.NEAREST).convert("L"))
                
                fixed_segmentations[nb_file, :, :, nb_segmentation] = current_segmentation

                # get the current segmentation of the predicted segmentations
                current_segmentation = np.array(Image.open(path_predicted_segmentations[counter]).resize(self.sizes, Image.NEAREST).convert("L"))
                predicted_segmentations[nb_file, :, :, nb_segmentation] = current_segmentation

                counter += 1

        self.fixed_segmentations = fixed_segmentations
        self.predicted_segmentations = predicted_segmentations


    def create_landmarks(self):
        """ Create the landmarks based on the fixed and predicted segmentations. 

        Returns:
            [np array]: Landmarks (xy coordinates) with shape (number of files, 2, number of segmentations). 
        """

        self.landmarks_fixed_segmentations = np.zeros((self.nb_files, 2, self.nb_segmentations))
        self.landmarks_predicted_segmentations = np.zeros((self.nb_files, 2, self.nb_segmentations))

        for nb_file in range(self.nb_files):
            for nb_segmentation in range(self.nb_segmentations):

                # get the current fixed segmentation
                current_segmentation = self.fixed_segmentations[nb_file, :, :, nb_segmentation]
                # convert the grayscale segmentation to a binary one
                ret, thresh = cv2.threshold(current_segmentation, 127, 255, 0)
                # calculate moments of the binary segmentation image
                moments = cv2.moments(thresh)
                # calculate x and y coordinate of the center of the segmentation
                xy_coordinates = np.array([int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])])
                # write the landmark to the landmarks fixed segmentations array
                self.landmarks_fixed_segmentations[nb_file, :, nb_segmentation] = xy_coordinates

                # get the current predicted segmentation
                current_segmentation = self.predicted_segmentations[nb_file, :, :, nb_segmentation]
                # convert the grayscale segmentation to a binary one
                ret, thresh = cv2.threshold(current_segmentation, 127, 255, 0)
                # calculate moments of the binary segmentation image
                moments = cv2.moments(thresh)
                # calculate x and y coordinate of the center of the segmentation
                xy_coordinates = np.array([int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])])
                # write the landmark to the landmarks predicted segmentations array
                self.landmarks_predicted_segmentations[nb_file, :, nb_segmentation] = xy_coordinates
        
        return self.landmarks_fixed_segmentations, self.landmarks_predicted_segmentations


    def show_landmarks(self, landmarks_fixed_segmentations, landmarks_predicted_segmentations, save_dir = "outputs/example_landmarks"):
        """ Show the generated landmarks in an example image.

        Args:
            llandmarks_fixed_segmentations ([np array]): Landmarks (xy coordinates) with shape (number of files, 2, number of segmentations).
            landmarks_predicted_segmentations ([np array]): Landmarks (xy coordinates) with shape (number of files, 2, number of segmentations).
            save_dir (str, optional): Directory for saving the landmark images.
                                    Defaults to "outputs/example_landmarks".

        """

        # Get the number of files
        nb_files = self.fixed_segmentations.shape[0]
        # Get the number of segmentations
        nb_segmentations = self.fixed_segmentations.shape[3]

        # Create numpy arrays to merge the segmentations to one coherent segmentation
        whole_fixed_segmentations = np.zeros((nb_files, *self.sizes))
        whole_predicted_segmentations = np.zeros((nb_files, *self.sizes))

        # Set the segmentations of the fixed and the predicted ones together as one large segmentation image (one for fixed and one for predicted).
        for nb_file in range(nb_files):
            for nb_segmentation in range(nb_segmentations):
                indizes = self.fixed_segmentations[nb_file, :, :, nb_segmentation] != 0
                whole_fixed_segmentations[nb_file, indizes] = 255
        for nb_file in range(nb_files):
            for nb_segmentation in range(nb_segmentations):
                indizes = self.predicted_segmentations[nb_file, :, :, nb_segmentation] != 0
                whole_predicted_segmentations[nb_file, indizes] = 255
        
        # Change the landmarks to tuples
        tup_landmarks_fixed_segmentations = []
        tup_landmarks_predicted_segmentations = []

        for nb_file in range(nb_files):
            curr_tuples = []
            for nb_segmentation in range(nb_segmentations):
                current_tuple = (landmarks_fixed_segmentations[nb_file, 0, nb_segmentation], landmarks_fixed_segmentations[nb_file, 1, nb_segmentation])
                curr_tuples.append(current_tuple)
            tup_landmarks_fixed_segmentations.append(curr_tuples)
        for nb_file in range(nb_files):
            curr_tuples = []
            for nb_segmentation in range(nb_segmentations):
                current_tuple = (landmarks_predicted_segmentations[nb_file, 0, nb_segmentation], landmarks_predicted_segmentations[nb_file, 1, nb_segmentation])
                curr_tuples.append(current_tuple)

            tup_landmarks_predicted_segmentations.append(curr_tuples)

        # Create the save directory if it does not exist
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
            
        # Save the fixed segmentation with the landmarks in it
        frame_fixed = whole_fixed_segmentations[0, :, :] 
        for tuple_xy in tup_landmarks_fixed_segmentations[0]:
            cv2.circle(frame_fixed, (int(tuple_xy[0]), int(tuple_xy[1])), radius=3, color=(100,100,100), thickness=-3)
        cv2.imwrite(f"{save_dir}/example_image_fixed.png", frame_fixed)
        # Save the predicted segmentation with the landmarks in it
        frame_predicted = whole_predicted_segmentations[0, :, :] 
        for tuple_xy in tup_landmarks_predicted_segmentations[0]:
            cv2.circle(frame_predicted, (int(tuple_xy[0]), int(tuple_xy[1])), radius=3, color=(100,100,100), thickness=-3)
        cv2.imwrite(f"{save_dir}/example_image_predicted.png", frame_predicted)


    def euclidean_distances(self, landmarks_fixed_segmentations, landmarks_predicted_segmentations):
        """ Calculate the Euclidean distance between the fixed and predicted landmarks.

        Args:
            landmarks_fixed_segmentations ([np array]): Landmarks (xy coordinates) with shape (number of files, 2, number of segmentations).
            landmarks_predicted_segmentations ([np array]): Landmarks (xy coordinates) with shape (number of files, 2, number of segmentations).

        Returns:
            [np array]: Euclidean distances between the landmarks. Shape: (number of files, number of segmentations)
        """
        # Calculate the Euclidean distances between the two segmentation arrays
        self.euclidean_distances = np.linalg.norm(landmarks_fixed_segmentations - landmarks_predicted_segmentations, axis = 1)

        return self.euclidean_distances


    def dice_score(self):
        """ Calculcate the Dice score between the fixed and predicted segmentations. 

        Returns:
            [list]: All dice scores (every dice score for every pair of segmentations) as a list. 
        """

        self.dice_score_list = []

        for nb_file in range(self.nb_files):
            for nb_segmentation in range(self.nb_segmentations):

                numerator = np.sum(self.predicted_segmentations[nb_file, :, :, nb_segmentation][self.fixed_segmentations[nb_file, :, :, nb_segmentation] == 255]) * 2.0
                denominator = np.sum(self.predicted_segmentations[nb_file, :, :, nb_segmentation]) + np.sum(self.fixed_segmentations[nb_file, :, :, nb_segmentation])

                dice_score = round(numerator / denominator, 5)
                self.dice_score_list.append(dice_score)

        return self.dice_score_list

    
    def intersection_over_union(self):
        """ Calculate the intersection over union between the fixed and the predicted segmentations.

        """
        
        self.iou_list = []

        for nb_file in range(self.nb_files):
            for nb_segmentation in range(self.nb_segmentations):

                union = np.logical_or(self.predicted_segmentations[nb_file, :, :, nb_segmentation], self.fixed_segmentations[nb_file, :, :, nb_segmentation])
                intersection = np.logical_and(self.predicted_segmentations[nb_file, :, :, nb_segmentation], self.fixed_segmentations[nb_file, :, :, nb_segmentation])

                iou = round(np.sum(intersection) / np.sum(union), 5)
                self.iou_list.append(iou)


    def summary(self):
        """ Create a summary of the metrics used in the evaluation class for every segmentation and every file.

        Returns:
            [df]: Dataframe for the summary.
            [df]: Dataframe for the summary of statistics (mean, std, min, etc.).
        """

        segmentation_names = ["mc1", "mc2", "mc3", "mc4", "mc5",
                                "pd1", "pd2", "pd3", "pd4", "pd5", 
                                "pm2", "pm3", "pm4", "pm5", 
                                "pp1", "pp2", "pp3", "pp4", "pp5"]

        number_segmentation_names = len(segmentation_names)
        number_files = int(len(self.predicted_segmentation_names) / number_segmentation_names)

        for i in range(number_files-1):
            segmentation_names += segmentation_names[:number_segmentation_names]

        index_counter = 0
        for nb_name in range(len(segmentation_names)):
            if index_counter < len(segmentation_names):
                segmentation_names[nb_name] = "R_02_" + segmentation_names[nb_name]
            if index_counter > len(segmentation_names) - 1:
                segmentation_names[nb_name] = "R_06_" + segmentation_names[nb_name]
            if index_counter == (len(segmentation_names)*2):
                index_counter = 0
                
        df = pd.DataFrame(segmentation_names, columns = ["Filename"])

        df["Euclidean distance"] = self.euclidean_distances.flatten().tolist()
        df["Dice score"] = self.dice_score_list
        df["IoU"] = self.iou_list

        summary = df
        summary_statistics = df.describe()

        summary_string = df.to_string()
        summary_statistics_string = df.describe()

        #print(summary_string)
        print(summary_statistics_string)

        # example: get only the Dice Score mean value :) 
        #print(summary_statistics_string.loc[["mean"]]["Dice score"].iloc[0])

        return summary, summary_statistics



# #################################################################################################
# ******************************************* FUNCTIONS *******************************************
# #################################################################################################
def load_data(path_data, resize = (256,256), data_augmentation = ["rotate", "higher_brightness", "lower_brightness", "translate", "b_spline", "combine"], aux_information = False):
    """ Load the data for training the model. 

    Args:
        path_data (str): Path to the data to be loaded. 
        resize (tuple, optional): Size of the images to be converted to after loading. 
                                Defaults to (256, 256).
        data_augmentation(list of strings, optional): Define which data augmentation techniques are used. 
                                                    Defaults to all available augmentation techniques are activated:
                                                    ["rotate", "higher_brightness", "lower_brightness", "translate", "b_spline", "combine"]
        aux_information (bool): Decision if using the unsupervised or semi-supervised model for training. It is important to state
                                this already in the data loading process as the input data differs between these models. 

    Returns:
        [np array] data_moved_fixed: The loaded data for the unsupervised model. Shape: (number of files, resize)
            or
        [list] data_moved_fixed_segmentations: The loaded data for the semi-supervised model. Shape: (2), Contains two numpy arrays.
                                                The first numpy array contains the moved_fixed data. The second one contains the segmentations. 
                                                Shapes: (number of files, resize) and (resize, number of segmentations)
    
    """
    
    # auxiliary information contains segmentations
    if aux_information: 
        # path_data e.g. "data_scaling/dataset_feet_flat/dataset_bones/train/" and then folders "segmentations" and "images"
        filelist_segs = sorted(glob.glob(path_data + "segmentations/" + '*.png'))
        filelist_imgs = sorted(glob.glob(path_data + "images/" + '*.png'))

        print("-" * 30)
        print(f"{len(filelist_imgs) // 2} pairs of images are going to be loaded.")
        print("-" * 30)
        
        data_segmentations = normalize(np.array([np.array(Image.open(fname).resize(resize, Image.NEAREST).convert("L")) for fname in filelist_segs]))
        data_moved_fixed = normalize(np.array([np.array(Image.open(fname).resize(resize, Image.NEAREST).convert("L")) for fname in filelist_imgs]))

        # Swap axes for correct shapes: (256, 256, 19) - 19 segmentations with size 256x256
        data_segmentations = np.swapaxes(data_segmentations, 0, 2)
        # Flipping and rotating necessary for the correct representation of the segmentation
        data_segmentations = np.fliplr(data_segmentations) 
        data_segmentations = np.rot90(data_segmentations)

        augmenter = Augmentation(image_data = data_moved_fixed, 
                                segmentation_data = data_segmentations)

        # Do data augmentation
        if data_augmentation:
            data_moved_fixed, data_segmentations = augmenter.augment(augmentation_techniques = data_augmentation)

            print("-" * 30)
            print(f"{data_moved_fixed.shape[0] // 2} pairs of images are loaded (Augmentation was successful).")
            print("-" * 30)
    
    # without auxiliary information its just the two input images in one array 
    # structure like: moved, fixed, moved, fixed, ...
    else: 
        filelist_imgs = sorted(glob.glob(path_data + "images/" + '*.png'))

        # Optional: sort out augmentations for evaluation purposes (can be deleted now as the Augmentation Class exists)
        filelist_imgs = [k for k in filelist_imgs if not '_4_' in k]

        filelist_imgs = [k for k in filelist_imgs if not '_3_' in k]

        print("-" * 30)
        print(f"{len(filelist_imgs) // 2} pairs of images are going to be loaded.")
        print("-" * 30)
        
        data_moved_fixed = normalize(np.array([np.array(Image.open(fname).resize(resize, Image.NEAREST).convert("L")) for fname in filelist_imgs]))
        # print(data_moved_fixed.shape) # yields: (88, 256, 256)

        augmenter = Augmentation(image_data = data_moved_fixed)

        # Do data augmentation
        if data_augmentation:
            data_moved_fixed = augmenter.augment(augmentation_techniques = data_augmentation)

            print("-" * 30)
            print(f"{data_moved_fixed.shape[0] // 2} pairs of images are loaded (Augmentation was successful).")
            print("-" * 30)
    
    return augmenter

def vxm_data_generator(x_data, batch_size = 1, aux_information = False, train = True):
    """ Data generator for the training and testing process of VoxelMorph.

    Args:
        x_data ([type]): Training data from the load_data function. 
        batch_size (int, optional): Set the batch size for generating data.  
                                    Defaults to 1.
        aux_information (bool, optional): Decision if the unsupervised or semi-supervised model should be used.
                                        Defaults to False.
        train (bool, optional): Decision if the generator is used for training or for testing.
                                Defaults to True.

    Yields:
        [(inputs, outputs)]: Inputs (and outputs for validate data) for VoxelMorph.
                            
                            For the unsupervised model: 
                            inputs = [moving_images, fixed_images]
                            outputs = [fixed_images, zero_phi]

                            For the semi-supervised model: 
                            inputs = [moving_images, fixed_images, segmentations_moving]
                            outputs = [fixed_images, zero_phi, segmentations_fixed]
    """
    # preliminary sizing
    if not aux_information: vol_shape = x_data.shape[1:] # extract data shape
    if aux_information: vol_shape = x_data[0].shape[1:]
    ndims = len(vol_shape)
    
    # prepare a zero array with the size of the deformation
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    if aux_information: 
        segs_data = x_data[1]
        x_data = x_data[0]

    while True: 
        # Inputs are moving and fixed image
        if not aux_information:
            if not train: 
                # Data Loader for Validation / Test Data
                moving_images = x_data[0::2][:batch_size]
                fixed_images = x_data[1::2][:batch_size]
                moving_images = moving_images[..., np.newaxis]
                fixed_images = fixed_images[..., np.newaxis]
                # dont forget to skip the data in the foor loop of the main then!
                # Otherwise x_data will always be the same which results in always the same images.
            elif train:
                # Data Loader for Training Data
                # generate random image indices depending on batch size
                rng = np.random.default_rng()
                indices = rng.choice(x_data.shape[0]-2, size=batch_size, replace=False)
                # if indices are odd: make them even
                for i, index in enumerate(indices):
                    if index % 2 != 0:
                        indices[i] += 1

                # get the pairs of images
                moving_images = x_data[indices, ..., np.newaxis] 
                fixed_images = x_data[indices + 1, ..., np.newaxis]

            inputs = [moving_images, fixed_images]
            outputs = [fixed_images, zero_phi]
            
        # Inputs are moving and fixed image and additionally segmentation maps for both 
        elif aux_information: 

            if not train: 
                # Data Loader for Validation / Test Data
                moving_images = x_data[0::2][:batch_size]
                fixed_images = x_data[1::2][:batch_size]
                moving_images = moving_images[..., np.newaxis]
                fixed_images = fixed_images[..., np.newaxis]

                # get all segmentation indices
                segs_indices_moving = np.arange(0,19)
                segs_indices_fixed = np.arange(19,38)

                # dont forget to skip the data in the foor loop of the main then!
                # Otherwise x_data will always be the same which results in always the same images.

            elif train: 
                # Data Loader for Training Data
                # generate random image indices depending on batch size
                rng = np.random.default_rng()
                indices = rng.choice(x_data.shape[0]-2, size=batch_size, replace=False)
                # if indices are odd: make them even
                for i, index in enumerate(indices):
                    if index % 2 != 0:
                        indices[i] += 1
                
                # get the pairs of images
                moving_images = x_data[indices, ..., np.newaxis] 
                fixed_images = x_data[indices + 1, ..., np.newaxis] 

                segs_indices_moving = []
                segs_indices_fixed = []
                # create segmentation indices 
                for index in indices: 
                    # e.g. indices = [0, 2, 6]
                    # -> segmentations [(0*19) bis (1*19) -1, (2*19) bis (3*19) -1, (6*19) bis (7*19) -1]
                    from_to_indices_moving = range( (index * 19), ((index + 1) * 19) )
                    from_to_indices_fixed = range( ((index+1) * 19), (((index+1) + 1) * 19) )
                    
                    for range_index in from_to_indices_moving: 
                        segs_indices_moving.append(range_index)
                    for range_index in from_to_indices_fixed:
                        segs_indices_fixed.append(range_index)

            # get all necessary segmentations and reshape them 
            segmentations_moving = segs_data[np.newaxis, ..., segs_indices_moving].reshape(batch_size, *vol_shape, 19)
            segmentations_fixed = segs_data[np.newaxis, ..., segs_indices_fixed].reshape(batch_size, *vol_shape, 19)

            inputs = [moving_images, fixed_images, segmentations_moving]
            outputs = [fixed_images, zero_phi, segmentations_fixed]

        yield (inputs, outputs)

def plot_loss_all_kfold(hist, save_dir):
    """ Plotting the losses for the cross validation training.

    Args:
        hist ([keras history dictionary]): History of the training.
        save_dir ([str]): Directory for saving the plots
    """
    plt.style.use('science') # needs latex
    plt.figure()
    # Plot each loss and val loss of the training with crossvalidation
    for idx, loss_vals in enumerate(hist):
        plt.plot(loss_vals.epoch, loss_vals.history["loss"], '.-', label = f"Train Loss Run {idx+1}", color=COLOR_DICT[idx])
        plt.plot(loss_vals.epoch, loss_vals.history["val_loss"], '--', label = f"Val Loss Run {idx+1}", color=COLOR_DICT[idx])
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc = "upper right")
    plt.title("Loss History of all Kfold runs")
    plt.savefig(save_dir + ".svg")

    average_train = sum([np.array(ele.history["loss"]) for ele in hist]) / len(hist)
    average_val = sum([np.array(ele.history["val_loss"]) for ele in hist]) / len(hist)
    plt.figure()
    plt.plot(loss_vals.epoch, average_train, '.-', label = f"Train Loss")
    plt.plot(loss_vals.epoch, average_val, '.-', label = f"Val Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc = "upper right")
    plt.title("Mean Loss History of all Kfold runs")
    plt.savefig(save_dir + "_mean" + ".svg")

def get_save_location(modelpath, file_name, losses = ["MSE", "L2"], args_input_data = "bones", args_body_part = "hf", auxinf = False):
    """ Generates the save locations for figures and flows.

    Args:
        file_name ([str]): Filename
        losses (list, optional): List of the losses which were used.  
                                Defaults to ["MSE", "L2"].
        args_input_data (str, optional): Variant of the data used.
                                        Defaults to "bones".
        args_body_part (str, optional): Variant of the body part used.
                                        Defaults to "hf".
        auxinf (bool, optional): Decision if the unsupervised or semi-supervised model is used.
                                Defaults to False (= Unsupervised model).

    Returns:
        [str] save_name_figure: Save name for the figure.
        [str] save_name_flow: Save name for the flow.
    """
    loss_ext = ""
    for loss in losses: 
        loss_ext += "_" + loss 

    output_root = modelpath + "figures/"

    if not os.path.isdir(output_root):
        os.mkdir(output_root)

    if auxinf:
        save_name_figure = output_root + "aux_" + file_name.split("/")[-1][:-4] + loss_ext + ".svg"
        save_name_flow = output_root + "aux_" + "flow_" + file_name.split("/")[-1][:-4] + loss_ext + ".svg"
    else:
        save_name_figure = output_root + file_name.split("/")[-1][:-4] + loss_ext + ".svg"
        save_name_flow = output_root + "flow_" + file_name.split("/")[-1][:-4] + loss_ext + ".svg"
    
    return save_name_figure, save_name_flow 

def create_segmentation_array(list_filenames, size = (256, 256)):
    """ Generate an array with segmentations in it.

    Args:
        list_filenames (list of strings): List with the paths to the segmentations which should be loaded into an array.
        size (tuple, optional): Size of the images used. 
                                Defaults to (256, 256).

    Returns:
        [numpy array]: Array with shape (256, 256, 19) for the 19 segmentations with size 256x256
    """
    
    data_segmentations = normalize(np.array([np.array(Image.open(fname).resize(size, Image.NEAREST).convert("L")) for fname in list_filenames]))

    # Swap axes for correct shapes: (256, 256, 19) - 19 segmentations with size 256x256
    data_segmentations = np.swapaxes(data_segmentations, 0, 2)
    # Flipping and rotating necessary for the correct representation of the segmentation
    data_segmentations = np.fliplr(data_segmentations) 
    data_segmentations = np.rot90(data_segmentations)

    return data_segmentations

def save_figure(images, titles, title):
    """ Save a figure with x images and the titles for them and the title for the whole figure.

    Args:
        images ([list]): List of images with one size.
        titles ([list]): List of titles depending on the images.
        title ([str]): Title and saving path.
    """
    figure = slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)
    figure[0].savefig(title)

def save_segmentation_figure(modelpath, orig_segmentations, segmentations, args_path, args_input_data, title):
    """ Save a figure for comparing the segmentations (original fixed vs. predicted).

    Args:
        orig_segmentations ([np array]): Array of the original segmentations. Shape: (256, 256, number of segmentations).
        segmentations ([np array]): Array of the predicted segmentations. Shape: (256, 256, number of segmentations).
        args_path ([str]): Decision of the path determines which body part is used. 
        args_input_data ([str]): Decision which variant is used, e.g. bones.
        title ([str]): Title and saving location for the figure.
    """
    nb_segmentations = segmentations.shape[-1]
 
    # define the names every segmentation in the segmentations array
    segmentation_names = ["mc1", "mc2", "mc3", "mc4", "mc5",
                                    "pd1", "pd2", "pd3", "pd4", "pd5", 
                                    "pm2", "pm3", "pm4", "pm5", 
                                    "pp1", "pp2", "pp3", "pp4", "pp5"]

    images = []
    titles = []

    segmentation_to_show = "mc1"
    images.append(orig_segmentations[:, :, segmentation_names.index(segmentation_to_show)])
    titles.append(f"original {segmentation_to_show}")
    images.append(segmentations[:, :, segmentation_names.index(segmentation_to_show)])
    titles.append(f"predicted {segmentation_to_show}")

    segmentation_to_show = "pd1"
    images.append(orig_segmentations[:, :, segmentation_names.index(segmentation_to_show)])
    titles.append(f"original {segmentation_to_show}")
    images.append(segmentations[:, :, segmentation_names.index(segmentation_to_show)])
    titles.append(f"predicted {segmentation_to_show}")

    figure = slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)
    figure[0].savefig(title)

def save_predicted_segmentations(modelpath, segmentations, file_name, args_path, args_input_data, auxinf = False):
    """ Save every segmentation of array segmentations under the save_root. 

    Args:
        segmentations ([numpy array]): Numpy array with shape (1, 256, 256, 19) for 19 warped segmentations
                                    of size 256x256. 
        file_name ([str]): Filename
        args_path ([str]): Decision of the path determines which body part is used. 
        args_input_data ([str]): Decision which variant is used, e.g. bones
        auxinf (bool, optional): Decision if the unsupervised or semi-supervised model should be used. 
                                Defaults to False.

    Returns:
        [numpy array]: Predicted segmentations. Shape: (256, 256, number of segmentations). 
    """
    # define the names every segmentation in the segmentations array
    segmentation_names = ["mc1", "mc2", "mc3", "mc4", "mc5",
                                    "pd1", "pd2", "pd3", "pd4", "pd5", 
                                    "pm2", "pm3", "pm4", "pm5", 
                                    "pp1", "pp2", "pp3", "pp4", "pp5"]
    
    nb_segmentations = segmentations.shape[-1]

    if auxinf: 
        file_name = "aux_" + file_name

    predicted_segmentations = np.zeros((segmentations.shape[1], segmentations.shape[2], segmentations.shape[3]))

    if not modelpath.endswith("/"):
        modelpath = modelpath + "/"

    if not os.path.isdir(modelpath + "segmentations/"):
        os.mkdir(modelpath + "segmentations/")

    # Iterate through all segmentations and save them 
    for nb_segmentation in range(nb_segmentations):
        
        # Generate the correct file name for every segmentation
        seg_file_name = modelpath + "segmentations/" + file_name + segmentation_names[nb_segmentation] + ".png"

        # Get the current segmentation and squeeze it to image shape
        image = segmentations[:, :, :, nb_segmentation]
        image = np.squeeze(image, axis = 0)

        # Add the segmentation image to the array
        predicted_segmentations[:, :, nb_segmentation] = image * 255

        image = cv2.threshold(predicted_segmentations[:, :, nb_segmentation] * 255, 128, 255, cv2.THRESH_BINARY)[1]

        # Save the segmentation image
        image = Image.fromarray(image).convert("L")
        image = image.save(seg_file_name)

    return predicted_segmentations

def save_predicted_image(modelpath, image, file_name, args_path, args_input_data, auxinf = False):
    """ Save the predictions of the model. 

    Args:
        image ([numpy array]): The predicted image. 
        file_name ([str]): Filename
        args_path ([str]): Decision of the path determines which body part is used. 
        args_input_data ([str]): Decision which variant is used, e.g. bones
        auxinf (bool, optional): Decision if the unsupervised or semi-supervised model should be used. 
                                Defaults to False.
    """

    if not modelpath.endswith("/"):
        modelpath = modelpath + "/"

    if not os.path.isdir(modelpath + "images/"):
        os.mkdir(modelpath + "images/")

    if auxinf:
        pred_file_name = modelpath + "images/" + "aux_" + file_name + ".png"
    else: 
        pred_file_name = modelpath + "images/" + file_name + ".png"

    image = np.squeeze(image, axis = 2)
    image = Image.fromarray(image * 255).convert("L")
    image = image.save(pred_file_name)
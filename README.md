# Fortgeschrittenes Praktikum Visual Computing

<p float="left">
  <img src="https://mupam.net/wp-content/uploads/2019/10/Logos_frauhofer-300x150.png" width="200" title="Logo Fraunhofer IGD" />
  <img src="https://upload.wikimedia.org/wikipedia/de/thumb/2/24/TU_Darmstadt_Logo.svg/1200px-TU_Darmstadt_Logo.svg.png" width="200" title="Logo TU Darmstadt" /> 
</p>


# Abstract
This is our use and implementation of VoxelMorph [1] for certain 2D image data. VoxelMorph [1] is doing a deformable, unsupervised image registration in a deep learning manner in this case. The project was done for/with our supervisor Anna-Sophia Hertlein from Fraunhofer IGD. 

Note: Maybe you have to change to master branch: `git checkout master`

# Installation

We recommend the use of a conda environment and we used the Python version 3.9.

1. Using `conda` you can install the environment with:
`conda create --name myenv tensorflow-gpu`
2. Activate your conda environment:
`conda activate myenv`
3. Install the remaining required packages:
`pip install -r requirements.txt`

As the GitHub for VoxelMorph is kind of outdated for certain features, we had to do some adjustments for directory/package/... calls. 
Small tutorial for the setup process: 

4. As the pip-installation of VoxelMorph delivers an outdated version, use the GitHub (https://github.com/voxelmorph/voxelmorph) for the installation. This is done by cloning the GitHub to `cd ~/anaconda3/envs/myenv/lib/python3.9/site-packages/`. Then go inside the folder "voxelmorph" and call `python setup.py install` via terminal. 
5. Same process for `git clone https://github.com/adalca/pystrum` and `python setup.py install`, which should also be located under "~/anaconda3/envs/myenv/lib/python3.9/site-packages/pystrum". (In case pystrum is already installed uninstall it first `pip uninstall pystrum`)
6. Copy the voxelmorph-directory from the this GitLab-folder "voxelmorph/installation_data" into "~/anaconda3/envs/myenv/lib/python3.9/site-packages/" (Yes, overwrite it).
7. Copy the neurite- and the neuron-directory from this GitLab-folder also into "~/anaconda3/envs/myenv/lib/python3.9/site-packages/" (Yes, overwrite it).
8. Same process as in step 4 and 5 for `git clone https://github.com/tueimage/gryds` in "~/anaconda3/envs/myenv/lib/python3.9/site-packages/" and call `python setup.py install` in the folder gryds.


# Dataset organisation
The dataset folder for the file _"preprocessing.py"_ should look like the following. This is the structure of the original folder that we recieved for our project. You can simply download the _"dataset_final"_ from the cloud container. Rename the folder to _"dataset"_ and import it into the _"voxelmorph"_ folder (Yes, overwrite it). 
```bash
├──dataset
    ├──dataset_1 (e.g. Feet_flat)
        ├──R_01_1_l
            ├──mt1.png
            ├──mt2.png
            ├──....png 
            ├──skin.png 
            ├──feetBones.png 
            ├──Left_flat.png
        ├──R_01_1_r_rotated
            ├──....png 
        ├──R_02_1_l
            ├──....png 
        ├──R_02_1_r_rotated
            ├──....png 
        ├──...
```

The folder structure of the folder _"outputs"_ should look like the following:
```bash
├──outputs 
    ├──trained_models (Hint: this folder with all its subfolders are later created automatically by running main.py with training arguments)
        ├──figures 
            ├──...
        ├──images  
            ├──...
        ├──segmentations  
            ├──...
        ├──checkpoint
        ├──testmodel.ckpt.index
        ├──...
    ├──training_cv 
        ├──...
``` 

If these two folder structures are (manually) created, one can start the file _"preprocessing.py"_ by calling `python3 preprocessing.py` (for more detailed information like different aurguments see voxelmorph/preprocessing.py). The folder structure of the folder _"dataset"_ which is created by _"preprocessing.py"_ should look like the following:
```bash
├──dataset (e.g. data_scaling)
    ├──dataset_1 (e.g. dataset_hands_flat)
        ├──dataset_variant_1 (e.g. dataset_bones)
            ├──test
                ├──images
                    ├──files.png
                ├──segmentations
                    ├──files.png
            ├──train
                ├──images
                    ├──files.png
                ├──segmentations
                    ├──files.png
        ├──dataset_variant_2 (e.g. dataset_muscles)
            ├──test
            ├──train
    ├──dataset_2
    ├──...
``` 
For creating the dataset called _"dataset_all"_, start the file `python3 merge_datasets.py` **after** starting the file _"preprocessing.py"_ three times with each option of the parameter input_data (`python3 preprocessing.py -i hf`, `python3 preprocessing.py -i hb`, `python3 preprocessing.py -i ff`). <br />
This folder structure is needed for the training, testing and evaluation process for which the instructions are following right now.


# Training (normal)
The training process can be started by using the arguments from the parser in _"main.py"_.  <br />
An example for variant padding with the body part "hf" (i.e. hands flat), input data "bones", image size of 512, 15 epochs on GPU 6, the modelpath "outputs/trained_models/" to save the trained model with the modelname "testmodel" and the use of auxiliary information (i.e. semi-supervised learning) would be:  <br />
```
python3 main.py --variant="padding" --body_part="hf" --input_data="bones" --size=512 --gpus=6 --epoch=15 --modelpath="outputs/trained_models/" --modelname="testmodel" --auxinf=True
```


# Training (cross validation)
The training process can be started by using the arguments from the parser in _"main.py"_. <br />
An example for variant padding with the body part "hf" (i.e. hands flat), input data "bones", image size of 512, 15 epochs on GPU 6, the use of cross validation, the modelpath "outputs/training_cv/" to save each trained model of the crossvalidation split with the modelname "testmodel" and the use of auxiliary information (i.e. semi-supervised learning) would be:  <br />
```
python3 main.py --variant="padding" --body_part="hf" --input_data="bones" --size=512 --gpus=6 --epoch=15  --crossvalidation=True --modelpath="outputs/training_cv/" --modelname="testmodel" --auxinf=True
```


# Testing 
The testing process can be started by using the arguments from the parser in _"main.py"_. For testing a model, a trained model has to be created before and it has to be saved somewhere on the computer (this location is called modelpath). <br />
An example for variant padding with the body part "hf" (i.e. hands flat), input data "bones", image size of 512, 15 epochs on GPU 6, the modelpath "outputs/trained_models/" to load the trained model with the trained modelname "testmodel" (Note: this modelpath and modelname needs to be the same like the one used for training(normal)) and the use of auxiliary information (i.e. semi-supervised learning) would be:  <br />
```
python3 main.py --variant="padding" --body_part="hf" --input_data="bones" --size=512 --gpus=6 --epoch=15 --inference=True --modelpath="outputs/trained_models/" --modelname="testmodel" --auxinf=True
```
The path to the test directory is created automatically if one is following the folder structures above. 


# Evaluation
The evaluation process can be started by using the arguments from the parser in _"main.py"_. For evaluating the test process of a trained model, a trained model has to be used before to create predictions (i.e. to do inference). The predictions have be stored somehwere on the computer to do the evaluation. <br />
An example for variant padding with the body part "hf" (i.e. hands flat), input data "bones", image size of 512, 15 epochs on GPU 6, the modelpath "outputs/trained_models/" to load the trained model with the trained modelname "testmodel" (Note: this modelpath and modelname needs to be the same like the one used for training(normal)) and the use of auxiliary information (i.e. semi-supervised learning) would be:  <br />
```
python3 main.py --variant="padding" --body_part="hf" --input_data="bones" --size=512 --gpus=6 --epoch=15 --modelpath="outputs/trained_models/" --modelname="testmodel" --auxinf=True --evaluation=True
```
The path to the original segmentation directory and predicted segmentation directory is created automatically if one is following the folder structures above. 


# References
[1] Balakrishnan, G., Zhao, A., Sabuncu, M., Guttag, J., & Dalca, A. (2019). VoxelMorph: A Learning Framework for Deformable Medical Image Registration. IEEE Transactions on Medical Imaging, 38(8), 1788–1800.

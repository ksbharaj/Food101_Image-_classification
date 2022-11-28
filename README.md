# Deep learning group project for ingredient prediction


## Installation
1. Download the images data set from (https://www.kaggle.com/datasets/dansbecker/food-101?select=food-101.zip)
2. Download the code from this repository (https://github.gatech.edu/preddy61/dl_project.git)
3. Download the dataframes (data-frames.zip) from this Dropbox link. These are .h5 files that link the images to the ingredients (https://www.dropbox.com/s/20qgila6s7rz8d1/data-frames.zip?dl=0)
5. Unzip the datasets for image data and "data-frames.zip" directory under root folder of the cloned project. Image data should follow the path "root -> food-101 -> images"
6. Download the check point of the best model from (https://www.dropbox.com/s/whsjez6xxf7jijo/food-101_Best_augmented.pt?dl=0) and place it in "checkpoints" folder under root of the project.
7. Run the note books for training and testing. "grad_cam_testing.py" can be used for GradCam related activations.
8. NOTE: in notebooks, change mini_data_set = False to run training on entire dataset

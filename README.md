# Inference of Technical Data from CCTV images

The submitted workflow is written using Python 3 and performs object detection of people on videos. It takes videos as inputs and outputs predictions in the shape of a table (dataframe) on Dataiku.

As this project involves the use of confidential data from Schlumberger, the training dataset won't be inserted in this repository.

## Contents
### cctv-ml-workflow folder
#### useful .py files
* bbox_util.py: useful functions for data augmentation in object detection
* coco_eval.py and coco_utils.py: functions used in engine.py 
* utils.py: useful functions
* engine.py: train and evaluate functions for object detection

#### main files
* custom_dataset.py: CustomImageTensorDataset class, written to apply transforms to both "img" and "target" (boxes). 
* data_augmentation.py: Custom transforms for data augmentation in object detection.
* ml.py: Main code for training. Uses all the previously mentioned files to train a model.
* compute_toto.py: Main code to perform predictions using a trained model. Used on Dataiku (https://www.dataiku.com/ for more information about the data platform).

#### For data preparation
Videos_to_images.ipynb: Extract frames from videos every per_frame frames and saves them as .jpg images. 

#### Analysis of results
people-vs-time_plot.ipynb: From predictions, plots the number of people detected vs time (i.e. at which time they were detected).

### Report
blabla.pdf: This report gives a detailed description of the steps taken and the analysis done.

## Requirements
### System
Windows or Linux

### Software
LabelImg software to get from https://github.com/tzutalin/labelImg

### Packages
#### opencv
```pip install opencv-python```

#### pytorch (with or without CUDA)
Follow the instructions on https://pytorch.org/

#### pycocotools
##### Windows
Follow the instructions on https://github.com/philferriere/cocoapi

##### Linux
Get the repository from https://github.com/cocodataset/cocoapi

## Usage


## Built with
Python 3

## Author
Laura Su (lcs18@ic.ac.uk)

## Acknowledgments
* Dr Robert Zimmerman (Imperial College London), Adam Bowler and David Halliday (Schlumberger), my supervisors
* Schlumberger for accepting me for an internship
* Family, friends and fellow interns for their moral support

## License
This project is licensed under the MIT license - check file for more details.

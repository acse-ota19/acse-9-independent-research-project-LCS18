# Inference of Technical Data from CCTV images

The submitted workflow is written using Python 3 and performs object detection of people on videos. It takes videos as inputs and outputs predictions in the shape of a table (dataframe) on Dataiku. Those outputs are used to plot a number of people vs time, necessary for analysis (described in detail in the report).

As this project involves the use of confidential data (videos) from Schlumberger, the training dataset and the videos processed won't be inserted in this repository.
Also, as it doesn't seem possible to upload it here, a .pt file was created after training and it is not present on this repo.

## Contents
### cctv-ml-workflow folder
#### Useful .py files
* bbox_util.py: useful functions for data augmentation in object detection
* coco_eval.py and coco_utils.py: functions used in engine.py 
* utils.py: useful functions
* engine.py: train and evaluate functions for object detection

#### Main files
* custom_dataset.py: CustomImageTensorDataset class, written to apply transforms to both "img" and "target" (boxes). 
* data_augmentation.py: Custom transforms for data augmentation in object detection.
* ml.py: Main code for training. Needs a dataset as input (loaded using the CustomImageTensorDataset class). Uses all the previously mentioned files to train a model. Outputs a .pt file.
* compute_toto.py: Main code to perform predictions using a trained model on videos available on Dataiku. Used on Dataiku (https://www.dataiku.com/ for more information about the data platform). Inputs: videos. Outputs: dataframe (table as .csv or .xls)
* yt-loading-and-predicting.ipynb: Gets the results of prediction for public data downloaded from Youtube. Can plot the corresponding frame and its predicted boxes (detected people). 

#### For data preparation
Videos_to_images.ipynb: Extract frames from videos every per_frame frames and saves them as .jpg images. 

#### For test/verification
verification-test.ipynb: Example of training a model, loading a model and check if it is performing well on the labelled data.

#### Analysis of results
people-vs-time_plot.ipynb: From predictions, plots the number of people detected vs time (i.e. at which time they were detected).

### Report
final-report_Laura-Su.pdf: This report gives a detailed description of the steps taken and the analysis done.

## Requirements
### System
Windows or Linux

### Software (optional)
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

#### pytube
```pip install pytube```

Update note: it seems that something in Youtube changed recently (between July and August 2019), thus breaking the code. One of the package file needs to be modified. To fix it, check https://github.com/nficano/pytube/pull/435/files

## Usage
With this repository, as the dataset and the .pt file are lacking, it won't be possible to run the code. Please contact me (lcs18@ic.ac.uk) if you want to get the .pt file to run the predictions.

Using Pycharm and Jupyter Notebook is recommended.
This repository can be downloaded on your computer. All required packages must be installed.

The optional software and Videos_to_images.ipynb are for labelling, if desired.

Once all the files are retrieved, open Jupyter Notebook and run yt-loading-and-predicting.ipynb (choose a proper input Youtube video involving roughnecks working on a rig) to have a first visualisation of possible results.

Steps taken for the project:
As part of the project, the labelling software and the Videos_to_images.ipynb file were used for data labelling.
For model training, ml.py was used. 
Compute_toto.py was used on Dataiku to process the available videos (confidential) and output a big table.
people-vs-time_plot.ipynb was used to plot some early analysis of the results (people vs time), put in the report.

## Test/Verification
To check the results, the labelled data were compared to the predicted results (plotting the image and drawing its boxes is also done as a visual verification). This verification is not doable using this repository as the training dataset is needed.

## Built with
Python 3

## Author
Laura Su (lcs18@ic.ac.uk)

## Acknowledgments
* Dr Robert Zimmerman (Imperial College London), Adam Bowler and David Halliday (Schlumberger), my supervisors for their support
* Schlumberger for accepting me for an internship
* Family, friends and fellow interns for their moral support

## License
This project is licensed under the MIT license - check LICENSE file for more details.

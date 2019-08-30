# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Imports

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import dataiku
import pandas as pd
from dataiku import pandasutils as pdu

def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = False

    return True

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
model_folder = dataiku.Folder("euVgYlvN")
model_folder_info = model_folder.get_info()

folder_path = model_folder.get_path()
print(folder_path)

# for file in os.listdir(folder_path):
#     file_path = os.path.join(folder_path,file)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Transforms

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import random
import torch

from torchvision.transforms import functional as F

## Custom Compose, RandomHorizontalFlip, ToTensor
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

## get_transform function
def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Loading a model on Faster R-CNN Resnet50-FPN architecture

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
device = torch.device('cpu')

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model_save_name = folder_path + '/' + 'faster-r-cnn-resnet50-fpn-finetuning-dividedby2sizeimage.pt'

# Load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Replace the classifier with a new one, that has num_classes which is user-defined
num_classes = 2  # 1 class (person) + background

# Get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load(model_save_name))
model.to(device)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Non max suppression (box selection)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# non-max suppression, cf pyimageresearch
# Felzenszwalb et al.
def non_max_suppression_slow(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)


    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)

    return [int(p) for p in pick]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Loading a video and doing predictions using the model and generating the outputs

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Some other imports
import os
from PIL import Image
from moviepy.editor import *
import datetime
import time


# Read recipe inputs
videos = dataiku.Folder("KSA_RIG_VIDEOS.GkUSxFy1")
videos_info = videos.get_info()

# Expected dimensions of the video frames
height = int(720/2)
width = int(1280/2)
dim = (width, height)

# Put the model in evaluation mode
model.eval()

# Compute recipe outputs
for file_name in videos.list_paths_in_partition():
    with videos.get_download_stream(file_name) as stream:
        video_data = stream.read()
        with open("temp.mp4",'wb') as file_tmp: # Get one video at a time
            file_tmp.write(video_data)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if file_name[102:112] != "2019.03.17" and file_name[102:112] != "2019.03.18" and file_name[102:112] != "2019.03.19" and file_name[102:112] != "2019.03.20":
            # Get the frames of the current video
            ## Number to save a frame every "per_frame" frames
            per_frame = 100
            frames = []

            print("Getting the frames...")
            ## Opens the Video file
            cap = cv2.VideoCapture("temp.mp4")
            i = 0
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == False:
                    break
                if i%per_frame == 0:
                    # Resize the image from 1280x720 to widthxheight
                    res = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)

                    # Conversion to a RGB PIL image
                    frames.append(Image.fromarray(res).convert("RGB"))
                i += 1

            cap.release()
            cv2.destroyAllWindows()

            nb_tot_frames = i # storing the total number of frames

            print("Got all the frames!")

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if len(frames) > 1:
                # Transform the frames so that it will be in the proper "format"
                for i in range(len(frames)):
                    frames[i] = ToTensor()(frames[i], {})

                # DataLoader
                dataloader_frames = DataLoader(frames)

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                print("Doing the predictions...")
                # Computing the predictions
                prediction = [] # list to store the predictions

                for X in dataloader_frames:
                    img, target = X # note: "img is in list[img] configuration"
                    with torch.no_grad():
                        prediction.append(model([img[0].to(device)]))
                print("Predictions done!")

                print(prediction)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # Get the list of indexes of boxes selected
                list_picks = []
                score_threshold = 0.95
                for image_nb in range(len(prediction)):
                    boxes_pred = prediction[image_nb][0]["boxes"][prediction[image_nb][0]["scores"] > score_threshold]
                    list_picks.append(non_max_suppression_slow(boxes_pred.cpu(), 0.3))

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # Constructing the updated prediction results (after box selection)
                true_preds = []
                tmp = {}
                for image_nb in range(len(prediction)):
                    for key in prediction[image_nb][0].keys():
                        tmp[key] = prediction[image_nb][0][key][list_picks[image_nb]]
                    true_preds.append(tmp)
                    tmp = {}

                prediction = [] # free some memory

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # Get time in seconds for each frame
                clip = VideoFileClip("temp.mp4")
                duration_in_sec = clip.duration

                time_per_frame = (per_frame * duration_in_sec) / nb_tot_frames
                time_list = []
                for k in range(len(frames)):
                    time_list.append(k * time_per_frame)

                os.remove("temp.mp4") # we don't need the video anymore
                frames = [] # free some memory
                clip = [] # free some memory

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # OUTPUT DATAFRAME (cf video_labels_w_UNIX_timestamps, also on dataiku)
                data = {}

                ## TIME OFFSET
                time_offset = []

                ### Getting the number of people (according to what was detected)
                nb_boxes = [len(picks) for picks in list_picks]

                list_picks = [] # free some memory

                ### Computing the times corresponding to each box for storage in a dataframe
                for t in range(len(time_list)):
                    if nb_boxes[t] == 0: # even if there is no boxes, it's still a frame to count
                        time_offset.append(time_list[t])
                    else:
                        for k in range(nb_boxes[t]): # when there is one or more boxes
                            time_offset.append(time_list[t])

                time_list = [] # free some memory

                data["time_offset"] = time_offset


                ## ENTITY DESCRIPTION LIST (= classification of each box)
                entity_description_list = []
                for dic in true_preds:
                    if len(dic["labels"]) != 0:
                        for label in list(dic["labels"]):
                            # note: add if statements if other labels to be added
                            if int(label) == 1:
                                entity_description_list.append("person")
                    else:
                        entity_description_list.append("background")

                data["entity_description_list"] = entity_description_list


                ## ENTITY ID LIST
                entity_id_list = []
                for i in range(len(entity_description_list)):
                    if entity_description_list[i] == "person":
                        entity_id_list.append(1)
                    else:
                        entity_id_list.append(0)

                data["entity_id_list"] = entity_id_list

                entity_id_list = [] # free some memory

                ## BOUNDING BOXES COORDINATES (TODO: values in %, btw 0 and 1)
                left = []
                right = []
                top = []
                bottom = []
                for dic in true_preds:
                    if len(dic["boxes"]) != 0:
                        for box in list(dic["boxes"]):
                            left.append(round(float(box[0]) / width, 4))
                            right.append(round(float(box[2]) / width, 4))
                            top.append(round(float(box[1]) / height, 4))
                            bottom.append(round(float(box[3]) / height, 4))
                    else:
                        left.append("NA")
                        right.append("NA")
                        top.append("NA")
                        bottom.append("NA")

                data["bounding_box_left"] = left
                data["bounding_box_right"] = right
                data["bounding_box_top"] = top
                data["bounding_box_bottom"] = bottom

                true_preds = [] # free some memory
                left = [] # free some memory
                right = [] # free some memory
                top = [] # free some memory
                bottom = [] # free some memory


                ## VIDEO FILE PATH
                ### TODO on dataiku
                data["video_file_path"] = ["" for i in range(len(entity_description_list))]


                ## VIDEO SUFFIX
                data["video_suffix"] = [file_name for i in range(len(entity_description_list))]


                ## FILE LIST
                data["file_list"] = [file_name for i in range(len(entity_description_list))]


                ## RIG
                data["RIG"] = [file_name[17:22] for i in range(len(entity_description_list))]


                ## DATE
#                 data["date"] = [file_name[102:112] for i in range(len(entity_description_list))]
                data["date"] = [file_name[-25:-15] for i in range(len(entity_description_list))]


                ## TIME
#                 data["time"] = [file_name[115:123] for i in range(len(entity_description_list))]
                data["time"] = [file_name[-12:-4] for i in range(len(entity_description_list))]


                ## DATE TIME VIDEO START
#                 data["date_time_video_start"] = [file_name[102:112] + "-" + file_name[115:123] for i in range(len(entity_description_list))]
                data["date_time_video_start"] = [file_name[-25:-15] + "-" + file_name[-12:-4] for i in range(len(entity_description_list))]


                ## DATE TIME VIDEO START STANDARD
#                 date = file_name[102:112]
                date = file_name[-25:-15]
                date = date.replace(".", "-")
#                 time_h = file_name[115:123]
                time_h = file_name[-12:-4]
                time_h = time_h.replace(".", ":")

                data["date_time_video_start_standard"] = [date + "T" + time_h + ".000Z" for i in range(len(entity_description_list))]


                ## FRAME DATETIME
                x = time.strptime(time_h,'%H:%M:%S')
                tempo = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds() # from "00:00:00" to seconds
                temp_sec = [tempo + elt for elt in time_offset]

                time_offset = [] # free some memory

                if time_h[0] == "0": # if hh is between 0 and 9
                    cur_time = ["0" + str(datetime.timedelta(seconds=t)) for t in temp_sec]
                else: # hh is between 10 and 23
                    cur_time = [str(datetime.timedelta(seconds=t)) for t in temp_sec]

                ### Converting the end results to "00:00:00.000" format
                for i in range(len(cur_time)):
                    if len(cur_time[i]) == 15: # the time is in "00:00:00.000000" format
                        cur_time[i] = cur_time[i][:-3]
                    elif len(cur_time[i]) == 10: # the time is in "00:00:00.00" format
                        cur_time[i] += "0"
                    elif len(cur_time[i]) == 9: # the time is in "00:00:00.0" format
                        cur_time[i] += "00"
                    elif len(cur_time[i]) == 8: # the time is in "00:00:00" format
                        cur_time[i] += ".000"

                data["frame_datetime"] = [date + "T" + cur_time[i] + "Z" for i in range(len(entity_description_list))]

                entity_description_list = [] # free some memory
                cur_time = [] # free some memory

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                toto_df = pd.DataFrame(data)

                data = {} # free some memory

                # Write recipe outputs
                toto = dataiku.Dataset("toto")
                toto.write_with_schema(toto_df)

            else:
                frames = []
                os.remove("temp.mp4")

        print(file_name)
        print("")
import os
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import torch


# Custom dataset
class CustomImageTensorDataset(Dataset):
    def __init__(self, root, transforms=None):
        """
        Args:
        #    data (Tensor): A tensor containing the data e.g. images
        #    targets (Tensor): A tensor containing all the labels
        #    transform (callable, optional): Optional transform to be applied
        #        on a sample.
        """
        self.root = root
        self.transforms = transforms

        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPGImages"))))
        self.boxes = list(sorted(os.listdir(os.path.join(root, "Boxes"))))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # load images and boxes
        img_path = os.path.join(self.root, "JPGImages", self.imgs[idx])
        box_path = os.path.join(self.root, "Boxes", self.boxes[idx])
        img = Image.open(img_path).convert("RGB")

        tree = ET.parse(box_path)
        treeroot = tree.getroot()
        boxes = []
        one_box = []
        for i in range(6, len(treeroot)):
            boxes.append([int(treeroot[i][4][j].text) for j in range(4)])

            # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

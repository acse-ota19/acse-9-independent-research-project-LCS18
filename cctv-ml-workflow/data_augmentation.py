"""Laura Su (GitHub: LCS18)"""

import torch
import random

from torchvision.transforms import functional as F

from bbox_util import *


# Custom transforms (adding transforms to the boxes)
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
            # if "keypoints" in target:
            #     keypoints = target["keypoints"]
            #     keypoints = _flip_coco_person_keypoints(keypoints, width)
            #     target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomScale(object):
    """Randomly scales an image

    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    scale: float or tuple(float)
        if **float**, the image is scaled by a factor drawn
        randomly from a range (1 - `scale` , 1 + `scale`). If **tuple**,
        the `scale` is drawn randomly from values specified by the
        tuple

    Returns
    -------

    numpy.ndarray
        Scaled image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Transformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, scale=0.2, diff=False):
        self.scale = scale

        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"
            assert self.scale[0] > -1, "Scale factor can't be less than -1"
            assert self.scale[1] > -1, "Scale factor can't be less than -1"
        else:
            assert self.scale > 0, "Please input a positive float"
            self.scale = (max(-1, -self.scale), self.scale)

        self.diff = diff

    def __call__(self, image, target):
        # Chose a random digit to scale by
        image = image.numpy()  # from tensor to numpy array
        image = image.transpose(1, 2, 0)  # from (c, h, w) to (h, w, c)
        image_shape = image.shape

        if self.diff:
            scale_x = random.uniform(*self.scale)
            scale_y = random.uniform(*self.scale)
        else:
            scale_x = random.uniform(*self.scale)
            scale_y = scale_x

        resize_scale_x = 1 + scale_x
        resize_scale_y = 1 + scale_y

        image = cv2.resize(image, None, fx=resize_scale_x, fy=resize_scale_y, interpolation=cv2.INTER_LINEAR)

        bboxes = target["boxes"].numpy()  # from tensor to numpy array
        bboxes[:, :4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]

        canvas = np.zeros(image_shape, dtype=np.uint8)

        y_lim = int(min(resize_scale_y, 1) * image_shape[0])
        x_lim = int(min(resize_scale_x, 1) * image_shape[1])

        canvas[:y_lim, :x_lim, :] = image[:y_lim, :x_lim, :]

        image = canvas
        image = torch.from_numpy(image).float()  # from numpy array to tensor

        target["boxes"] = clip_box(bboxes, [0, 0, 1 + image_shape[1], image_shape[0]], 0.25)
        target["boxes"] = torch.from_numpy(target["boxes"]).float()  # from numpy array to tensor

        image = image.permute(2, 0, 1)  # from (h, w, c) to (c, h, w)

        return image, target


class RandomRotate(object):
    """Randomly rotates an image

    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    angle: float or tuple(float)
        if **float**, the image is rotated by a factor drawn
        randomly from a range (-`angle`, `angle`). If **tuple**,
        the `angle` is drawn randomly from values specified by the
        tuple

    Returns
    -------

    numpy.ndarray
        Rotated image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Transformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, angle=10):
        self.angle = angle

        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"
        else:
            self.angle = (-self.angle, self.angle)

    def __call__(self, image, target):
        angle = random.uniform(*self.angle)

        image = image.numpy()  # from tensor to numpy array
        image = image.transpose(1, 2, 0)  # from (c, h, w) to (h, w, c)
        w, h = image.shape[1], image.shape[0]
        cx, cy = w // 2, h // 2

        image = rotate_im(image, angle)
        bboxes = target["boxes"].numpy()  # from tensor to numpy array

        corners = get_corners(bboxes)
        corners = np.hstack((corners, bboxes[:, 4:]))

        if len(corners) != 0:
            corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)

        new_bbox = get_enclosing_box(corners)

        scale_factor_x = image.shape[1] / w
        scale_factor_y = image.shape[0] / h

        image = cv2.resize(image, (w, h))

        new_bbox[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]
        bboxes = new_bbox
        target["boxes"] = clip_box(bboxes, [0, 0, w, h], 0.25)
        target["boxes"] = torch.from_numpy(target["boxes"]).float()  # from numpy array to tensor

        image = torch.from_numpy(image).float()  # from numpy array to tensor
        image = image.permute(2, 0, 1)  # from (h, w, c) to (c, h, w)

        return image, target


class RandomTranslate(object):
    """Randomly Translates the image

    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    translate: float or tuple(float)
        if **float**, the image is translated by a factor drawn
        randomly from a range (1 - `translate` , 1 + `translate`). If **tuple**,
        `translate` is drawn randomly from values specified by the
        tuple

    Returns
    -------

    numpy.ndarray
        Translated image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Transformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, translate=0.2, diff=False):
        self.translate = translate

        if type(self.translate) == tuple:
            assert len(self.translate) == 2, "Invalid range"
            assert self.translate[0] > 0 & self.translate[0] < 1
            assert self.translate[1] > 0 & self.translate[1] < 1
        else:
            assert self.translate > 0 and self.translate < 1
            self.translate = (-self.translate, self.translate)

        self.diff = diff

    def __call__(self, image, target):
        # Chose a random digit to scale by
        image = image.numpy()  # from tensor to numpy array
        image = image.transpose(1, 2, 0)  # from (c, h, w) to (h, w, c)
        image_shape = image.shape

        # translate the image
        # percentage of the dimension of the image to translate
        translate_factor_x = random.uniform(*self.translate)
        translate_factor_y = random.uniform(*self.translate)

        if not self.diff:
            translate_factor_y = translate_factor_x

        canvas = np.zeros(image_shape).astype(np.uint8)

        corner_x = int(translate_factor_x * image.shape[1])
        corner_y = int(translate_factor_y * image.shape[0])

        # change the origin to the top-left corner of the translated box
        orig_box_cords = [max(0, corner_y), max(corner_x, 0), min(image_shape[0], corner_y + image.shape[0]),
                          min(image_shape[1], corner_x + image.shape[1])]

        mask = image[max(-corner_y, 0):min(image.shape[0], -corner_y + image_shape[0]),
               max(-corner_x, 0):min(image.shape[1], -corner_x + image_shape[1]), :]
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3], :] = mask
        image = canvas

        bboxes = target["boxes"].numpy()  # from tensor to numpy array
        bboxes[:, :4] += [corner_x, corner_y, corner_x, corner_y]
        target["boxes"] = clip_box(bboxes, [0, 0, image_shape[1], image_shape[0]], 0.25)
        target["boxes"] = torch.from_numpy(target["boxes"]).float()  # from numpy array to tensor

        image = torch.from_numpy(image).float()  # from numpy array to tensor
        image = image.permute(2, 0, 1)  # from (h, w, c) to (c, h, w)

        return image, target


def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
        # transforms.append(RandomTranslate(0.3))
        # transforms.append(RandomScale(0.5))  # scale
        # transforms.append(RandomRotate(10))

    return Compose(transforms)

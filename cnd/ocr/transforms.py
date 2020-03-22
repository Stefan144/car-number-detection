import torch
from torchvision.transforms import Compose
import numpy as np
import cv2
from albumentations.augmentations.transforms import RandomCrop, Rotate, \
                                                    RandomBrightnessContrast, GaussNoise

# Scale - last thing to do


class ImageNormalizationTransform(object):
    def __call__(self, image):
        return image / 255.


class GrayTransform(object):
    def __call__(self, image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image


class ToTensorTransform(object):
    def __call__(self, image):
        image = image.astype(np.float32)[None, :, :]
        return torch.from_numpy(image)


class ScaleTransform(object):
    def __init__(self, resize_shape):
        self.new_shape = tuple(resize_shape[::-1])

    def __call__(self, image, interpolation=cv2.INTER_LINEAR):
        image = cv2.resize(image, dsize=self.new_shape, interpolation=interpolation)
        return image


class RandomCropTransform(object):
    def __call__(self, image):
        h, w = np.random.randint(1, image.shape[0]), np.random.randint(1, image.shape[1]) 
        image = RandomCrop(height=h, width=w, p=0.5)(image=image)['image']
        return image


class RandomFlipTransform(object):
    def __call__(self, image, angle=180):
        image = Rotate(limit=angle, p=0.5, border_mode=1)(image=image)['image']
        return image


class RandomBrightnessContrastTransform(object):
    def __call__(self, image, brightness_limit=0.5, contrast_limit=0.5):
        image = RandomBrightnessContrast(brightness_limit=brightness_limit,
                                         contrast_limit=contrast_limit,
                                         p=0.5)(image=image)['image']
        return image


class GaussNoiseTransform(object):
    def __call__(self, image, var_limit=(100, 500)):
        image = GaussNoise(var_limit=var_limit, p=0.5)(image=image)['image']
        return image


def get_transforms(image_size):
    transform = Compose([GrayTransform(),
                         ImageNormalizationTransform(),
                         ScaleTransform(image_size),
                         ToTensorTransform()])
    return transform

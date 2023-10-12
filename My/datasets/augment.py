import albumentations as A
import numpy as np


# define heavy augmentations
def get_training_augmentation(height, width):
    def wrapper(image, mask):
        train_transform = [

            A.HorizontalFlip(p=0.5),

            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

            A.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0),
            A.RandomCrop(height=height, width=width, always_apply=True),

            A.IAAAdditiveGaussianNoise(p=0.2),
            A.IAAPerspective(p=0.5),

            A.OneOf(
                [
                    A.CLAHE(p=1),
                    A.RandomBrightness(p=1),
                    A.RandomGamma(p=1),
                ],
                p=0.9,
            ),

            A.OneOf(
                [
                    A.IAASharpen(p=1),
                    A.Blur(blur_limit=3, p=1),
                    A.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),

            A.OneOf(
                [
                    A.RandomContrast(p=1),
                    A.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
        ]
        return A.Compose(train_transform)

    return wrapper


def get_validation_augmentation(height, width):
    """Add paddings to make image shape divisible by 32"""

    def wrapper(image, mask):
        test_transform = [
            A.PadIfNeeded(height, width)
        ]
        return A.Compose(test_transform)

    return wrapper

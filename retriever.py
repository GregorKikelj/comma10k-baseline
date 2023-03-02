import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import cv2
import albumentations as A
from albumentations.core.composition import Compose
from typing import Callable, List
from pathlib import Path
from torch.utils.data import Dataset


def pad_to_multiple(x, k=32):
    return int(k * (np.ceil(x / k)))


def get_scale_transform(height: int, width: int):
    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.PadIfNeeded(
                pad_to_multiple(height),
                pad_to_multiple(width),
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
            ),
        ]
    )


def get_train_transforms(height: int, width: int, level: str):
    if level == "none":
        return get_scale_transform(height, width)

    elif level == "l1":
        return A.Compose(
            [
                get_scale_transform(height, width),
                A.HorizontalFlip(p=0.5),
                A.ISONoise(p=1),
                A.Sharpen(p=1),
                A.CLAHE(clip_limit=2.0, p=1),
                A.FancyPCA(alpha=0.5, p=1),
            ]
        )
    elif level == "l2":
        return A.Compose(
            [
                get_scale_transform(height, width),
                A.HorizontalFlip(p=0.5),
                A.GaussNoise(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.2, p=1
                ),
                A.CLAHE(clip_limit=2.0, p=1),
                A.Sharpen(p=1),
                A.CoarseDropout(
                    min_height=1,
                    max_height=192,
                    min_width=1,
                    max_width=192,
                    mask_fill_value=0,
                    p=1,
                ),
                # moderate grid distortion
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.15,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                    p=1,
                ),
            ]
        )
    if level == "light":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.GaussNoise(p=0.2),
                A.OneOf(
                    [
                        A.CLAHE(p=1.0),
                        A.RandomBrightness(p=1.0),
                        A.RandomGamma(p=1.0),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.Sharpen(p=1.0),
                        A.Blur(blur_limit=3, p=1.0),
                        A.MotionBlur(blur_limit=3, p=1.0),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.RandomContrast(p=1.0),
                        A.HueSaturationValue(p=1.0),
                    ],
                    p=0.5,
                ),
                A.Resize(height=height, width=width, p=1.0),
                A.PadIfNeeded(
                    pad_to_multiple(height),
                    pad_to_multiple(width),
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                ),
            ],
            p=1.0,
        )

    elif level == "hard":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.GaussNoise(p=0.2),
                A.OneOf(
                    [
                        A.GridDistortion(
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=0,
                            p=1.0,
                        ),
                        A.ElasticTransform(
                            alpha_affine=10,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=0,
                            p=1.0,
                        ),
                        A.ShiftScaleRotate(
                            shift_limit=0,
                            scale_limit=0,
                            rotate_limit=10,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=0,
                            p=1.0,
                        ),
                        A.OpticalDistortion(
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=0,
                            p=1.0,
                        ),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.CLAHE(p=1.0),
                        A.RandomBrightness(p=1.0),
                        A.RandomGamma(p=1.0),
                        A.ISONoise(p=1.0),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.Sharpen(p=1.0),
                        A.Blur(blur_limit=3, p=1.0),
                        A.MotionBlur(blur_limit=3, p=1.0),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.RandomContrast(p=1.0),
                        A.HueSaturationValue(p=1.0),
                    ],
                    p=0.5,
                ),
                A.Resize(height=height, width=width, p=1.0),
                A.Cutout(p=0.3),
                A.PadIfNeeded(
                    pad_to_multiple(height),
                    pad_to_multiple(width),
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                ),
            ],
            p=1.0,
        )
    elif level == "hard_weather":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.GaussNoise(p=0.2),
                A.OneOf(
                    [
                        A.GridDistortion(
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=0,
                            p=1.0,
                        ),
                        A.ElasticTransform(
                            alpha_affine=10,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=0,
                            p=1.0,
                        ),
                        A.ShiftScaleRotate(
                            shift_limit=0,
                            scale_limit=0,
                            rotate_limit=10,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=0,
                            p=1.0,
                        ),
                        A.OpticalDistortion(
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=0,
                            p=1.0,
                        ),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.CLAHE(p=1.0),
                        A.RandomBrightness(p=1.0),
                        A.RandomGamma(p=1.0),
                        A.ISONoise(p=1.0),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.Sharpen(p=1.0),
                        A.Blur(blur_limit=3, p=1.0),
                        A.MotionBlur(blur_limit=3, p=1.0),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.RandomContrast(p=1.0),
                        A.HueSaturationValue(p=1.0),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.RandomFog(fog_coef_upper=0.8, p=1.0),
                        A.RandomRain(p=1.0),
                        A.RandomSnow(p=1.0),
                        A.RandomSunFlare(src_radius=100, p=1.0),
                    ],
                    p=0.4,
                ),
                A.Resize(height=height, width=width, p=1.0),
                A.Cutout(p=0.3),
                A.PadIfNeeded(
                    pad_to_multiple(height),
                    pad_to_multiple(width),
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                ),
            ],
            p=1.0,
        )
    else:
        # throw error
        print("Invalid augmentation level")
        raise ValueError


def get_valid_transforms(height: int, width: int):
    return get_scale_transform(height, width)


def to_tensor(x, **_):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn: Callable):
    return A.Compose(
        [
            A.Lambda(image=preprocessing_fn),
            A.Lambda(image=to_tensor, mask=to_tensor),
        ]
    )


class TrainRetriever(Dataset):
    def __init__(
        self,
        data_path: Path,
        image_names: List[str],
        preprocess_fn: Callable,
        transforms: Compose,
        class_values: List[int],
    ):
        super().__init__()

        self.data_path = data_path
        self.image_names = image_names
        self.transforms = transforms
        self.preprocess = get_preprocessing(preprocess_fn)
        self.class_values = class_values

    def __getitem__(self, index: int):
        image_name = self.image_names[index]

        image = cv2.imread(str(self.data_path / "imgs" / image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(
            str(self.data_path / "masks" / image_name), cv2.IMREAD_GRAYSCALE
        ).astype(
            "uint8"
        )  # masks aren't grayscale but this seems to work

        if self.transforms:
            sample = self.transforms(image=image, mask=mask)
            image = sample["image"]
            mask = sample["mask"]

        mask = np.stack([(mask == v) for v in self.class_values], axis=-1).astype(
            "uint8"
        )

        if self.preprocess:
            sample = self.preprocess(image=image, mask=mask)
            image = sample["image"]
            mask = sample["mask"]

        return image, mask

    def __len__(self) -> int:
        return len(self.image_names)

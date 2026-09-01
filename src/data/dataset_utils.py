from pathlib import Path

import numpy as np
from PIL import Image

from src.config import (
    DATASET_ROOT,
    CITYSCAPES_LABEL_IDS,
)


def get_sample_paths(
    split: str,
    sample_id: str,
) -> dict:
    """
    Return all paths associated with one sample.
    """

    split_root = DATASET_ROOT / split

    return {
        "image": (
            split_root
            / "images"
            / f"{sample_id}_leftImg8bit.png"
        ),

        "label": (
            split_root
            / "label_ids"
            / f"{sample_id}_gtFine_labelIds.png"
        ),

        "instance": (
            split_root
            / "instance_ids"
            / f"{sample_id}_gtFine_instanceIds.png"
        ),

        "target": (
            split_root
            / "target_masks"
            / f"{sample_id}_targetMask.png"
        ),
    }


def load_rgb_image(path: Path) -> np.ndarray:
    """
    Load RGB image as NumPy array.
    """

    with Image.open(path) as img:
        image = np.asarray(
            img.convert("RGB")
        )

    return image


def load_mask(path: Path) -> np.ndarray:
    """
    Load label / instance / target mask
    without changing its original integer values.
    """

    with Image.open(path) as img:
        mask = np.asarray(img)

    return mask


def load_sample(
    split: str,
    sample_id: str,
) -> dict:
    """
    Load all data belonging to one sample.
    """

    paths = get_sample_paths(
        split=split,
        sample_id=sample_id,
    )

    return {
        "paths": paths,

        "image": load_rgb_image(
            paths["image"]
        ),

        "label": load_mask(
            paths["label"]
        ),

        "instance": load_mask(
            paths["instance"]
        ),

        "target": load_mask(
            paths["target"]
        ),
    }


def count_target_pixels(
    target: np.ndarray,
    class_id: int,
) -> int:
    """
    Count pixels belonging to target class.
    """

    return int(
        np.count_nonzero(
            target == class_id
        )
    )


def count_instances(
    instance_mask: np.ndarray,
    target_class_id: int,
) -> int:
    """
    Count Cityscapes object instances.

    Cityscapes instance IDs follow approximately:

        class_id * 1000 + instance_number

    Example:
        24000 -> person instance
        25000 -> rider instance
        26000 -> car instance
        32000 -> motorcycle instance

    Group / non-instance labels below 1000
    are not counted as individual instances.
    """

    label_ids = CITYSCAPES_LABEL_IDS[
        target_class_id
    ]

    total = 0

    for label_id in label_ids:

        class_pixels = (
            (instance_mask >= 1000)
            &
            (
                instance_mask // 1000
                == label_id
            )
        )

        instance_ids = np.unique(
            instance_mask[class_pixels]
        )

        total += len(instance_ids)

    return int(total)


def build_expected_target_from_labels(
    label_mask: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct the expected target mask
    directly from the original Cityscapes label IDs.

    target:
        0 = background
        1 = person / rider
        2 = car
        3 = motorcycle
    """

    expected = np.zeros(
        label_mask.shape,
        dtype=np.uint8,
    )

    for target_class_id, label_ids in (
        CITYSCAPES_LABEL_IDS.items()
    ):

        class_mask = np.isin(
            label_mask,
            label_ids,
        )

        expected[class_mask] = (
            target_class_id
        )

    return expected

import numpy as np

from sporco import signal


def normalize_rgb(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert uint8 RGB image [0, 255]
    to float32 RGB image [0, 1].
    """

    if image.ndim != 3:
        raise ValueError(
            f"Expected RGB image with 3 dimensions, "
            f"found shape {image.shape}"
        )

    if image.shape[2] != 3:
        raise ValueError(
            f"Expected 3 RGB channels, "
            f"found shape {image.shape}"
        )

    image = image.astype(
        np.float32,
        copy=False,
    )

    image /= 255.0

    return image


def rgb_to_grayscale(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert RGB image to grayscale.

    Input
    -----
    H x W x 3 float32 image in [0, 1]

    Output
    ------
    H x W float32 grayscale image.
    """

    grayscale = signal.rgb2gray(
        image
    )

    grayscale = np.asarray(
        grayscale,
        dtype=np.float32,
    )

    return grayscale


def tikhonov_highpass(
    grayscale: np.ndarray,
    lmbda: float,
    padding: int,
):
    """
    Split a grayscale image into low-pass
    and high-pass components using
    SPORCO Tikhonov filtering.

    highpass = grayscale - lowpass
    """

    if grayscale.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale image, "
            f"found shape {grayscale.shape}"
        )

    lowpass, highpass = (
        signal.tikhonov_filter(
            grayscale,
            lmbda,
            padding,
        )
    )

    lowpass = np.asarray(
        lowpass,
        dtype=np.float32,
    )

    highpass = np.asarray(
        highpass,
        dtype=np.float32,
    )

    return lowpass, highpass


def preprocess_image(
    image: np.ndarray,
    lmbda: float,
    padding: int,
) -> dict:
    """
    Complete STEP 02 preprocessing pipeline.

    RGB uint8
        ->
    RGB float32 [0,1]
        ->
    Grayscale
        ->
    Tikhonov decomposition
        ->
    Low-pass + High-pass
    """

    rgb_float = normalize_rgb(
        image
    )

    grayscale = rgb_to_grayscale(
        rgb_float
    )

    lowpass, highpass = (
        tikhonov_highpass(
            grayscale,
            lmbda=lmbda,
            padding=padding,
        )
    )

    return {
        "rgb": rgb_float,
        "grayscale": grayscale,
        "lowpass": lowpass,
        "highpass": highpass,
    }


def validate_preprocessed(
    original_image: np.ndarray,
    grayscale: np.ndarray,
    lowpass: np.ndarray,
    highpass: np.ndarray,
    target_mask: np.ndarray,
) -> list:
    """
    Check that preprocessing did not alter
    spatial dimensions and that all numeric
    arrays contain finite values.

    Returns a list of validation errors.
    """

    errors = []

    original_shape = (
        original_image.shape[:2]
    )

    arrays = {
        "grayscale": grayscale,
        "lowpass": lowpass,
        "highpass": highpass,
        "target": target_mask,
    }

    for name, array in arrays.items():

        if array.shape != original_shape:

            errors.append(
                f"{name} shape "
                f"{array.shape} != "
                f"original spatial shape "
                f"{original_shape}"
            )

    for name, array in [
        ("grayscale", grayscale),
        ("lowpass", lowpass),
        ("highpass", highpass),
    ]:

        if not np.all(
            np.isfinite(array)
        ):

            errors.append(
                f"{name} contains "
                f"NaN or infinite values"
            )

    if grayscale.size > 0:

        gray_min = float(
            grayscale.min()
        )

        gray_max = float(
            grayscale.max()
        )

        if (
            gray_min < -1e-6
            or gray_max > 1.0 + 1e-6
        ):

            errors.append(
                "grayscale values outside "
                f"expected [0,1] range: "
                f"min={gray_min:.6f}, "
                f"max={gray_max:.6f}"
            )

    return errors

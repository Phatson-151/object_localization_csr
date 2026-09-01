import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy import fft as spfft
import numpy as np
import pandas as pd

from skimage.transform import resize

from src.config import (
    STEP02_OUTPUT,
)

from src.utils.output_utils import (
    sample_filename,
)


# ============================================================
# STEP 02 HIGH-PASS LOADING
# ============================================================

def get_highpass_path(
    sample_id: str,
):
    """
    Return STEP 02 high-pass array path.
    """

    return (
        STEP02_OUTPUT
        / "arrays"
        / sample_filename(
            step=2,
            sample_id=sample_id,
            name="highpass",
            extension="npy",
        )
    )


def load_highpass(
    sample_id: str,
) -> np.ndarray:
    """
    Load signed float32 high-pass image
    generated in STEP 02.
    """

    path = get_highpass_path(
        sample_id
    )

    if not path.exists():

        raise FileNotFoundError(
            f"STEP 02 high-pass file "
            f"not found: {path}"
        )

    signal = np.load(
        path,
        allow_pickle=False,
    )

    signal = np.asarray(
        signal,
        dtype=np.float32,
    )

    if signal.ndim != 2:

        raise ValueError(
            f"{sample_id}: expected "
            f"2D high-pass array, "
            f"found {signal.shape}"
        )

    if not np.all(
        np.isfinite(signal)
    ):

        raise ValueError(
            f"{sample_id}: high-pass "
            f"contains NaN or Inf."
        )

    return signal


# ============================================================
# MODEL RESOLUTION
# ============================================================

def resize_highpass(
    signal: np.ndarray,
    scale: float,
) -> np.ndarray:
    """
    Resize signed high-pass signal.

    STEP 02 full-resolution files remain
    unchanged. This resize is only used for
    computationally manageable CSC training.
    """

    if scale <= 0.0 or scale > 1.0:

        raise ValueError(
            "scale must satisfy "
            "0 < scale <= 1"
        )

    if scale == 1.0:

        return signal.astype(
            np.float32,
            copy=False,
        )

    height, width = signal.shape

    new_height = max(
        1,
        int(
            round(
                height * scale
            )
        ),
    )

    new_width = max(
        1,
        int(
            round(
                width * scale
            )
        ),
    )

    resized = resize(
        signal,
        (
            new_height,
            new_width,
        ),
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    )

    return np.asarray(
        resized,
        dtype=np.float32,
    )


# ============================================================
# TRAINING SIGNAL STACK
# ============================================================

def build_training_stack(
    train_manifest: pd.DataFrame,
    num_samples: int,
    scale: float,
):
    """
    Load STEP 02 high-pass arrays and stack
    them in SPORCO layout:

        H x W x N

    where N is the number of training images.
    """

    if num_samples <= 0:

        raise ValueError(
            "num_samples must be > 0"
        )

    if num_samples > len(
        train_manifest
    ):

        raise ValueError(
            f"Requested {num_samples} "
            f"training samples, but only "
            f"{len(train_manifest)} exist."
        )

    selected_manifest = (
        train_manifest.iloc[
            :num_samples
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    signals = []

    metadata_rows = []

    expected_shape = None

    for index, row in (
        selected_manifest.iterrows()
    ):

        sample_id = str(
            row["sample_id"]
        )

        signal = load_highpass(
            sample_id
        )

        original_shape = (
            signal.shape
        )

        model_signal = (
            resize_highpass(
                signal,
                scale=scale,
            )
        )

        if expected_shape is None:

            expected_shape = (
                model_signal.shape
            )

        if (
            model_signal.shape
            != expected_shape
        ):

            raise ValueError(
                f"{sample_id}: "
                f"model signal shape "
                f"{model_signal.shape} "
                f"!= {expected_shape}"
            )

        signals.append(
            model_signal
        )

        metadata_rows.append(
            {
                "order":
                    index,

                "sample_id":
                    sample_id,

                "split":
                    row["split"],

                "original_height":
                    original_shape[0],

                "original_width":
                    original_shape[1],

                "model_height":
                    model_signal.shape[0],

                "model_width":
                    model_signal.shape[1],

                "scale":
                    scale,

                "highpass_min":
                    float(
                        model_signal.min()
                    ),

                "highpass_max":
                    float(
                        model_signal.max()
                    ),

                "highpass_mean":
                    float(
                        model_signal.mean()
                    ),

                "highpass_std":
                    float(
                        model_signal.std()
                    ),
            }
        )

    S = np.stack(
        signals,
        axis=2,
    ).astype(
        np.float32,
        copy=False,
    )

    metadata = pd.DataFrame(
        metadata_rows
    )

    return (
        S,
        metadata,
    )


# ============================================================
# INITIAL DICTIONARY
# ============================================================

def initialize_dictionary(
    filter_size: int,
    num_filters: int,
    seed: int,
) -> np.ndarray:
    """
    Create reproducible random initial
    dictionary.

    Shape:
        filter_size x filter_size x K

    Each filter is:
        - zero mean
        - unit L2 norm
    """

    if filter_size <= 0:

        raise ValueError(
            "filter_size must be > 0"
        )

    if num_filters <= 0:

        raise ValueError(
            "num_filters must be > 0"
        )

    rng = np.random.default_rng(
        seed
    )

    dictionary = (
        rng.standard_normal(
            (
                filter_size,
                filter_size,
                num_filters,
            )
        )
        .astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # ZERO MEAN
    # --------------------------------------------------------

    dictionary -= (
        dictionary.mean(
            axis=(0, 1),
            keepdims=True,
        )
    )

    # --------------------------------------------------------
    # UNIT NORM
    # --------------------------------------------------------

    norms = np.sqrt(
        np.sum(
            dictionary ** 2,
            axis=(0, 1),
            keepdims=True,
        )
    )

    norms = np.maximum(
        norms,
        1e-12,
    )

    dictionary /= norms

    return dictionary.astype(
        np.float32
    )


# ============================================================
# DICTIONARY FORMATTING
# ============================================================

def squeeze_dictionary(
    dictionary: np.ndarray,
) -> np.ndarray:
    """
    Convert SPORCO dictionary output to:

        H x W x K
    """

    D = np.asarray(
        dictionary
    )

    D = np.squeeze(
        D
    )

    if D.ndim == 2:

        D = D[
            ...,
            np.newaxis
        ]

    if D.ndim != 3:

        raise ValueError(
            f"Expected dictionary "
            f"with 3 dimensions after "
            f"squeeze, found {D.shape}"
        )

    return D.astype(
        np.float32,
        copy=False,
    )


# ============================================================
# DICTIONARY STATISTICS
# ============================================================

def dictionary_statistics(
    dictionary: np.ndarray,
) -> dict:
    """
    Compute filter norm and mean statistics.
    """

    D = squeeze_dictionary(
        dictionary
    )

    norms = np.sqrt(
        np.sum(
            D ** 2,
            axis=(0, 1),
        )
    )

    means = D.mean(
        axis=(0, 1)
    )

    return {
        "num_filters":
            int(D.shape[-1]),

        "filter_height":
            int(D.shape[0]),

        "filter_width":
            int(D.shape[1]),

        "norm_min":
            float(
                norms.min()
            ),

        "norm_max":
            float(
                norms.max()
            ),

        "norm_mean":
            float(
                norms.mean()
            ),

        "abs_mean_filter_mean":
            float(
                np.abs(
                    means
                ).mean()
            ),

        "finite":
            bool(
                np.all(
                    np.isfinite(D)
                )
            ),
    }


# ============================================================
# ITERATION STATISTICS
# ============================================================

def iteration_stats_to_dataframe(
    iteration_stats,
) -> pd.DataFrame:
    """
    Convert SPORCO getitstat() output
    to DataFrame.
    """

    if not hasattr(
        iteration_stats,
        "_fields",
    ):

        raise TypeError(
            "Unexpected SPORCO "
            "iteration statistics format."
        )

    data = {}

    for field in (
        iteration_stats._fields
    ):

        value = np.asarray(
            getattr(
                iteration_stats,
                field,
            )
        )

        if value.ndim == 1:

            data[field] = value

    return pd.DataFrame(
        data
    )


# ============================================================
# DICTIONARY VISUALIZATION
# ============================================================

def save_dictionary_grid(
    dictionary: np.ndarray,
    output_path,
    title: str,
):
    """
    Save tiled visualization of dictionary
    filters.
    """

    D = squeeze_dictionary(
        dictionary
    )

    num_filters = D.shape[-1]

    columns = int(
        math.ceil(
            math.sqrt(
                num_filters
            )
        )
    )

    rows = int(
        math.ceil(
            num_filters
            / columns
        )
    )

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            2.2 * columns,
            2.2 * rows,
        ),
        squeeze=False,
    )

    absolute_max = float(
        np.max(
            np.abs(D)
        )
    )

    if absolute_max == 0.0:

        absolute_max = 1.0

    for filter_index in range(
        rows * columns
    ):

        axis = axes.flat[
            filter_index
        ]

        axis.axis(
            "off"
        )

        if filter_index >= num_filters:

            continue

        axis.imshow(
            D[
                ...,
                filter_index
            ],
            cmap="gray",
            vmin=-absolute_max,
            vmax=absolute_max,
        )

        axis.set_title(
            f"d{filter_index + 1}"
        )

    figure.suptitle(
        title,
        fontsize=16,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_dictionary_comparison(
    initial_dictionary: np.ndarray,
    final_dictionary: np.ndarray,
    output_path,
):
    """
    Save initial vs learned filters.
    """

    D0 = squeeze_dictionary(
        initial_dictionary
    )

    D1 = squeeze_dictionary(
        final_dictionary
    )

    num_filters = min(
        D0.shape[-1],
        D1.shape[-1],
    )

    figure, axes = plt.subplots(
        2,
        num_filters,
        figsize=(
            max(
                12,
                1.5 * num_filters,
            ),
            4,
        ),
        squeeze=False,
    )

    absolute_max = max(
        float(
            np.max(
                np.abs(D0)
            )
        ),
        float(
            np.max(
                np.abs(D1)
            )
        ),
    )

    if absolute_max == 0.0:

        absolute_max = 1.0

    for k in range(
        num_filters
    ):

        axes[0, k].imshow(
            D0[..., k],
            cmap="gray",
            vmin=-absolute_max,
            vmax=absolute_max,
        )

        axes[0, k].axis(
            "off"
        )

        axes[0, k].set_title(
            f"d{k + 1}"
        )

        axes[1, k].imshow(
            D1[..., k],
            cmap="gray",
            vmin=-absolute_max,
            vmax=absolute_max,
        )

        axes[1, k].axis(
            "off"
        )

    axes[0, 0].set_ylabel(
        "Initial"
    )

    axes[1, 0].set_ylabel(
        "Learned"
    )

    figure.suptitle(
        "STEP 04 — Initial vs Learned CSC Dictionary"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# TRAINING CURVES
# ============================================================

def save_training_curves(
    log_df: pd.DataFrame,
    output_path,
):
    """
    Save SPORCO optimization diagnostics.
    """

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
    )

    # --------------------------------------------------------
    # OBJECTIVE
    # --------------------------------------------------------

    if "ObjFun" in log_df:

        axes[0].plot(
            log_df["Iter"],
            log_df["ObjFun"],
        )

        axes[0].set_title(
            "Objective Function"
        )

        axes[0].set_xlabel(
            "Outer Iteration"
        )

        axes[0].set_ylabel(
            "Objective"
        )

    # --------------------------------------------------------
    # RESIDUALS
    # --------------------------------------------------------

    residual_fields = [
        "XPrRsdl",
        "XDlRsdl",
        "DPrRsdl",
        "DDlRsdl",
    ]

    for field in residual_fields:

        if field in log_df:

            values = (
                log_df[field]
                .to_numpy()
            )

            values = np.maximum(
                values,
                1e-16,
            )

            axes[1].semilogy(
                log_df["Iter"],
                values,
                label=field,
            )

    axes[1].set_title(
        "ADMM Residuals"
    )

    axes[1].set_xlabel(
        "Outer Iteration"
    )

    axes[1].legend()

    # --------------------------------------------------------
    # RHO
    # --------------------------------------------------------

    for field in [
        "XRho",
        "DRho",
    ]:

        if field in log_df:

            axes[2].semilogy(
                log_df["Iter"],
                log_df[field],
                label=field,
            )

    axes[2].set_title(
        "ADMM Penalty Parameters"
    )

    axes[2].set_xlabel(
        "Outer Iteration"
    )

    axes[2].legend()

    figure.suptitle(
        "STEP 04 — CSC Training Diagnostics"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )
    
# ============================================================
# DATA-DRIVEN LAMBDA ESTIMATION
# ============================================================

def estimate_cbpdn_lambda_max(
    dictionary: np.ndarray,
    signals: np.ndarray,
) -> float:
    """
    Estimate lambda_max for convolutional BPDN.

    For

        min_X 0.5 ||D X - S||_2^2
              + lambda ||X||_1

    the all-zero sparse representation is optimal
    when lambda is sufficiently large.

    A useful reference is

        lambda_max = ||D^T S||_infinity

    where D^T is the adjoint convolution operator.

    Parameters
    ----------
    dictionary:
        Initial dictionary, shape

            filter_height x filter_width x K

    signals:
        Training signals, shape

            H x W x N

    Returns
    -------
    float
        Estimated lambda_max over all filters,
        pixels, and training images.
    """

    D = squeeze_dictionary(
        dictionary
    )

    S = np.asarray(
        signals,
        dtype=np.float32,
    )

    if S.ndim != 3:

        raise ValueError(
            f"Expected training tensor H x W x N, "
            f"found shape {S.shape}"
        )

    height = S.shape[0]
    width = S.shape[1]

    filter_height = D.shape[0]
    filter_width = D.shape[1]
    num_filters = D.shape[2]

    if (
        filter_height > height
        or filter_width > width
    ):

        raise ValueError(
            "Dictionary filters are larger than "
            "the training signal."
        )

    # --------------------------------------------------------
    # ZERO-PAD FILTERS TO SIGNAL SIZE
    # --------------------------------------------------------

    padded_dictionary = np.zeros(
        (
            height,
            width,
            num_filters,
        ),
        dtype=np.float32,
    )

    padded_dictionary[
        :filter_height,
        :filter_width,
        :
    ] = D

    # --------------------------------------------------------
    # FFT OF TRAINING SIGNALS
    # --------------------------------------------------------

    signal_fft = spfft.rfft2(
        S,
        axes=(0, 1),
    )

    dictionary_fft = spfft.rfft2(
        padded_dictionary,
        axes=(0, 1),
    )

    lambda_max = 0.0

    # --------------------------------------------------------
    # ADJOINT CONVOLUTION D^T S
    #
    # Process one filter at a time to avoid creating one
    # extremely large K x N temporary array.
    # --------------------------------------------------------

    for filter_index in range(
        num_filters
    ):

        correlation_fft = (
            np.conj(
                dictionary_fft[
                    ...,
                    filter_index
                ]
            )[
                ...,
                np.newaxis
            ]
            *
            signal_fft
        )

        correlation = spfft.irfft2(
            correlation_fft,
            s=(
                height,
                width,
            ),
            axes=(0, 1),
        )

        current_max = float(
            np.max(
                np.abs(
                    correlation
                )
            )
        )

        lambda_max = max(
            lambda_max,
            current_max,
        )

        del correlation_fft
        del correlation

    if not np.isfinite(
        lambda_max
    ):

        raise ValueError(
            "Estimated lambda_max is NaN or Inf."
        )

    if lambda_max <= 0.0:

        raise ValueError(
            "Estimated lambda_max <= 0. "
            "Training signal or dictionary may be invalid."
        )

    return float(
        lambda_max
    )
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

from sklearn.linear_model import (
    LogisticRegression,
)

from sklearn.pipeline import (
    Pipeline,
)

from sklearn.preprocessing import (
    StandardScaler,
)

from sporco.admm import cbpdn


# ============================================================
# SPARSE MAP SHAPE
# ============================================================

def squeeze_sparse_maps(
    coefficient_array,
    num_filters,
    expected_spatial_shape,
):
    """
    Convert a SPORCO ConvBPDN coefficient array into

        H x W x K

    where K is the number of dictionary filters.
    """

    X = np.asarray(
        coefficient_array
    )

    X = np.squeeze(
        X
    )

    if X.ndim == 2:

        if num_filters != 1:

            raise ValueError(
                f"2D sparse array found but "
                f"K={num_filters}"
            )

        X = X[
            ...,
            np.newaxis
        ]

    if X.ndim != 3:

        raise ValueError(
            f"Expected 3D sparse maps after squeeze, "
            f"found {X.shape}"
        )

    # --------------------------------------------------------
    # FIND FILTER AXIS
    # --------------------------------------------------------

    if X.shape[-1] != num_filters:

        candidate_axes = [
            axis
            for axis, size in enumerate(
                X.shape
            )
            if size == num_filters
        ]

        if len(candidate_axes) != 1:

            raise ValueError(
                f"Unable to identify filter axis "
                f"in sparse map shape {X.shape}"
            )

        X = np.moveaxis(
            X,
            candidate_axes[0],
            -1,
        )

    if (
        X.shape[0]
        != expected_spatial_shape[0]
        or
        X.shape[1]
        != expected_spatial_shape[1]
    ):

        raise ValueError(
            f"Sparse map spatial shape "
            f"{X.shape[:2]} != expected "
            f"{expected_spatial_shape}"
        )

    return np.asarray(
        X,
        dtype=np.float32,
    )


# ============================================================
# SINGLE-IMAGE CSC INFERENCE
# ============================================================

def infer_sparse_maps(
    dictionary,
    signal,
    lmbda,
    rho,
    max_iter,
    rel_stop_tol,
):
    """
    Infer sparse coefficient maps Z for one image using
    ConvBPDN with the fixed CSC dictionary.

    Returns
    -------
    sparse_maps:
        H x W x K float32

    diagnostics:
        dict containing optimization statistics.
    """

    D = np.asarray(
        dictionary,
        dtype=np.float32,
    )

    S = np.asarray(
        signal,
        dtype=np.float32,
    )

    if D.ndim != 3:

        raise ValueError(
            f"Dictionary must be Hf x Wf x K, "
            f"found {D.shape}"
        )

    if S.ndim != 2:

        raise ValueError(
            f"Signal must be 2D, found {S.shape}"
        )

    num_filters = int(
        D.shape[-1]
    )

    options = (
        cbpdn.ConvBPDN.Options(
            {
                "Verbose":
                    False,

                "MaxMainIter":
                    int(max_iter),

                "RelStopTol":
                    float(rel_stop_tol),

                "AuxVarObj":
                    False,

                "HighMemSolve":
                    False,

                "rho":
                    float(rho),

                "AutoRho": {
                    "Enabled":
                        False,
                },
            }
        )
    )

    solver = cbpdn.ConvBPDN(
        D,
        S,
        float(lmbda),
        opt=options,
        dimK=0,
    )

    coefficient_array = (
        solver.solve()
    )

    solve_time = float(
        solver.timer.elapsed(
            "solve"
        )
    )

    sparse_maps = (
        squeeze_sparse_maps(
            coefficient_array,
            num_filters=
                num_filters,
            expected_spatial_shape=
                S.shape,
        )
    )

    stats = solver.getitstat()

    def last_value(
        field,
        default=np.nan,
    ):

        if not hasattr(
            stats,
            field,
        ):

            return float(
                default
            )

        values = np.asarray(
            getattr(
                stats,
                field,
            )
        )

        if len(values) == 0:

            return float(
                default
            )

        return float(
            values[-1]
        )

    iterations = 0

    if hasattr(
        stats,
        "Iter",
    ):

        iter_values = np.asarray(
            stats.Iter
        )

        if len(iter_values) > 0:

            iterations = int(
                iter_values[-1]
            ) + 1

    diagnostics = {
        "iterations":
            iterations,

        "solve_time_seconds":
            solve_time,

        "objective":
            last_value(
                "ObjFun"
            ),

        "data_fidelity":
            last_value(
                "DFid"
            ),

        "reg_l1":
            last_value(
                "RegL1"
            ),

        "primal_residual":
            last_value(
                "PrimalRsdl"
            ),

        "dual_residual":
            last_value(
                "DualRsdl"
            ),

        "rho":
            last_value(
                "Rho",
                rho,
            ),
    }

    return (
        sparse_maps,
        diagnostics,
    )


# ============================================================
# BILINEAR FEATURE SAMPLING
# ============================================================

def sample_sparse_features_bilinear(
    sparse_maps,
    coordinates,
    original_shape,
):
    """
    Sample CSC features at full-resolution supervision
    coordinates.

    Supervision coordinates from STEP 03 are stored as

        [row, col] = [y, x]

    at 1024 x 2048.

    Sparse maps are at model resolution, e.g.

        256 x 512.

    Bilinear interpolation is used instead of simply rounding
    coordinates. This avoids collapsing multiple nearby
    supervision points onto exactly the same model pixel.

    Returns
    -------
    features:
        N x K float32
    """

    Z = np.asarray(
        sparse_maps,
        dtype=np.float32,
    )

    coords = np.asarray(
        coordinates,
        dtype=np.float64,
    )

    if coords.ndim != 2:

        raise ValueError(
            "coordinates must have shape N x 2"
        )

    if coords.shape[1] != 2:

        raise ValueError(
            "coordinates must contain [row, col]"
        )

    if len(coords) == 0:

        return np.empty(
            (
                0,
                Z.shape[-1],
            ),
            dtype=np.float32,
        )

    original_height = int(
        original_shape[0]
    )

    original_width = int(
        original_shape[1]
    )

    model_height = int(
        Z.shape[0]
    )

    model_width = int(
        Z.shape[1]
    )

    # --------------------------------------------------------
    # PIXEL-CENTRE COORDINATE MAPPING
    # --------------------------------------------------------

    y = (
        (
            coords[:, 0]
            + 0.5
        )
        *
        model_height
        /
        original_height
        -
        0.5
    )

    x = (
        (
            coords[:, 1]
            + 0.5
        )
        *
        model_width
        /
        original_width
        -
        0.5
    )

    y = np.clip(
        y,
        0.0,
        model_height - 1.0,
    )

    x = np.clip(
        x,
        0.0,
        model_width - 1.0,
    )

    y0 = np.floor(
        y
    ).astype(
        np.int32
    )

    x0 = np.floor(
        x
    ).astype(
        np.int32
    )

    y1 = np.minimum(
        y0 + 1,
        model_height - 1,
    )

    x1 = np.minimum(
        x0 + 1,
        model_width - 1,
    )

    wy = (
        y - y0
    ).astype(
        np.float32
    )

    wx = (
        x - x0
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # FOUR NEIGHBOURS
    # --------------------------------------------------------

    f00 = Z[
        y0,
        x0,
        :
    ]

    f01 = Z[
        y0,
        x1,
        :
    ]

    f10 = Z[
        y1,
        x0,
        :
    ]

    f11 = Z[
        y1,
        x1,
        :
    ]

    w00 = (
        (1.0 - wy)
        *
        (1.0 - wx)
    )[
        :,
        np.newaxis
    ]

    w01 = (
        (1.0 - wy)
        *
        wx
    )[
        :,
        np.newaxis
    ]

    w10 = (
        wy
        *
        (1.0 - wx)
    )[
        :,
        np.newaxis
    ]

    w11 = (
        wy
        *
        wx
    )[
        :,
        np.newaxis
    ]

    features = (
        w00 * f00
        +
        w01 * f01
        +
        w10 * f10
        +
        w11 * f11
    )

    return np.asarray(
        features,
        dtype=np.float32,
    )


# ============================================================
# CLASSIFIER
# ============================================================

def build_logistic_classifier(
    C,
    max_iter,
):
    """
    StandardScaler + binary logistic regression.

    Labels:
        +1 = target class
        -1 = all other classes / background
    """

    pipeline = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),

            (
                "logistic",
                LogisticRegression(
                    C=float(C),
                    solver="lbfgs",
                    max_iter=int(max_iter),
                    class_weight=None,
                ),
            ),
        ]
    )

    return pipeline


# ============================================================
# CLASS SCORE MAP
# ============================================================

def classifier_score_map(
    classifier,
    sparse_maps,
):
    """
    Apply a binary classifier to every model-resolution
    sparse feature vector.

    Returns a H x W decision-score map.

    score > 0:
        target class side

    score < 0:
        non-target side
    """

    Z = np.asarray(
        sparse_maps,
        dtype=np.float32,
    )

    height = Z.shape[0]
    width = Z.shape[1]
    num_filters = Z.shape[2]

    flat_features = Z.reshape(
        -1,
        num_filters,
    )

    scores = (
        classifier
        .decision_function(
            flat_features
        )
    )

    return np.asarray(
        scores.reshape(
            height,
            width,
        ),
        dtype=np.float32,
    )


# ============================================================
# SPARSE MAP VISUALIZATION
# ============================================================

def save_sparse_map_grid(
    sparse_maps,
    output_path,
    sample_id,
):
    """
    Save |z_k| for all CSC filters.
    """

    Z = np.asarray(
        sparse_maps
    )

    num_filters = Z.shape[-1]

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
            /
            columns
        )
    )

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            columns * 3,
            rows * 2.2,
        ),
        squeeze=False,
    )

    magnitude = np.abs(
        Z
    )

    for k in range(
        rows * columns
    ):

        axis = axes.flat[
            k
        ]

        axis.axis(
            "off"
        )

        if k >= num_filters:

            continue

        values = magnitude[
            ...,
            k
        ]

        vmax = float(
            np.percentile(
                values,
                99.5,
            )
        )

        if vmax <= 0.0:

            vmax = 1.0

        axis.imshow(
            values,
            cmap="gray",
            vmin=0.0,
            vmax=vmax,
        )

        axis.set_title(
            f"|z{k + 1}|"
        )

    figure.suptitle(
        f"STEP 05 — CSC Sparse Coefficient Maps\n"
        f"{sample_id}",
        fontsize=16,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=130,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# CLASS-SCORE VISUALIZATION
# ============================================================

def save_class_score_visualization(
    image,
    person_score,
    car_score,
    motorcycle_score,
    output_path,
    sample_id,
):
    """
    Visualize raw One-vs-All decision score maps.

    These are NOT final thresholded localization maps.
    """

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(16, 9),
    )

    axes[0, 0].imshow(
        image
    )

    axes[0, 0].set_title(
        "Original RGB"
    )

    score_maps = [
        (
            "Person raw score",
            person_score,
            axes[0, 1],
        ),

        (
            "Car raw score",
            car_score,
            axes[1, 0],
        ),

        (
            "Motorcycle raw score",
            motorcycle_score,
            axes[1, 1],
        ),
    ]

    for title, score, axis in score_maps:

        limit = float(
            np.percentile(
                np.abs(
                    score
                ),
                99.0,
            )
        )

        if limit <= 0.0:

            limit = 1.0

        image_handle = axis.imshow(
            score,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )

        axis.set_title(
            title
        )

        figure.colorbar(
            image_handle,
            ax=axis,
            fraction=0.046,
            pad=0.04,
        )

    for axis in axes.flat:

        axis.axis(
            "off"
        )

    figure.suptitle(
        f"STEP 05 — Raw CSC Class Scores\n"
        f"{sample_id}",
        fontsize=16,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=130,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )
    
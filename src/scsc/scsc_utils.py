import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from scipy.fft import (
    rfft2,
    irfft2,
)

from scipy.optimize import (
    minimize,
)

from scipy.sparse.linalg import (
    LinearOperator,
    cg,
)

from scipy.special import (
    expit,
)

from sklearn.metrics import (
    average_precision_score,
)

from sporco.admm import (
    ccmod,
)

from src.csc.csc_utils import (
    squeeze_dictionary,
)

from src.csc.step05_utils import (
    sample_sparse_features_bilinear,
)


# ============================================================
# BASIC NUMERICAL UTILITIES
# ============================================================

def soft_threshold(
    values,
    threshold,
):
    values = np.asarray(
        values
    )

    return (
        np.sign(
            values
        )
        *
        np.maximum(
            np.abs(
                values
            )
            -
            threshold,
            0.0,
        )
    )


def relative_dictionary_change(
    initial_dictionary,
    current_dictionary,
):
    D0 = np.asarray(
        squeeze_dictionary(
            initial_dictionary
        ),
        dtype=np.float64,
    )

    D1 = np.asarray(
        squeeze_dictionary(
            current_dictionary
        ),
        dtype=np.float64,
    )

    if D0.shape != D1.shape:

        return float(
            "inf"
        )

    denominator = max(
        float(
            np.linalg.norm(
                D0
            )
        ),
        1e-12,
    )

    return float(
        np.linalg.norm(
            D1 - D0
        )
        /
        denominator
    )


# ============================================================
# CONVOLUTION OPERATORS
# ============================================================

def prepare_dictionary_fft(
    dictionary,
    spatial_shape,
):
    """
    Zero-pad cropped dictionary filters to the signal size
    and compute the 2-D real FFT.

    Returns
    -------
    Df:
        H x W_fft x K
    """

    D = np.asarray(
        squeeze_dictionary(
            dictionary
        ),
        dtype=np.float32,
    )

    height = int(
        spatial_shape[
            0
        ]
    )

    width = int(
        spatial_shape[
            1
        ]
    )

    filter_height = int(
        D.shape[
            0
        ]
    )

    filter_width = int(
        D.shape[
            1
        ]
    )

    num_filters = int(
        D.shape[
            2
        ]
    )

    if (
        filter_height > height
        or
        filter_width > width
    ):

        raise ValueError(
            "Dictionary filter support is "
            "larger than the signal."
        )

    padded = np.zeros(
        (
            height,
            width,
            num_filters,
        ),
        dtype=np.float32,
    )

    padded[
        :filter_height,
        :filter_width,
        :
    ] = D

    return rfft2(
        padded,
        axes=(
            0,
            1,
        ),
    )


def convolution_forward(
    sparse_maps,
    dictionary_fft,
    spatial_shape,
):
    """
    Compute:

        sum_k d_k * z_k
    """

    Z = np.asarray(
        sparse_maps,
        dtype=np.float32,
    )

    Zf = rfft2(
        Z,
        axes=(
            0,
            1,
        ),
    )

    reconstruction_fft = np.sum(
        dictionary_fft
        *
        Zf,
        axis=2,
    )

    reconstruction = irfft2(
        reconstruction_fft,
        s=
            spatial_shape,
        axes=(
            0,
            1,
        ),
    )

    return np.asarray(
        reconstruction,
        dtype=np.float32,
    )


def convolution_adjoint(
    signal,
    dictionary_fft,
    spatial_shape,
):
    """
    Compute:

        D^T signal
    """

    signal = np.asarray(
        signal,
        dtype=np.float32,
    )

    signal_fft = rfft2(
        signal,
        axes=(
            0,
            1,
        ),
    )

    result_fft = (
        np.conj(
            dictionary_fft
        )
        *
        signal_fft[
            ...,
            np.newaxis
        ]
    )

    result = irfft2(
        result_fft,
        s=
            spatial_shape,
        axes=(
            0,
            1,
        ),
    )

    return np.asarray(
        result,
        dtype=np.float32,
    )


# ============================================================
# SUPERVISION COORDINATE MAPPING
# ============================================================

def build_bilinear_mapping(
    coordinates,
    original_shape,
    model_shape,
):
    """
    Map STEP 03 full-resolution coordinates to CSC model
    resolution using pixel-centre alignment.
    """

    coords = np.asarray(
        coordinates,
        dtype=np.float64,
    )

    if (
        coords.ndim != 2
        or
        coords.shape[1] != 2
    ):

        raise ValueError(
            "coordinates must have shape "
            "N x 2 as [row, col]."
        )

    original_height = int(
        original_shape[
            0
        ]
    )

    original_width = int(
        original_shape[
            1
        ]
    )

    model_height = int(
        model_shape[
            0
        ]
    )

    model_width = int(
        model_shape[
            1
        ]
    )

    y = (
        (
            coords[
                :,
                0
            ]
            +
            0.5
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
            coords[
                :,
                1
            ]
            +
            0.5
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
        model_height
        -
        1.0,
    )

    x = np.clip(
        x,
        0.0,
        model_width
        -
        1.0,
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
        y
        -
        y0
    ).astype(
        np.float32
    )

    wx = (
        x
        -
        x0
    ).astype(
        np.float32
    )

    return {
        "y0":
            y0,

        "x0":
            x0,

        "y1":
            y1,

        "x1":
            x1,

        "w00":
            (
                (
                    1.0
                    -
                    wy
                )
                *
                (
                    1.0
                    -
                    wx
                )
            ).astype(
                np.float32
            ),

        "w01":
            (
                (
                    1.0
                    -
                    wy
                )
                *
                wx
            ).astype(
                np.float32
            ),

        "w10":
            (
                wy
                *
                (
                    1.0
                    -
                    wx
                )
            ).astype(
                np.float32
            ),

        "w11":
            (
                wy
                *
                wx
            ).astype(
                np.float32
            ),
    }


def build_image_supervision_records(
    supervision,
    class_names,
    class_factors,
    original_shape,
    model_shape,
):
    """
    Merge the three STEP 03 One-vs-All supervision sets
    into one record collection for a single image.
    """

    parts = []

    for class_id in (
        1,
        2,
        3,
    ):

        class_name = (
            class_names[
                class_id
            ]
        )

        coordinates = np.asarray(
            supervision[
                f"{class_name}_coords"
            ]
        )

        labels = np.asarray(
            supervision[
                f"{class_name}_labels"
            ],
            dtype=np.float32,
        )

        mapping = (
            build_bilinear_mapping(
                coordinates=
                    coordinates,

                original_shape=
                    original_shape,

                model_shape=
                    model_shape,
            )
        )

        count = len(
            labels
        )

        parts.append(
            {
                **mapping,

                "labels":
                    labels,

                "class_ids":
                    np.full(
                        count,
                        class_id,
                        dtype=np.int8,
                    ),

                "class_factors":
                    np.full(
                        count,
                        class_factors[
                            class_id
                        ],
                        dtype=np.float32,
                    ),
            }
        )

    if not parts:

        raise ValueError(
            "No supervision records "
            "were created."
        )

    records = {}

    for key in (
        parts[
            0
        ].keys()
    ):

        records[
            key
        ] = np.concatenate(
            [
                part[
                    key
                ]
                for part
                in parts
            ]
        )

    return records


# ============================================================
# SUPERVISION LINEAR OPERATOR W
# ============================================================

def interpolate_record_features(
    sparse_maps,
    records,
):
    """
    Bilinearly interpolate K-dimensional sparse features
    for all supervision records.
    """

    Z = np.asarray(
        sparse_maps,
        dtype=np.float32,
    )

    f00 = Z[
        records[
            "y0"
        ],
        records[
            "x0"
        ],
        :
    ]

    f01 = Z[
        records[
            "y0"
        ],
        records[
            "x1"
        ],
        :
    ]

    f10 = Z[
        records[
            "y1"
        ],
        records[
            "x0"
        ],
        :
    ]

    f11 = Z[
        records[
            "y1"
        ],
        records[
            "x1"
        ],
        :
    ]

    features = (
        records[
            "w00"
        ][
            :,
            np.newaxis
        ]
        *
        f00
        +
        records[
            "w01"
        ][
            :,
            np.newaxis
        ]
        *
        f01
        +
        records[
            "w10"
        ][
            :,
            np.newaxis
        ]
        *
        f10
        +
        records[
            "w11"
        ][
            :,
            np.newaxis
        ]
        *
        f11
    )

    return np.asarray(
        features,
        dtype=np.float32,
    )


def w_forward(
    sparse_maps,
    records,
    classifier_parameters,
):
    """
    Apply Wz for the class associated with every
    supervision record.

    Bias is deliberately not included here because it
    appears as a constant offset in the logistic term.
    """

    features = (
        interpolate_record_features(
            sparse_maps,
            records,
        )
    )

    class_index = (
        records[
            "class_ids"
        ].astype(
            np.int32
        )
        -
        1
    )

    weights = (
        classifier_parameters[
            "weights"
        ][
            class_index
        ]
    )

    values = np.einsum(
        "rk,rk->r",
        features,
        weights,
    )

    return np.asarray(
        values,
        dtype=np.float32,
    )


def w_adjoint(
    values,
    records,
    classifier_parameters,
    sparse_shape,
):
    """
    Apply W^T to record-space values.
    """

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    class_index = (
        records[
            "class_ids"
        ].astype(
            np.int32
        )
        -
        1
    )

    weights = (
        classifier_parameters[
            "weights"
        ][
            class_index
        ]
    )

    contribution = (
        values[
            :,
            np.newaxis
        ]
        *
        weights
    )

    output = np.zeros(
        sparse_shape,
        dtype=np.float32,
    )

    np.add.at(
        output,
        (
            records[
                "y0"
            ],
            records[
                "x0"
            ],
        ),
        (
            records[
                "w00"
            ][
                :,
                np.newaxis
            ]
            *
            contribution
        ),
    )

    np.add.at(
        output,
        (
            records[
                "y0"
            ],
            records[
                "x1"
            ],
        ),
        (
            records[
                "w01"
            ][
                :,
                np.newaxis
            ]
            *
            contribution
        ),
    )

    np.add.at(
        output,
        (
            records[
                "y1"
            ],
            records[
                "x0"
            ],
        ),
        (
            records[
                "w10"
            ][
                :,
                np.newaxis
            ]
            *
            contribution
        ),
    )

    np.add.at(
        output,
        (
            records[
                "y1"
            ],
            records[
                "x1"
            ],
        ),
        (
            records[
                "w11"
            ][
                :,
                np.newaxis
            ]
            *
            contribution
        ),
    )

    return output


# ============================================================
# LOGISTIC PROXIMAL OPERATOR
# ============================================================

def logistic_prox_newton(
    v,
    records,
    classifier_parameters,
    gamma,
    rho,
    max_iter,
    tolerance,
):
    """
    Proximal operator for:

        gamma * a_r *
        log(1 + exp(-y_r (u_r + b_c)))

    solved independently per supervision record using
    Newton's method.
    """

    values = np.asarray(
        v,
        dtype=np.float64,
    )

    labels = np.asarray(
        records[
            "labels"
        ],
        dtype=np.float64,
    )

    class_index = (
        records[
            "class_ids"
        ].astype(
            np.int32
        )
        -
        1
    )

    biases = np.asarray(
        classifier_parameters[
            "biases"
        ][
            class_index
        ],
        dtype=np.float64,
    )

    coefficient = (
        float(
            gamma
        )
        *
        np.asarray(
            records[
                "class_factors"
            ],
            dtype=np.float64,
        )
    )

    u = (
        values.copy()
    )

    for _ in range(
        int(
            max_iter
        )
    ):

        margin = (
            -labels
            *
            (
                u
                +
                biases
            )
        )

        probability = expit(
            margin
        )

        gradient = (
            float(
                rho
            )
            *
            (
                u
                -
                values
            )
            -
            coefficient
            *
            labels
            *
            probability
        )

        hessian = (
            float(
                rho
            )
            +
            coefficient
            *
            probability
            *
            (
                1.0
                -
                probability
            )
        )

        step = (
            gradient
            /
            np.maximum(
                hessian,
                1e-12,
            )
        )

        u -= (
            step
        )

        if (
            float(
                np.max(
                    np.abs(
                        step
                    )
                )
            )
            <
            tolerance
        ):

            break

    return np.asarray(
        u,
        dtype=np.float32,
    )


# ============================================================
# EXACT SUPERVISED Z OBJECTIVE — EQ. 6
# ============================================================

def z_subproblem_objective(
    sparse_maps,
    signal,
    dictionary_fft,
    spatial_shape,
    records,
    classifier_parameters,
    lmbda,
    gamma,
):
    """
    Exact Z-subproblem objective for fixed D and theta:

        0.5 ||S - DZ||^2
        + lambda ||Z||_1
        + gamma * L_logistic(Z, theta)

    theta regularisation is omitted because theta is fixed
    and is therefore constant with respect to Z.
    """

    Z = np.asarray(
        sparse_maps,
        dtype=np.float32,
    )

    reconstruction = (
        convolution_forward(
            Z,
            dictionary_fft,
            spatial_shape,
        )
    )

    residual = (
        signal
        -
        reconstruction
    )

    data_fidelity = (
        0.5
        *
        float(
            np.sum(
                residual
                ** 2
            )
        )
    )

    l1_penalty = (
        float(
            lmbda
        )
        *
        float(
            np.sum(
                np.abs(
                    Z
                )
            )
        )
    )

    class_index = (
        records[
            "class_ids"
        ].astype(
            np.int32
        )
        -
        1
    )

    labels = np.asarray(
        records[
            "labels"
        ],
        dtype=np.float64,
    )

    factors = np.asarray(
        records[
            "class_factors"
        ],
        dtype=np.float64,
    )

    biases = np.asarray(
        classifier_parameters[
            "biases"
        ][
            class_index
        ],
        dtype=np.float64,
    )

    linear_scores = (
        w_forward(
            Z,
            records,
            classifier_parameters,
        )
        .astype(
            np.float64
        )
    )

    scores = (
        linear_scores
        +
        biases
    )

    logistic_loss = float(
        np.sum(
            factors
            *
            np.logaddexp(
                0.0,
                -labels
                *
                scores,
            )
        )
    )

    return (
        data_fidelity
        +
        l1_penalty
        +
        float(
            gamma
        )
        *
        logistic_loss
    )


# ============================================================
# SUPERVISED Z UPDATE — EQ. 6
# ============================================================

def supervised_sparse_code_admm(
    dictionary,
    signal,
    initial_sparse_maps,
    records,
    classifier_parameters,
    lmbda,
    gamma,
    rho,
    admm_iterations,
    admm_tolerance,
    cg_max_iter,
    cg_tolerance,
    newton_iterations,
    newton_tolerance,
):
    """
    Solve the supervised coding subproblem with scaled ADMM.

    Variable splitting:

        U_rec = D Z
        U_l1  = Z
        U_cls = W Z

    The Z linear system is solved with conjugate gradient.

    IMPORTANT:
    ADMM does not require the raw primal objective to
    decrease at every internal iteration.

    Therefore this routine stores and returns the BEST
    primal Z encountered, including the input Z as a valid
    candidate.
    """

    signal = np.asarray(
        signal,
        dtype=np.float32,
    )

    z = np.asarray(
        initial_sparse_maps,
        dtype=np.float32,
    ).copy()

    spatial_shape = (
        int(
            signal.shape[
                0
            ]
        ),
        int(
            signal.shape[
                1
            ]
        ),
    )

    sparse_shape = (
        z.shape
    )

    dictionary_fft = (
        prepare_dictionary_fft(
            dictionary,
            spatial_shape,
        )
    )

    initial_subproblem_objective = (
        z_subproblem_objective(
            sparse_maps=
                z,

            signal=
                signal,

            dictionary_fft=
                dictionary_fft,

            spatial_shape=
                spatial_shape,

            records=
                records,

            classifier_parameters=
                classifier_parameters,

            lmbda=
                lmbda,

            gamma=
                gamma,
        )
    )

    best_subproblem_objective = float(
        initial_subproblem_objective
    )

    best_z = (
        z.copy()
    )

    best_iteration = 0

    # --------------------------------------------------------
    # INITIAL SPLIT VARIABLES
    # --------------------------------------------------------

    dz = convolution_forward(
        z,
        dictionary_fft,
        spatial_shape,
    )

    wz = w_forward(
        z,
        records,
        classifier_parameters,
    )

    u_rec = (
        dz.copy()
    )

    u_l1 = (
        z.copy()
    )

    u_cls = (
        wz.copy()
    )

    dual_rec = np.zeros_like(
        u_rec
    )

    dual_l1 = np.zeros_like(
        u_l1
    )

    dual_cls = np.zeros_like(
        u_cls
    )

    final_primal = np.inf
    final_dual = np.inf

    cg_nonconverged = 0

    completed_iterations = 0

    # --------------------------------------------------------
    # A = D^T D + I + W^T W
    # --------------------------------------------------------

    variable_size = int(
        z.size
    )

    def matvec(
        vector,
    ):

        candidate = np.asarray(
            vector,
            dtype=np.float32,
        ).reshape(
            sparse_shape
        )

        d_candidate = (
            convolution_forward(
                candidate,
                dictionary_fft,
                spatial_shape,
            )
        )

        dtd_candidate = (
            convolution_adjoint(
                d_candidate,
                dictionary_fft,
                spatial_shape,
            )
        )

        w_candidate = (
            w_forward(
                candidate,
                records,
                classifier_parameters,
            )
        )

        wtw_candidate = (
            w_adjoint(
                w_candidate,
                records,
                classifier_parameters,
                sparse_shape,
            )
        )

        result = (
            dtd_candidate
            +
            candidate
            +
            wtw_candidate
        )

        return np.asarray(
            result,
            dtype=np.float32,
        ).ravel()

    linear_operator = (
        LinearOperator(
            (
                variable_size,
                variable_size,
            ),
            matvec=
                matvec,
            dtype=
                np.float32,
        )
    )

    # --------------------------------------------------------
    # ADMM ITERATIONS
    # --------------------------------------------------------

    for admm_iteration in range(
        int(
            admm_iterations
        )
    ):

        old_u_rec = (
            u_rec.copy()
        )

        old_u_l1 = (
            u_l1.copy()
        )

        old_u_cls = (
            u_cls.copy()
        )

        # ====================================================
        # Z STEP
        # ====================================================

        rhs = (
            convolution_adjoint(
                u_rec
                -
                dual_rec,
                dictionary_fft,
                spatial_shape,
            )
            +
            (
                u_l1
                -
                dual_l1
            )
            +
            w_adjoint(
                u_cls
                -
                dual_cls,
                records,
                classifier_parameters,
                sparse_shape,
            )
        )

        solution, cg_info = cg(
            linear_operator,
            rhs.ravel(),
            x0=
                z.ravel(),
            rtol=
                float(
                    cg_tolerance
                ),
            atol=
                0.0,
            maxiter=
                int(
                    cg_max_iter
                ),
        )

        if cg_info != 0:

            cg_nonconverged += 1

        z = np.asarray(
            solution,
            dtype=np.float32,
        ).reshape(
            sparse_shape
        )

        # ====================================================
        # U_rec PROX
        # ====================================================

        dz = convolution_forward(
            z,
            dictionary_fft,
            spatial_shape,
        )

        v_rec = (
            dz
            +
            dual_rec
        )

        u_rec = (
            signal
            +
            float(
                rho
            )
            *
            v_rec
        ) / (
            1.0
            +
            float(
                rho
            )
        )

        # ====================================================
        # U_l1 PROX
        # ====================================================

        v_l1 = (
            z
            +
            dual_l1
        )

        u_l1 = (
            soft_threshold(
                v_l1,
                float(
                    lmbda
                )
                /
                float(
                    rho
                ),
            )
            .astype(
                np.float32
            )
        )

        # ====================================================
        # U_cls PROX
        # ====================================================

        wz = w_forward(
            z,
            records,
            classifier_parameters,
        )

        v_cls = (
            wz
            +
            dual_cls
        )

        u_cls = logistic_prox_newton(
            v=
                v_cls,

            records=
                records,

            classifier_parameters=
                classifier_parameters,

            gamma=
                gamma,

            rho=
                rho,

            max_iter=
                newton_iterations,

            tolerance=
                newton_tolerance,
        )

        # ====================================================
        # DUAL UPDATE
        # ====================================================

        residual_rec = (
            dz
            -
            u_rec
        )

        residual_l1 = (
            z
            -
            u_l1
        )

        residual_cls = (
            wz
            -
            u_cls
        )

        dual_rec += (
            residual_rec
        )

        dual_l1 += (
            residual_l1
        )

        dual_cls += (
            residual_cls
        )

        # ====================================================
        # RESIDUAL DIAGNOSTICS
        # ====================================================

        primal_sq = (
            float(
                np.sum(
                    residual_rec
                    ** 2
                )
            )
            +
            float(
                np.sum(
                    residual_l1
                    ** 2
                )
            )
            +
            float(
                np.sum(
                    residual_cls
                    ** 2
                )
            )
        )

        primal_count = (
            residual_rec.size
            +
            residual_l1.size
            +
            residual_cls.size
        )

        final_primal = math.sqrt(
            primal_sq
            /
            max(
                primal_count,
                1,
            )
        )

        delta_rec = (
            u_rec
            -
            old_u_rec
        )

        delta_l1 = (
            u_l1
            -
            old_u_l1
        )

        delta_cls = (
            u_cls
            -
            old_u_cls
        )

        dual_z = (
            convolution_adjoint(
                delta_rec,
                dictionary_fft,
                spatial_shape,
            )
            +
            delta_l1
            +
            w_adjoint(
                delta_cls,
                records,
                classifier_parameters,
                sparse_shape,
            )
        )

        final_dual = (
            float(
                rho
            )
            *
            float(
                np.linalg.norm(
                    dual_z.ravel()
                )
            )
            /
            math.sqrt(
                max(
                    dual_z.size,
                    1,
                )
            )
        )

        completed_iterations = (
            admm_iteration
            +
            1
        )

        # ====================================================
        # OBJECTIVE SAFEGUARD
        # ====================================================

        candidate_objective = (
            z_subproblem_objective(
                sparse_maps=
                    z,

                signal=
                    signal,

                dictionary_fft=
                    dictionary_fft,

                spatial_shape=
                    spatial_shape,

                records=
                    records,

                classifier_parameters=
                    classifier_parameters,

                lmbda=
                    lmbda,

                gamma=
                    gamma,
            )
        )

        if (
            candidate_objective
            <
            best_subproblem_objective
        ):

            best_subproblem_objective = float(
                candidate_objective
            )

            best_z = (
                z.copy()
            )

            best_iteration = (
                admm_iteration
                +
                1
            )

        if (
            final_primal
            <
            admm_tolerance
            and
            final_dual
            <
            admm_tolerance
        ):

            break

    # --------------------------------------------------------
    # IMPORTANT:
    # Return best primal Z instead of blindly returning
    # final internal ADMM iterate.
    # --------------------------------------------------------

    z = (
        best_z
    )

    reconstruction = (
        convolution_forward(
            z,
            dictionary_fft,
            spatial_shape,
        )
    )

    data_fidelity = (
        0.5
        *
        float(
            np.sum(
                (
                    signal
                    -
                    reconstruction
                )
                ** 2
            )
        )
    )

    l1_norm = float(
        np.sum(
            np.abs(
                z
            )
        )
    )

    diagnostics = {
        "iterations":
            int(
                completed_iterations
            ),

        "primal_residual":
            float(
                final_primal
            ),

        "dual_residual":
            float(
                final_dual
            ),

        "data_fidelity":
            data_fidelity,

        "l1_norm":
            l1_norm,

        "cg_nonconverged_steps":
            int(
                cg_nonconverged
            ),

        "initial_subproblem_objective":
            float(
                initial_subproblem_objective
            ),

        "best_subproblem_objective":
            float(
                best_subproblem_objective
            ),

        "objective_improvement":
            float(
                initial_subproblem_objective
                -
                best_subproblem_objective
            ),

        "best_iteration":
            int(
                best_iteration
            ),
    }

    return (
        np.asarray(
            z,
            dtype=np.float32,
        ),
        diagnostics,
    )


# ============================================================
# INTERNAL SCSC CLASSIFIERS — EQ. 7
# ============================================================

def fit_internal_classifiers(
    sparse_maps_list,
    supervision_list,
    original_shapes,
    class_names,
    class_factors,
    alpha,
    max_iter,
    gradient_tolerance,
    initial_parameters=None,
):
    """
    Fit three One-vs-All linear logistic classifiers used
    internally during SCSC coordinate descent.

    These are INTERNAL training parameters and are NOT
    final evaluation classifiers.
    """

    num_filters = int(
        sparse_maps_list[
            0
        ].shape[
            -1
        ]
    )

    weights = np.zeros(
        (
            3,
            num_filters,
        ),
        dtype=np.float64,
    )

    biases = np.zeros(
        3,
        dtype=np.float64,
    )

    if initial_parameters is not None:

        weights[:] = np.asarray(
            initial_parameters[
                "weights"
            ],
            dtype=np.float64,
        )

        biases[:] = np.asarray(
            initial_parameters[
                "biases"
            ],
            dtype=np.float64,
        )

    diagnostics = {}

    for class_id in (
        1,
        2,
        3,
    ):

        class_name = (
            class_names[
                class_id
            ]
        )

        feature_parts = []
        label_parts = []

        for (
            sparse_maps,
            supervision,
            original_shape,
        ) in zip(
            sparse_maps_list,
            supervision_list,
            original_shapes,
        ):

            coordinates = (
                supervision[
                    f"{class_name}_coords"
                ]
            )

            labels = np.asarray(
                supervision[
                    f"{class_name}_labels"
                ],
                dtype=np.float64,
            )

            features = (
                sample_sparse_features_bilinear(
                    sparse_maps=
                        sparse_maps,

                    coordinates=
                        coordinates,

                    original_shape=
                        original_shape,
                )
            )

            feature_parts.append(
                features
            )

            label_parts.append(
                labels
            )

        X = np.vstack(
            feature_parts
        ).astype(
            np.float64
        )

        y = np.concatenate(
            label_parts
        ).astype(
            np.float64
        )

        factor = float(
            class_factors[
                class_id
            ]
        )

        initial_theta = np.concatenate(
            [
                weights[
                    class_id
                    -
                    1
                ],
                [
                    biases[
                        class_id
                        -
                        1
                    ]
                ],
            ]
        )

        def objective_and_gradient(
            theta,
        ):

            w = (
                theta[
                    :-1
                ]
            )

            b = float(
                theta[
                    -1
                ]
            )

            scores = (
                X
                @
                w
                +
                b
            )

            margin = (
                -y
                *
                scores
            )

            logistic_values = (
                np.logaddexp(
                    0.0,
                    margin,
                )
            )

            probability = expit(
                margin
            )

            logistic_loss = (
                factor
                *
                float(
                    np.sum(
                        logistic_values
                    )
                )
            )

            regularization = (
                float(
                    alpha
                )
                *
                (
                    float(
                        np.dot(
                            w,
                            w,
                        )
                    )
                    +
                    b
                    *
                    b
                )
            )

            loss = (
                logistic_loss
                +
                regularization
            )

            common = (
                factor
                *
                (
                    -y
                    *
                    probability
                )
            )

            gradient_w = (
                X.T
                @
                common
                +
                2.0
                *
                float(
                    alpha
                )
                *
                w
            )

            gradient_b = (
                float(
                    np.sum(
                        common
                    )
                )
                +
                2.0
                *
                float(
                    alpha
                )
                *
                b
            )

            gradient = np.concatenate(
                [
                    gradient_w,
                    [
                        gradient_b
                    ],
                ]
            )

            return (
                loss,
                gradient,
            )

        result = minimize(
            fun=lambda theta:
                objective_and_gradient(
                    theta
                ),

            x0=
                initial_theta,

            method=
                "L-BFGS-B",

            jac=
                True,

            options={
                "maxiter":
                    int(
                        max_iter
                    ),

                "gtol":
                    float(
                        gradient_tolerance
                    ),
            },
        )

        theta = (
            result.x
        )

        weights[
            class_id
            -
            1
        ] = (
            theta[
                :-1
            ]
        )

        biases[
            class_id
            -
            1
        ] = (
            theta[
                -1
            ]
        )

        scores = (
            X
            @
            weights[
                class_id
                -
                1
            ]
            +
            biases[
                class_id
                -
                1
            ]
        )

        y_binary = (
            y
            ==
            1
        ).astype(
            np.int8
        )

        ap = float(
            average_precision_score(
                y_binary,
                scores,
            )
        )

        weighted_logistic_loss = (
            factor
            *
            float(
                np.sum(
                    np.logaddexp(
                        0.0,
                        -y
                        *
                        scores,
                    )
                )
            )
        )

        diagnostics[
            class_id
        ] = {
            "class_name":
                class_name,

            "samples":
                int(
                    len(
                        y
                    )
                ),

            "ap":
                ap,

            "weighted_logistic_loss":
                weighted_logistic_loss,

            "optimizer_success":
                bool(
                    result.success
                ),

            "optimizer_iterations":
                int(
                    result.nit
                ),

            "weight_norm":
                float(
                    np.linalg.norm(
                        weights[
                            class_id
                            -
                            1
                        ]
                    )
                ),

            "bias":
                float(
                    biases[
                        class_id
                        -
                        1
                    ]
                ),
        }

    parameters = {
        "weights":
            np.asarray(
                weights,
                dtype=np.float32,
            ),

        "biases":
            np.asarray(
                biases,
                dtype=np.float32,
            ),
    }

    return (
        parameters,
        diagnostics,
    )


# ============================================================
# OBJECTIVE EVALUATION
# ============================================================

def reconstruction_and_sparsity(
    dictionary,
    training_signals,
    sparse_maps_list,
    lmbda,
):
    """
    Compute CSC reconstruction + sparsity objective
    over all training images.
    """

    spatial_shape = (
        int(
            training_signals.shape[
                0
            ]
        ),
        int(
            training_signals.shape[
                1
            ]
        ),
    )

    dictionary_fft = (
        prepare_dictionary_fft(
            dictionary,
            spatial_shape,
        )
    )

    reconstruction_loss = 0.0
    l1_norm = 0.0

    for (
        image_index,
        sparse_maps,
    ) in enumerate(
        sparse_maps_list
    ):

        reconstruction = (
            convolution_forward(
                sparse_maps,
                dictionary_fft,
                spatial_shape,
            )
        )

        residual = (
            training_signals[
                ...,
                image_index
            ]
            -
            reconstruction
        )

        reconstruction_loss += (
            0.5
            *
            float(
                np.sum(
                    residual
                    ** 2
                )
            )
        )

        l1_norm += float(
            np.sum(
                np.abs(
                    sparse_maps
                )
            )
        )

    sparsity_penalty = (
        float(
            lmbda
        )
        *
        l1_norm
    )

    return {
        "data_fidelity":
            reconstruction_loss,

        "l1_norm":
            l1_norm,

        "sparsity_penalty":
            sparsity_penalty,

        "base_objective":
            (
                reconstruction_loss
                +
                sparsity_penalty
            ),
    }


def classification_loss_from_maps(
    sparse_maps_list,
    supervision_list,
    original_shapes,
    class_names,
    class_factors,
    classifier_parameters,
):
    """
    Compute weighted One-vs-All logistic loss for the
    current Z and theta over all training supervision
    points.
    """

    total_loss = 0.0

    per_class_loss = {}

    for class_id in (
        1,
        2,
        3,
    ):

        class_name = (
            class_names[
                class_id
            ]
        )

        factor = float(
            class_factors[
                class_id
            ]
        )

        class_loss = 0.0

        w = np.asarray(
            classifier_parameters[
                "weights"
            ][
                class_id
                -
                1
            ],
            dtype=np.float64,
        )

        b = float(
            classifier_parameters[
                "biases"
            ][
                class_id
                -
                1
            ]
        )

        for (
            sparse_maps,
            supervision,
            original_shape,
        ) in zip(
            sparse_maps_list,
            supervision_list,
            original_shapes,
        ):

            coordinates = (
                supervision[
                    f"{class_name}_coords"
                ]
            )

            labels = np.asarray(
                supervision[
                    f"{class_name}_labels"
                ],
                dtype=np.float64,
            )

            features = (
                sample_sparse_features_bilinear(
                    sparse_maps=
                        sparse_maps,

                    coordinates=
                        coordinates,

                    original_shape=
                        original_shape,
                )
                .astype(
                    np.float64
                )
            )

            scores = (
                features
                @
                w
                +
                b
            )

            class_loss += (
                factor
                *
                float(
                    np.sum(
                        np.logaddexp(
                            0.0,
                            -labels
                            *
                            scores,
                        )
                    )
                )
            )

        per_class_loss[
            class_id
        ] = (
            class_loss
        )

        total_loss += (
            class_loss
        )

    return (
        total_loss,
        per_class_loss,
    )


def evaluate_internal_classifier_diagnostics(
    sparse_maps_list,
    supervision_list,
    original_shapes,
    class_names,
    class_factors,
    classifier_parameters,
):
    """
    Evaluate AP and weighted logistic loss for current
    internal SCSC classifiers WITHOUT changing parameters.
    """

    diagnostics = {}

    for class_id in (
        1,
        2,
        3,
    ):

        class_name = (
            class_names[
                class_id
            ]
        )

        factor = float(
            class_factors[
                class_id
            ]
        )

        feature_parts = []
        label_parts = []

        for (
            sparse_maps,
            supervision,
            original_shape,
        ) in zip(
            sparse_maps_list,
            supervision_list,
            original_shapes,
        ):

            coordinates = (
                supervision[
                    f"{class_name}_coords"
                ]
            )

            labels = np.asarray(
                supervision[
                    f"{class_name}_labels"
                ],
                dtype=np.float64,
            )

            features = (
                sample_sparse_features_bilinear(
                    sparse_maps=
                        sparse_maps,

                    coordinates=
                        coordinates,

                    original_shape=
                        original_shape,
                )
            )

            feature_parts.append(
                features
            )

            label_parts.append(
                labels
            )

        X = np.vstack(
            feature_parts
        ).astype(
            np.float64
        )

        y = np.concatenate(
            label_parts
        ).astype(
            np.float64
        )

        w = np.asarray(
            classifier_parameters[
                "weights"
            ][
                class_id
                -
                1
            ],
            dtype=np.float64,
        )

        b = float(
            classifier_parameters[
                "biases"
            ][
                class_id
                -
                1
            ]
        )

        scores = (
            X
            @
            w
            +
            b
        )

        y_binary = (
            y
            ==
            1
        ).astype(
            np.int8
        )

        ap = float(
            average_precision_score(
                y_binary,
                scores,
            )
        )

        weighted_logistic_loss = (
            factor
            *
            float(
                np.sum(
                    np.logaddexp(
                        0.0,
                        -y
                        *
                        scores,
                    )
                )
            )
        )

        diagnostics[
            class_id
        ] = {
            "class_name":
                class_name,

            "samples":
                int(
                    len(
                        y
                    )
                ),

            "ap":
                ap,

            "weighted_logistic_loss":
                weighted_logistic_loss,

            "optimizer_success":
                True,

            "optimizer_iterations":
                0,

            "weight_norm":
                float(
                    np.linalg.norm(
                        w
                    )
                ),

            "bias":
                b,
        }

    return diagnostics


def evaluate_scsc_objective(
    dictionary,
    training_signals,
    sparse_maps_list,
    lmbda,
    supervision_list,
    original_shapes,
    class_names,
    class_factors,
    classifier_parameters,
    gamma,
    alpha,
):
    """
    Evaluate the SAME full SCSC objective for any
    current (D, Z, theta) state.

    Used for:
        - stage safeguards
        - logging
        - final validation
    """

    base = reconstruction_and_sparsity(
        dictionary=
            dictionary,

        training_signals=
            training_signals,

        sparse_maps_list=
            sparse_maps_list,

        lmbda=
            lmbda,
    )

    (
        classification_loss,
        per_class_loss,
    ) = classification_loss_from_maps(
        sparse_maps_list=
            sparse_maps_list,

        supervision_list=
            supervision_list,

        original_shapes=
            original_shapes,

        class_names=
            class_names,

        class_factors=
            class_factors,

        classifier_parameters=
            classifier_parameters,
    )

    theta_norm_squared = (
        float(
            np.sum(
                np.asarray(
                    classifier_parameters[
                        "weights"
                    ],
                    dtype=np.float64,
                )
                ** 2
            )
        )
        +
        float(
            np.sum(
                np.asarray(
                    classifier_parameters[
                        "biases"
                    ],
                    dtype=np.float64,
                )
                ** 2
            )
        )
    )

    classifier_regularization = (
        float(
            alpha
        )
        *
        theta_norm_squared
    )

    supervised_term = (
        classification_loss
        +
        classifier_regularization
    )

    total_objective = (
        base[
            "base_objective"
        ]
        +
        float(
            gamma
        )
        *
        supervised_term
    )

    return {
        **base,

        "classification_loss":
            classification_loss,

        "person_classification_loss":
            float(
                per_class_loss[
                    1
                ]
            ),

        "car_classification_loss":
            float(
                per_class_loss[
                    2
                ]
            ),

        "motorcycle_classification_loss":
            float(
                per_class_loss[
                    3
                ]
            ),

        "theta_norm_squared":
            theta_norm_squared,

        "classifier_regularization":
            classifier_regularization,

        "supervised_term":
            supervised_term,

        "total_objective":
            total_objective,
    }


# ============================================================
# DICTIONARY UPDATE — EQ. 2
# ============================================================

def pad_dictionary_for_ccmod(
    dictionary,
    spatial_shape,
):
    """
    Convert cropped Hf x Wf x K dictionary into SPORCO
    full dictionary layout:

        H x W x 1 x 1 x K
    """

    D = np.asarray(
        squeeze_dictionary(
            dictionary
        ),
        dtype=np.float32,
    )

    height = int(
        spatial_shape[
            0
        ]
    )

    width = int(
        spatial_shape[
            1
        ]
    )

    output = np.zeros(
        (
            height,
            width,
            1,
            1,
            D.shape[
                -1
            ],
        ),
        dtype=np.float32,
    )

    output[
        :D.shape[
            0
        ],
        :D.shape[
            1
        ],
        0,
        0,
        :
    ] = D

    return output


def update_dictionary_sporco(
    sparse_maps_list,
    training_signals,
    current_dictionary,
    filter_size,
    num_filters,
    max_iter,
    rho,
):
    """
    Solve SCSC dictionary subproblem using SPORCO
    Consensus ADMM.

    With Z fixed, the supervised term does not depend
    directly on D, so this is the same constrained MOD
    dictionary subproblem as CSC.
    """

    coefficient_stack = np.stack(
        sparse_maps_list,
        axis=2,
    ).astype(
        np.float32
    )

    # SPORCO layout:
    # H x W x C x N x K
    coefficient_stack = (
        coefficient_stack[
            :,
            :,
            np.newaxis,
            :,
            :
        ]
    )

    spatial_shape = (
        int(
            training_signals.shape[
                0
            ]
        ),
        int(
            training_signals.shape[
                1
            ]
        ),
    )

    initial_y = (
        pad_dictionary_for_ccmod(
            current_dictionary,
            spatial_shape,
        )
    )

    options = (
        ccmod.ConvCnstrMODOptions(
            {
                "Verbose":
                    False,

                "MaxMainIter":
                    int(
                        max_iter
                    ),

                "rho":
                    float(
                        rho
                    ),

                "ZeroMean":
                    True,

                "Y0":
                    initial_y,

                "AutoRho": {
                    "Enabled":
                        False,
                },
            },

            method=
                "cns",
        )
    )

    solver = ccmod.ConvCnstrMOD(
        coefficient_stack,
        np.asarray(
            training_signals,
            dtype=np.float32,
        ),
        (
            int(
                filter_size
            ),
            int(
                filter_size
            ),
            int(
                num_filters
            ),
        ),
        opt=
            options,
        method=
            "cns",
        dimK=
            1,
        dimN=
            2,
    )

    solver.solve()

    dictionary = (
        squeeze_dictionary(
            solver.getdict(
                crop=True
            )
        )
        .astype(
            np.float32
        )
    )

    stats = (
        solver.getitstat()
    )

    def last(
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

        if len(
            values
        ) == 0:

            return float(
                default
            )

        return float(
            values[
                -1
            ]
        )

    diagnostics = {
        "data_fidelity":
            last(
                "DFid"
            ),

        "constraint":
            last(
                "Cnstr"
            ),

        "primal_residual":
            last(
                "PrimalRsdl"
            ),

        "dual_residual":
            last(
                "DualRsdl"
            ),

        "rho":
            last(
                "Rho",
                rho,
            ),
    }

    del (
        coefficient_stack
    )

    return (
        dictionary,
        diagnostics,
    )


# ============================================================
# VISUALISATIONS
# ============================================================

def save_scsc_training_curves(
    log_df,
    output_path,
):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(
            14,
            10,
        ),
    )

    iteration = (
        log_df[
            "outer_iteration"
        ]
    )

    axes[
        0,
        0
    ].plot(
        iteration,
        log_df[
            "total_objective"
        ],
        label=
            "Total",
    )

    axes[
        0,
        0
    ].plot(
        iteration,
        log_df[
            "base_objective"
        ],
        label=
            "CSC part",
    )

    axes[
        0,
        0
    ].set_title(
        "SCSC Objective"
    )

    axes[
        0,
        0
    ].set_xlabel(
        "Outer iteration"
    )

    axes[
        0,
        0
    ].legend()

    axes[
        0,
        1
    ].plot(
        iteration,
        log_df[
            "classification_loss"
        ],
    )

    axes[
        0,
        1
    ].set_title(
        "Weighted Logistic Loss"
    )

    axes[
        0,
        1
    ].set_xlabel(
        "Outer iteration"
    )

    axes[
        1,
        0
    ].plot(
        iteration,
        log_df[
            "person_ap"
        ],
        label=
            "Person",
    )

    axes[
        1,
        0
    ].plot(
        iteration,
        log_df[
            "car_ap"
        ],
        label=
            "Car",
    )

    axes[
        1,
        0
    ].plot(
        iteration,
        log_df[
            "motorcycle_ap"
        ],
        label=
            "Motorcycle",
    )

    axes[
        1,
        0
    ].set_ylim(
        0.0,
        1.0,
    )

    axes[
        1,
        0
    ].set_title(
        "Internal Training AP"
    )

    axes[
        1,
        0
    ].set_xlabel(
        "Outer iteration"
    )

    axes[
        1,
        0
    ].legend()

    axes[
        1,
        1
    ].plot(
        iteration,
        log_df[
            "dictionary_change"
        ],
    )

    axes[
        1,
        1
    ].set_title(
        "Dictionary Change from D0"
    )

    axes[
        1,
        1
    ].set_xlabel(
        "Outer iteration"
    )

    figure.suptitle(
        "STEP 06 — SCSC Training Diagnostics",
        fontsize=16,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches=
            "tight",
    )

    plt.close(
        figure
    )


def save_scsc_dictionary_comparison(
    initial_dictionary,
    final_dictionary,
    output_path,
):
    """
    STEP 06-specific comparison plot.

    This avoids modifying the STEP 04 visualization helper.
    """

    D0 = np.asarray(
        squeeze_dictionary(
            initial_dictionary
        )
    )

    D1 = np.asarray(
        squeeze_dictionary(
            final_dictionary
        )
    )

    if D0.shape != D1.shape:

        raise ValueError(
            f"Dictionary shapes do not match: "
            f"{D0.shape} vs {D1.shape}"
        )

    num_filters = int(
        D0.shape[
            -1
        ]
    )

    figure, axes = plt.subplots(
        2,
        num_filters,
        figsize=(
            max(
                16,
                num_filters
                *
                1.4,
            ),
            4.2,
        ),
        squeeze=False,
    )

    for filter_index in range(
        num_filters
    ):

        combined = np.concatenate(
            [
                D0[
                    ...,
                    filter_index
                ].ravel(),

                D1[
                    ...,
                    filter_index
                ].ravel(),
            ]
        )

        vmax = float(
            np.max(
                np.abs(
                    combined
                )
            )
        )

        if vmax <= 0.0:

            vmax = 1.0

        axes[
            0,
            filter_index
        ].imshow(
            D0[
                ...,
                filter_index
            ],
            cmap=
                "gray",
            vmin=
                -vmax,
            vmax=
                vmax,
        )

        axes[
            1,
            filter_index
        ].imshow(
            D1[
                ...,
                filter_index
            ],
            cmap=
                "gray",
            vmin=
                -vmax,
            vmax=
                vmax,
        )

        axes[
            0,
            filter_index
        ].set_title(
            f"d{filter_index + 1}",
            fontsize=9,
        )

        axes[
            0,
            filter_index
        ].axis(
            "off"
        )

        axes[
            1,
            filter_index
        ].axis(
            "off"
        )

    axes[
        0,
        0
    ].set_ylabel(
        "Initial D0",
        fontsize=10,
    )

    axes[
        1,
        0
    ].set_ylabel(
        "Learned SCSC",
        fontsize=10,
    )

    figure.suptitle(
        "STEP 06 — Initial vs Learned SCSC Dictionary",
        fontsize=15,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches=
            "tight",
    )

    plt.close(
        figure
    )
    
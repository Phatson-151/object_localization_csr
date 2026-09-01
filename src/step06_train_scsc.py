import argparse
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd

from src.config import (
    CLASS_NAMES,
    EXPECTED_HEIGHT,
    EXPECTED_WIDTH,
    MANIFEST_PATH,

    STEP03_OUTPUT,
    STEP04_OUTPUT,
    STEP06_OUTPUT,

    SCSC_SOURCE_STEP04_RUN,
    SCSC_NUM_TRAIN_SAMPLES,

    SCSC_OUTER_ITER,

    SCSC_Z_ADMM_ITER,
    SCSC_Z_ADMM_RHO,
    SCSC_Z_ADMM_TOL,

    SCSC_Z_CG_MAX_ITER,
    SCSC_Z_CG_TOL,

    SCSC_LOGISTIC_NEWTON_ITER,
    SCSC_LOGISTIC_NEWTON_TOL,

    SCSC_INIT_Z_MAX_ITER,
    SCSC_INIT_Z_REL_STOP_TOL,

    SCSC_D_ITER,
    SCSC_D_RHO,

    SCSC_ALPHA,
    SCSC_GAMMA_RATIO,

    SCSC_CLASSIFIER_MAX_ITER,
    SCSC_CLASSIFIER_GTOL,

    SCSC_RANDOM_SEED,

    SCSC_MIN_DICTIONARY_REL_CHANGE,
    SCSC_DEGENERATE_L1_TOL,

    SCSC_STAGE_OBJECTIVE_REL_TOL,
    SCSC_FINAL_OBJECTIVE_REL_TOL,
)

from src.csc.csc_utils import (
    build_training_stack,
    dictionary_statistics,
    save_dictionary_grid,
)

from src.csc.step05_utils import (
    infer_sparse_maps,
)

from src.scsc.scsc_utils import (
    build_image_supervision_records,
    evaluate_internal_classifier_diagnostics,
    evaluate_scsc_objective,
    fit_internal_classifiers,
    relative_dictionary_change,
    save_scsc_dictionary_comparison,
    save_scsc_training_curves,
    supervised_sparse_code_admm,
    update_dictionary_sporco,
)

from src.utils.output_utils import (
    create_output_dir,
    sample_filename,
)


STEP_NUMBER = 6


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "STEP 06: Full Supervised "
            "Convolutional Sparse Coding training."
        )
    )

    parser.add_argument(
        "--step04-run-name",
        type=str,
        default=
            SCSC_SOURCE_STEP04_RUN,
    )

    parser.add_argument(
        "--num-train-samples",
        type=int,
        default=
            SCSC_NUM_TRAIN_SAMPLES,
    )

    parser.add_argument(
        "--outer-iter",
        type=int,
        default=
            SCSC_OUTER_ITER,
    )

    parser.add_argument(
        "--z-iter",
        type=int,
        default=
            SCSC_Z_ADMM_ITER,
    )

    parser.add_argument(
        "--d-iter",
        type=int,
        default=
            SCSC_D_ITER,
    )

    parser.add_argument(
        "--gamma-ratio",
        type=float,
        default=
            SCSC_GAMMA_RATIO,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help=(
            "Optional fixed gamma. "
            "If omitted, gamma is scaled "
            "from --gamma-ratio using the "
            "initial training objective."
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=
            SCSC_ALPHA,
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=
            "scsc_reference_k16_fixed",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# SMALL HELPERS
# ============================================================

def get_supervision_path(
    sample_id,
):

    return (
        STEP03_OUTPUT
        / "data"
        / sample_filename(
            step=3,
            sample_id=sample_id,
            name="supervision",
            extension="npz",
        )
    )


def load_supervision_file(
    path,
):

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        return {
            key:
                np.array(
                    data[
                        key
                    ]
                )
            for key
            in data.files
        }


def copy_classifier_parameters(
    parameters,
):

    return {
        "weights":
            np.asarray(
                parameters[
                    "weights"
                ],
                dtype=np.float32,
            ).copy(),

        "biases":
            np.asarray(
                parameters[
                    "biases"
                ],
                dtype=np.float32,
            ).copy(),
    }


def stage_is_acceptable(
    previous_objective,
    candidate_objective,
):

    tolerance = (
        SCSC_STAGE_OBJECTIVE_REL_TOL
        *
        max(
            abs(
                float(
                    previous_objective
                )
            ),
            1.0,
        )
    )

    return (
        float(
            candidate_objective
        )
        <=
        float(
            previous_objective
        )
        +
        tolerance
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # ========================================================
    # ARGUMENT VALIDATION
    # ========================================================

    if args.num_train_samples <= 0:

        raise ValueError(
            "--num-train-samples must be > 0"
        )

    if args.outer_iter <= 0:

        raise ValueError(
            "--outer-iter must be > 0"
        )

    if args.z_iter <= 0:

        raise ValueError(
            "--z-iter must be > 0"
        )

    if args.d_iter <= 0:

        raise ValueError(
            "--d-iter must be > 0"
        )

    if (
        args.gamma is None
        and
        args.gamma_ratio <= 0.0
    ):

        raise ValueError(
            "--gamma-ratio must be > 0"
        )

    if (
        args.gamma is not None
        and
        args.gamma <= 0.0
    ):

        raise ValueError(
            "--gamma must be > 0"
        )

    if args.alpha < 0.0:

        raise ValueError(
            "--alpha must be >= 0"
        )

    np.random.seed(
        SCSC_RANDOM_SEED
    )

    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

    step_output = (
        create_output_dir(
            STEP06_OUTPUT
        )
    )

    run_output = (
        step_output
        /
        args.run_name
    )

    if run_output.exists():

        if args.overwrite:

            shutil.rmtree(
                run_output
            )

        elif any(
            run_output.iterdir()
        ):

            raise FileExistsError(
                f"STEP 06 run already exists:\n"
                f"{run_output}\n"
                f"Use --overwrite."
            )

    create_output_dir(
        run_output
    )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    config_path = (
        run_output
        /
        "step06_run_config.json"
    )

    training_samples_path = (
        run_output
        /
        "step06_training_samples.csv"
    )

    supervision_summary_path = (
        run_output
        /
        "step06_supervision_summary.csv"
    )

    initial_dictionary_path = (
        run_output
        /
        "step06_initial_dictionary.npy"
    )

    final_dictionary_path = (
        run_output
        /
        "step06_final_dictionary.npy"
    )

    classifier_path = (
        run_output
        /
        "step06_classifier_parameters.npz"
    )

    log_path = (
        run_output
        /
        "step06_training_log.csv"
    )

    initial_dictionary_png = (
        run_output
        /
        "step06_initial_dictionary.png"
    )

    final_dictionary_png = (
        run_output
        /
        "step06_final_dictionary.png"
    )

    comparison_png = (
        run_output
        /
        "step06_dictionary_comparison.png"
    )

    curves_png = (
        run_output
        /
        "step06_training_curves.png"
    )

    report_path = (
        run_output
        /
        "step06_training_report.txt"
    )

    # ========================================================
    # LOAD STEP 04 SOURCE
    # ========================================================

    step04_run = (
        STEP04_OUTPUT
        /
        args.step04_run_name
    )

    d0_path = (
        step04_run
        /
        "step04_initial_dictionary.npy"
    )

    lambda_path = (
        step04_run
        /
        "step04_lambda_diagnostics.json"
    )

    step04_config_path = (
        step04_run
        /
        "step04_run_config.json"
    )

    for required_path in (
        d0_path,
        lambda_path,
        step04_config_path,
    ):

        if not required_path.exists():

            raise FileNotFoundError(
                f"Missing required STEP 04 file:\n"
                f"{required_path}"
            )

    D0 = np.load(
        d0_path,
        allow_pickle=False,
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # SCSC starts from the SAME D0 as STEP 04 CSC.
    # It does NOT start from the learned CSC dictionary.
    # --------------------------------------------------------

    D = (
        D0.copy()
    )

    lambda_diagnostics = json.loads(
        lambda_path.read_text(
            encoding="utf-8"
        )
    )

    step04_config = json.loads(
        step04_config_path.read_text(
            encoding="utf-8"
        )
    )

    lmbda = float(
        lambda_diagnostics[
            "effective_lambda"
        ]
    )

    scale = float(
        step04_config[
            "scale"
        ]
    )

    initial_csc_rho = float(
        step04_config[
            "cbpdn_rho"
        ]
    )

    filter_size = int(
        D.shape[
            0
        ]
    )

    num_filters = int(
        D.shape[
            -1
        ]
    )

    # ========================================================
    # TRAIN MANIFEST — TRAIN ONLY
    # ========================================================

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    train_manifest = (
        manifest[
            manifest[
                "split"
            ]
            ==
            "train"
        ]
        .reset_index(
            drop=True
        )
    )

    if (
        args.num_train_samples
        >
        len(
            train_manifest
        )
    ):

        raise ValueError(
            f"Requested "
            f"{args.num_train_samples} "
            f"training images, but only "
            f"{len(train_manifest)} exist."
        )

    train_manifest = (
        train_manifest
        .iloc[
            :args.num_train_samples
        ]
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # HEADER
    # ========================================================

    print(
        "=" * 80
    )

    print(
        "STEP 06: SUPERVISED "
        "CONVOLUTIONAL SPARSE CODING"
    )

    print(
        "=" * 80
    )

    print(
        f"STEP 04 source      : "
        f"{args.step04_run_name}"
    )

    print(
        f"Training images     : "
        f"{len(train_manifest)}"
    )

    print(
        "Validation images   : 0 "
        "(NOT USED IN STEP 06)"
    )

    print(
        f"Dictionary          : "
        f"{D.shape}"
    )

    print(
        f"Scale               : "
        f"{scale}"
    )

    print(
        f"Lambda              : "
        f"{lmbda:.8e}"
    )

    print(
        f"Outer iterations    : "
        f"{args.outer_iter}"
    )

    print(
        f"Supervised Z steps  : "
        f"{args.z_iter}"
    )

    print(
        f"Z CG max iterations : "
        f"{SCSC_Z_CG_MAX_ITER}"
    )

    print(
        f"Dictionary steps    : "
        f"{args.d_iter}"
    )

    print(
        f"Alpha               : "
        f"{args.alpha}"
    )

    if args.gamma is None:

        print(
            f"Gamma mode          : "
            f"DATA-DRIVEN "
            f"(ratio={args.gamma_ratio})"
        )

    else:

        print(
            f"Gamma mode          : "
            f"FIXED "
            f"({args.gamma})"
        )

    print(
        "=" * 80
    )

    print()

    # ========================================================
    # TRAINING SIGNALS
    # ========================================================

    print(
        "Loading STEP 02 training signals..."
    )

    S, training_metadata = (
        build_training_stack(
            train_manifest=
                train_manifest,

            num_samples=
                len(
                    train_manifest
                ),

            scale=
                scale,
        )
    )

    training_metadata.to_csv(
        training_samples_path,
        index=False,
    )

    model_shape = (
        int(
            S.shape[
                0
            ]
        ),
        int(
            S.shape[
                1
            ]
        ),
    )

    print(
        f"Training tensor     : "
        f"{S.shape}"
    )

    print()

    # ========================================================
    # STEP 03 SUPERVISION
    # ========================================================

    print(
        "Loading STEP 03 supervision..."
    )

    supervision_list = []

    original_shapes = []

    sample_ids = []

    class_counts = {
        1: 0,
        2: 0,
        3: 0,
    }

    for _, row in (
        train_manifest.iterrows()
    ):

        sample_id = str(
            row[
                "sample_id"
            ]
        )

        sample_ids.append(
            sample_id
        )

        supervision_path = (
            get_supervision_path(
                sample_id
            )
        )

        if not supervision_path.exists():

            raise FileNotFoundError(
                f"Missing STEP 03 supervision:\n"
                f"{supervision_path}"
            )

        supervision = (
            load_supervision_file(
                supervision_path
            )
        )

        supervision_list.append(
            supervision
        )

        original_shape = (
            EXPECTED_HEIGHT,
            EXPECTED_WIDTH,
        )

        original_shapes.append(
            original_shape
        )

        for class_id in (
            1,
            2,
            3,
        ):

            class_name = (
                CLASS_NAMES[
                    class_id
                ]
            )

            class_counts[
                class_id
            ] += int(
                len(
                    supervision[
                        f"{class_name}_labels"
                    ]
                )
            )

    mean_class_count = float(
        np.mean(
            list(
                class_counts.values()
            )
        )
    )

    # --------------------------------------------------------
    # MULTICLASS BALANCING
    #
    # Each One-vs-All class contributes approximately
    # the same total supervision weight.
    # --------------------------------------------------------

    class_factors = {
        class_id:
            (
                mean_class_count
                /
                class_counts[
                    class_id
                ]
            )
        for class_id
        in (
            1,
            2,
            3,
        )
    }

    supervision_rows = []

    for class_id in (
        1,
        2,
        3,
    ):

        supervision_rows.append(
            {
                "class_id":
                    class_id,

                "class_name":
                    CLASS_NAMES[
                        class_id
                    ],

                "supervision_points":
                    class_counts[
                        class_id
                    ],

                "class_balance_factor":
                    class_factors[
                        class_id
                    ],
            }
        )

        print(
            f"{CLASS_NAMES[class_id].title():<12} "
            f"| points="
            f"{class_counts[class_id]:>6} "
            f"| balance factor="
            f"{class_factors[class_id]:.6f}"
        )

    pd.DataFrame(
        supervision_rows
    ).to_csv(
        supervision_summary_path,
        index=False,
    )

    print()

    # ========================================================
    # BUILD FIXED SUPERVISION RECORDS
    # ========================================================

    supervision_records = []

    for (
        supervision,
        original_shape,
    ) in zip(
        supervision_list,
        original_shapes,
    ):

        records = (
            build_image_supervision_records(
                supervision=
                    supervision,

                class_names=
                    CLASS_NAMES,

                class_factors=
                    class_factors,

                original_shape=
                    original_shape,

                model_shape=
                    model_shape,
            )
        )

        supervision_records.append(
            records
        )

    # ========================================================
    # SAVE INITIAL DICTIONARY
    # ========================================================

    np.save(
        initial_dictionary_path,
        D0,
        allow_pickle=False,
    )

    save_dictionary_grid(
        D0,
        initial_dictionary_png,
        title=(
            "STEP 06 — "
            "Initial SCSC Dictionary"
        ),
    )

    # ========================================================
    # INITIALISE Z WITH ORDINARY CSC
    # ========================================================

    print(
        "=" * 80
    )

    print(
        "INITIALISING SCSC SPARSE MAPS"
    )

    print(
        "=" * 80
    )

    sparse_maps_list = []

    for (
        image_index,
        sample_id,
    ) in enumerate(
        sample_ids
    ):

        sparse_maps, diagnostics = (
            infer_sparse_maps(
                dictionary=
                    D,

                signal=
                    S[
                        ...,
                        image_index
                    ],

                lmbda=
                    lmbda,

                rho=
                    initial_csc_rho,

                max_iter=
                    SCSC_INIT_Z_MAX_ITER,

                rel_stop_tol=
                    SCSC_INIT_Z_REL_STOP_TOL,
            )
        )

        sparse_maps_list.append(
            sparse_maps
        )

        print(
            f"[{image_index + 1:02d}/"
            f"{len(sample_ids):02d}] "
            f"{sample_id} | "
            f"L1="
            f"{np.sum(np.abs(sparse_maps)):.3e} "
            f"| PASS"
        )

    print()

    # ========================================================
    # INITIAL THETA
    # ========================================================

    print(
        "Initialising internal "
        "One-vs-All classifiers..."
    )

    (
        classifier_parameters,
        classifier_diagnostics,
    ) = fit_internal_classifiers(
        sparse_maps_list=
            sparse_maps_list,

        supervision_list=
            supervision_list,

        original_shapes=
            original_shapes,

        class_names=
            CLASS_NAMES,

        class_factors=
            class_factors,

        alpha=
            args.alpha,

        max_iter=
            SCSC_CLASSIFIER_MAX_ITER,

        gradient_tolerance=
            SCSC_CLASSIFIER_GTOL,

        initial_parameters=
            None,
    )

    # ========================================================
    # DATA-DRIVEN GAMMA
    # ========================================================

    objective_gamma_one = (
        evaluate_scsc_objective(
            dictionary=
                D,

            training_signals=
                S,

            sparse_maps_list=
                sparse_maps_list,

            lmbda=
                lmbda,

            supervision_list=
                supervision_list,

            original_shapes=
                original_shapes,

            class_names=
                CLASS_NAMES,

            class_factors=
                class_factors,

            classifier_parameters=
                classifier_parameters,

            gamma=
                1.0,

            alpha=
                args.alpha,
        )
    )

    if args.gamma is None:

        gamma = (
            args.gamma_ratio
            *
            objective_gamma_one[
                "base_objective"
            ]
            /
            max(
                objective_gamma_one[
                    "supervised_term"
                ],
                1e-12,
            )
        )

        gamma_source = (
            "data_driven_ratio"
        )

    else:

        gamma = float(
            args.gamma
        )

        gamma_source = (
            "fixed"
        )

    objective = (
        evaluate_scsc_objective(
            dictionary=
                D,

            training_signals=
                S,

            sparse_maps_list=
                sparse_maps_list,

            lmbda=
                lmbda,

            supervision_list=
                supervision_list,

            original_shapes=
                original_shapes,

            class_names=
                CLASS_NAMES,

            class_factors=
                class_factors,

            classifier_parameters=
                classifier_parameters,

            gamma=
                gamma,

            alpha=
                args.alpha,
        )
    )

    initial_objective = float(
        objective[
            "total_objective"
        ]
    )

    print()

    print(
        f"Initial CSC objective : "
        f"{objective['base_objective']:.8e}"
    )

    print(
        f"Initial supervised    : "
        f"{objective['supervised_term']:.8e}"
    )

    print(
        f"Effective gamma       : "
        f"{gamma:.8e}"
    )

    print(
        f"gamma * supervision  : "
        f"{gamma * objective['supervised_term']:.8e}"
    )

    print(
        f"Initial total object. : "
        f"{initial_objective:.8e}"
    )

    print()

    # ========================================================
    # SAVE RUN CONFIG
    # ========================================================

    run_config = {
        "step":
            STEP_NUMBER,

        "source_step04_run":
            args.step04_run_name,

        "training_images":
            len(
                train_manifest
            ),

        "validation_images_used":
            0,

        "dictionary_shape":
            list(
                D.shape
            ),

        "num_filters":
            num_filters,

        "filter_size":
            filter_size,

        "scale":
            scale,

        "lambda":
            lmbda,

        "outer_iterations":
            args.outer_iter,

        "z_admm_iterations":
            args.z_iter,

        "z_admm_rho":
            SCSC_Z_ADMM_RHO,

        "z_admm_tolerance":
            SCSC_Z_ADMM_TOL,

        "z_cg_max_iterations":
            SCSC_Z_CG_MAX_ITER,

        "z_cg_tolerance":
            SCSC_Z_CG_TOL,

        "logistic_newton_iterations":
            SCSC_LOGISTIC_NEWTON_ITER,

        "dictionary_iterations":
            args.d_iter,

        "dictionary_rho":
            SCSC_D_RHO,

        "alpha":
            args.alpha,

        "gamma_mode":
            gamma_source,

        "gamma_ratio":
            args.gamma_ratio,

        "effective_gamma":
            gamma,

        "multiclass_strategy":
            "One-vs-All",

        "class_balancing": {
            str(
                class_id
            ):
                class_factors[
                    class_id
                ]
            for class_id
            in (
                1,
                2,
                3,
            )
        },

        "initial_dictionary":
            (
                "Same STEP 04 D0 used "
                "for unsupervised CSC"
            ),

        "z_solver":
            (
                "Supervised ADMM + CG + "
                "Newton logistic prox + "
                "best-primal-objective safeguard"
            ),

        "dictionary_solver":
            (
                "SPORCO Consensus ADMM "
                "ConvCnstrMOD"
            ),

        "stage_objective_relative_tolerance":
            SCSC_STAGE_OBJECTIVE_REL_TOL,

        "final_objective_relative_tolerance":
            SCSC_FINAL_OBJECTIVE_REL_TOL,

        "random_seed":
            SCSC_RANDOM_SEED,
    }

    config_path.write_text(
        json.dumps(
            run_config,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # INITIAL LOG ROW
    # ========================================================

    training_rows = []

    training_rows.append(
        {
            "outer_iteration":
                0,

            **objective,

            "objective_before_z":
                initial_objective,

            "objective_after_z":
                initial_objective,

            "objective_after_d":
                initial_objective,

            "objective_after_theta":
                initial_objective,

            "z_update_accepted":
                True,

            "d_update_accepted":
                True,

            "theta_update_accepted":
                True,

            "z_images_improved":
                0,

            "cg_nonconverged_steps":
                0,

            "dictionary_change":
                0.0,

            "person_ap":
                classifier_diagnostics[
                    1
                ][
                    "ap"
                ],

            "car_ap":
                classifier_diagnostics[
                    2
                ][
                    "ap"
                ],

            "motorcycle_ap":
                classifier_diagnostics[
                    3
                ][
                    "ap"
                ],

            "mean_z_primal":
                np.nan,

            "mean_z_dual":
                np.nan,

            "dictionary_constraint":
                np.nan,

            "dictionary_primal":
                np.nan,

            "dictionary_dual":
                np.nan,

            "outer_time_seconds":
                0.0,
        }
    )

    # ========================================================
    # SCSC COORDINATE DESCENT
    # ========================================================

    print(
        "=" * 80
    )

    print(
        "FULL SCSC COORDINATE DESCENT"
    )

    print(
        "=" * 80
    )

    total_start_time = (
        time.perf_counter()
    )

    total_cg_nonconverged_all = 0

    z_rollback_count = 0

    d_rollback_count = 0

    theta_rollback_count = 0

    for outer_iteration in range(
        1,
        args.outer_iter
        +
        1,
    ):

        outer_start_time = (
            time.perf_counter()
        )

        print()

        print(
            "=" * 80
        )

        print(
            f"SCSC OUTER ITERATION "
            f"{outer_iteration}/"
            f"{args.outer_iter}"
        )

        print(
            "=" * 80
        )

        objective_before_z_state = (
            evaluate_scsc_objective(
                dictionary=
                    D,

                training_signals=
                    S,

                sparse_maps_list=
                    sparse_maps_list,

                lmbda=
                    lmbda,

                supervision_list=
                    supervision_list,

                original_shapes=
                    original_shapes,

                class_names=
                    CLASS_NAMES,

                class_factors=
                    class_factors,

                classifier_parameters=
                    classifier_parameters,

                gamma=
                    gamma,

                alpha=
                    args.alpha,
            )
        )

        objective_before_z = float(
            objective_before_z_state[
                "total_objective"
            ]
        )

        # ====================================================
        # 1/3 UPDATE Z — SUPERVISED EQ. 6
        # ====================================================

        print(
            "1/3 UPDATE Z "
            "(Supervised ADMM + objective safeguard)"
        )

        old_sparse_maps_list = [
            sparse_maps.copy()
            for sparse_maps
            in sparse_maps_list
        ]

        updated_sparse_maps = []

        z_diagnostics = []

        for (
            image_index,
            sample_id,
        ) in enumerate(
            sample_ids
        ):

            (
                sparse_maps,
                diagnostics,
            ) = supervised_sparse_code_admm(
                dictionary=
                    D,

                signal=
                    S[
                        ...,
                        image_index
                    ],

                initial_sparse_maps=
                    sparse_maps_list[
                        image_index
                    ],

                records=
                    supervision_records[
                        image_index
                    ],

                classifier_parameters=
                    classifier_parameters,

                lmbda=
                    lmbda,

                gamma=
                    gamma,

                rho=
                    SCSC_Z_ADMM_RHO,

                admm_iterations=
                    args.z_iter,

                admm_tolerance=
                    SCSC_Z_ADMM_TOL,

                cg_max_iter=
                    SCSC_Z_CG_MAX_ITER,

                cg_tolerance=
                    SCSC_Z_CG_TOL,

                newton_iterations=
                    SCSC_LOGISTIC_NEWTON_ITER,

                newton_tolerance=
                    SCSC_LOGISTIC_NEWTON_TOL,
            )

            updated_sparse_maps.append(
                sparse_maps
            )

            z_diagnostics.append(
                diagnostics
            )

            print(
                f"  [{image_index + 1:02d}/"
                f"{len(sample_ids):02d}] "
                f"{sample_id} | "
                f"r="
                f"{diagnostics['primal_residual']:.3e} "
                f"| s="
                f"{diagnostics['dual_residual']:.3e} "
                f"| L1="
                f"{diagnostics['l1_norm']:.3e} "
                f"| Obj "
                f"{diagnostics['initial_subproblem_objective']:.3e}"
                f" -> "
                f"{diagnostics['best_subproblem_objective']:.3e} "
                f"| best="
                f"{diagnostics['best_iteration']} "
                f"| CGfail="
                f"{diagnostics['cg_nonconverged_steps']}"
            )

        sparse_maps_list = (
            updated_sparse_maps
        )

        objective_after_z_state = (
            evaluate_scsc_objective(
                dictionary=
                    D,

                training_signals=
                    S,

                sparse_maps_list=
                    sparse_maps_list,

                lmbda=
                    lmbda,

                supervision_list=
                    supervision_list,

                original_shapes=
                    original_shapes,

                class_names=
                    CLASS_NAMES,

                class_factors=
                    class_factors,

                classifier_parameters=
                    classifier_parameters,

                gamma=
                    gamma,

                alpha=
                    args.alpha,
            )
        )

        objective_after_z_candidate = float(
            objective_after_z_state[
                "total_objective"
            ]
        )

        z_update_accepted = (
            stage_is_acceptable(
                objective_before_z,
                objective_after_z_candidate,
            )
        )

        if not z_update_accepted:

            print()

            print(
                "  WARNING: Z update increased "
                "the full SCSC objective. "
                "Rolling Z back."
            )

            sparse_maps_list = (
                old_sparse_maps_list
            )

            objective_after_z_state = (
                objective_before_z_state
            )

            z_rollback_count += 1

        objective_after_z = float(
            objective_after_z_state[
                "total_objective"
            ]
        )

        improved_z_images = int(
            sum(
                item[
                    "objective_improvement"
                ]
                >
                0.0
                for item
                in z_diagnostics
            )
        )

        total_cg_nonconverged = int(
            sum(
                item[
                    "cg_nonconverged_steps"
                ]
                for item
                in z_diagnostics
            )
        )

        total_cg_nonconverged_all += (
            total_cg_nonconverged
        )

        mean_z_primal = float(
            np.mean(
                [
                    item[
                        "primal_residual"
                    ]
                    for item
                    in z_diagnostics
                ]
            )
        )

        mean_z_dual = float(
            np.mean(
                [
                    item[
                        "dual_residual"
                    ]
                    for item
                    in z_diagnostics
                ]
            )
        )

        print()

        print(
            f"  Z full objective     : "
            f"{objective_before_z:.6e} "
            f"-> "
            f"{objective_after_z:.6e}"
        )

        print(
            f"  Z objective improved : "
            f"{improved_z_images}/"
            f"{len(z_diagnostics)} images"
        )

        print(
            f"  CG non-converged     : "
            f"{total_cg_nonconverged} steps"
        )

        print(
            f"  Z update accepted    : "
            f"{z_update_accepted}"
        )

        # ====================================================
        # 2/3 UPDATE D — EQ. 2
        # ====================================================

        print()

        print(
            "2/3 UPDATE D "
            "(SPORCO Consensus ADMM + safeguard)"
        )

        D_before_update = (
            D.copy()
        )

        (
            D_candidate,
            dictionary_diagnostics,
        ) = update_dictionary_sporco(
            sparse_maps_list=
                sparse_maps_list,

            training_signals=
                S,

            current_dictionary=
                D,

            filter_size=
                filter_size,

            num_filters=
                num_filters,

            max_iter=
                args.d_iter,

            rho=
                SCSC_D_RHO,
        )

        objective_after_d_candidate_state = (
            evaluate_scsc_objective(
                dictionary=
                    D_candidate,

                training_signals=
                    S,

                sparse_maps_list=
                    sparse_maps_list,

                lmbda=
                    lmbda,

                supervision_list=
                    supervision_list,

                original_shapes=
                    original_shapes,

                class_names=
                    CLASS_NAMES,

                class_factors=
                    class_factors,

                classifier_parameters=
                    classifier_parameters,

                gamma=
                    gamma,

                alpha=
                    args.alpha,
            )
        )

        objective_after_d_candidate = float(
            objective_after_d_candidate_state[
                "total_objective"
            ]
        )

        d_update_accepted = (
            stage_is_acceptable(
                objective_after_z,
                objective_after_d_candidate,
            )
        )

        if d_update_accepted:

            D = (
                D_candidate
            )

            objective_after_d_state = (
                objective_after_d_candidate_state
            )

        else:

            print(
                "  WARNING: Dictionary update "
                "increased the full SCSC objective. "
                "Rolling D back."
            )

            D = (
                D_before_update
            )

            objective_after_d_state = (
                objective_after_z_state
            )

            d_rollback_count += 1

        objective_after_d = float(
            objective_after_d_state[
                "total_objective"
            ]
        )

        print(
            f"  SPORCO DFid="
            f"{dictionary_diagnostics['data_fidelity']:.4e} "
            f"| Constraint="
            f"{dictionary_diagnostics['constraint']:.4e}"
        )

        print(
            f"  D full objective     : "
            f"{objective_after_z:.6e} "
            f"-> "
            f"{objective_after_d:.6e}"
        )

        print(
            f"  D update accepted    : "
            f"{d_update_accepted}"
        )

        # ====================================================
        # 3/3 UPDATE THETA — EQ. 7
        # ====================================================

        print()

        print(
            "3/3 UPDATE THETA "
            "(One-vs-All Logistic + safeguard)"
        )

        theta_before_update = (
            copy_classifier_parameters(
                classifier_parameters
            )
        )

        (
            theta_candidate,
            theta_candidate_diagnostics,
        ) = fit_internal_classifiers(
            sparse_maps_list=
                sparse_maps_list,

            supervision_list=
                supervision_list,

            original_shapes=
                original_shapes,

            class_names=
                CLASS_NAMES,

            class_factors=
                class_factors,

            alpha=
                args.alpha,

            max_iter=
                SCSC_CLASSIFIER_MAX_ITER,

            gradient_tolerance=
                SCSC_CLASSIFIER_GTOL,

            initial_parameters=
                classifier_parameters,
        )

        objective_after_theta_candidate_state = (
            evaluate_scsc_objective(
                dictionary=
                    D,

                training_signals=
                    S,

                sparse_maps_list=
                    sparse_maps_list,

                lmbda=
                    lmbda,

                supervision_list=
                    supervision_list,

                original_shapes=
                    original_shapes,

                class_names=
                    CLASS_NAMES,

                class_factors=
                    class_factors,

                classifier_parameters=
                    theta_candidate,

                gamma=
                    gamma,

                alpha=
                    args.alpha,
            )
        )

        objective_after_theta_candidate = float(
            objective_after_theta_candidate_state[
                "total_objective"
            ]
        )

        theta_update_accepted = (
            stage_is_acceptable(
                objective_after_d,
                objective_after_theta_candidate,
            )
        )

        if theta_update_accepted:

            classifier_parameters = (
                theta_candidate
            )

            classifier_diagnostics = (
                theta_candidate_diagnostics
            )

            objective_after_theta_state = (
                objective_after_theta_candidate_state
            )

        else:

            print(
                "  WARNING: Theta update increased "
                "the full SCSC objective. "
                "Rolling theta back."
            )

            classifier_parameters = (
                theta_before_update
            )

            classifier_diagnostics = (
                evaluate_internal_classifier_diagnostics(
                    sparse_maps_list=
                        sparse_maps_list,

                    supervision_list=
                        supervision_list,

                    original_shapes=
                        original_shapes,

                    class_names=
                        CLASS_NAMES,

                    class_factors=
                        class_factors,

                    classifier_parameters=
                        classifier_parameters,
                )
            )

            objective_after_theta_state = (
                objective_after_d_state
            )

            theta_rollback_count += 1

        objective_after_theta = float(
            objective_after_theta_state[
                "total_objective"
            ]
        )

        for class_id in (
            1,
            2,
            3,
        ):

            diagnostics = (
                classifier_diagnostics[
                    class_id
                ]
            )

            print(
                f"  "
                f"{diagnostics['class_name'].title():<12} "
                f"| AP="
                f"{diagnostics['ap']:.4f} "
                f"| logistic="
                f"{diagnostics['weighted_logistic_loss']:.4e}"
            )

        print(
            f"  Theta full objective : "
            f"{objective_after_d:.6e} "
            f"-> "
            f"{objective_after_theta:.6e}"
        )

        print(
            f"  Theta accepted       : "
            f"{theta_update_accepted}"
        )

        # ====================================================
        # OUTER SUMMARY
        # ====================================================

        objective = (
            objective_after_theta_state
        )

        dictionary_change = (
            relative_dictionary_change(
                D0,
                D,
            )
        )

        outer_time = (
            time.perf_counter()
            -
            outer_start_time
        )

        row = {
            "outer_iteration":
                outer_iteration,

            **objective,

            "objective_before_z":
                objective_before_z,

            "objective_after_z":
                objective_after_z,

            "objective_after_d":
                objective_after_d,

            "objective_after_theta":
                objective_after_theta,

            "z_update_accepted":
                bool(
                    z_update_accepted
                ),

            "d_update_accepted":
                bool(
                    d_update_accepted
                ),

            "theta_update_accepted":
                bool(
                    theta_update_accepted
                ),

            "z_images_improved":
                improved_z_images,

            "cg_nonconverged_steps":
                total_cg_nonconverged,

            "dictionary_change":
                dictionary_change,

            "person_ap":
                classifier_diagnostics[
                    1
                ][
                    "ap"
                ],

            "car_ap":
                classifier_diagnostics[
                    2
                ][
                    "ap"
                ],

            "motorcycle_ap":
                classifier_diagnostics[
                    3
                ][
                    "ap"
                ],

            "mean_z_primal":
                mean_z_primal,

            "mean_z_dual":
                mean_z_dual,

            "dictionary_constraint":
                dictionary_diagnostics[
                    "constraint"
                ],

            "dictionary_primal":
                dictionary_diagnostics[
                    "primal_residual"
                ],

            "dictionary_dual":
                dictionary_diagnostics[
                    "dual_residual"
                ],

            "outer_time_seconds":
                outer_time,
        }

        training_rows.append(
            row
        )

        pd.DataFrame(
            training_rows
        ).to_csv(
            log_path,
            index=False,
        )

        print()

        print(
            f"Outer "
            f"{outer_iteration} "
            f"summary"
        )

        print(
            f"  Total objective     : "
            f"{objective['total_objective']:.6e}"
        )

        print(
            f"  Reconstruction      : "
            f"{objective['data_fidelity']:.6e}"
        )

        print(
            f"  Sparsity penalty    : "
            f"{objective['sparsity_penalty']:.6e}"
        )

        print(
            f"  Classification loss : "
            f"{objective['classification_loss']:.6e}"
        )

        print(
            f"  Dictionary change   : "
            f"{dictionary_change:.6e}"
        )

        print(
            f"  Mean Z residual r/s : "
            f"{mean_z_primal:.3e} / "
            f"{mean_z_dual:.3e}"
        )

        print(
            f"  Time                : "
            f"{outer_time:.2f} s"
        )

    total_time = (
        time.perf_counter()
        -
        total_start_time
    )

    # ========================================================
    # FINAL OBJECTIVE
    # ========================================================

    final_objective_state = (
        evaluate_scsc_objective(
            dictionary=
                D,

            training_signals=
                S,

            sparse_maps_list=
                sparse_maps_list,

            lmbda=
                lmbda,

            supervision_list=
                supervision_list,

            original_shapes=
                original_shapes,

            class_names=
                CLASS_NAMES,

            class_factors=
                class_factors,

            classifier_parameters=
                classifier_parameters,

            gamma=
                gamma,

            alpha=
                args.alpha,
        )
    )

    final_objective = float(
        final_objective_state[
            "total_objective"
        ]
    )

    final_l1 = float(
        final_objective_state[
            "l1_norm"
        ]
    )

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    np.save(
        final_dictionary_path,
        D.astype(
            np.float32
        ),
        allow_pickle=False,
    )

    np.savez_compressed(
        classifier_path,

        weights=
            classifier_parameters[
                "weights"
            ],

        biases=
            classifier_parameters[
                "biases"
            ],

        class_ids=
            np.asarray(
                [
                    1,
                    2,
                    3,
                ],
                dtype=np.int8,
            ),

        gamma=
            np.float64(
                gamma
            ),

        alpha=
            np.float64(
                args.alpha
            ),
    )

    training_df = (
        pd.DataFrame(
            training_rows
        )
    )

    training_df.to_csv(
        log_path,
        index=False,
    )

    # ========================================================
    # VISUALISATIONS
    # ========================================================

    save_dictionary_grid(
        D,
        final_dictionary_png,
        title=(
            "STEP 06 — "
            "Learned SCSC Dictionary"
        ),
    )

    save_scsc_dictionary_comparison(
        D0,
        D,
        comparison_png,
    )

    save_scsc_training_curves(
        training_df,
        curves_png,
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    errors = []

    warnings = []

    dictionary_stats = (
        dictionary_statistics(
            D
        )
    )

    final_dictionary_change = (
        relative_dictionary_change(
            D0,
            D,
        )
    )

    if not dictionary_stats[
        "finite"
    ]:

        errors.append(
            "Final SCSC dictionary "
            "contains NaN or Inf."
        )

    if (
        final_l1
        <=
        SCSC_DEGENERATE_L1_TOL
    ):

        errors.append(
            "Final SCSC sparse maps "
            "collapsed to all-zero."
        )

    if (
        final_dictionary_change
        <=
        SCSC_MIN_DICTIONARY_REL_CHANGE
    ):

        errors.append(
            "SCSC dictionary did not "
            "change meaningfully from D0."
        )

    if not np.isfinite(
        final_objective
    ):

        errors.append(
            "Final SCSC objective "
            "is not finite."
        )

    maximum_allowed_final_objective = (
        initial_objective
        +
        SCSC_FINAL_OBJECTIVE_REL_TOL
        *
        max(
            abs(
                initial_objective
            ),
            1.0,
        )
    )

    if (
        final_objective
        >
        maximum_allowed_final_objective
    ):

        errors.append(
            "Final SCSC objective is higher "
            "than the initial objective beyond "
            "the allowed numerical tolerance. "
            "Coordinate-descent optimisation "
            "was not accepted."
        )

    for class_id in (
        1,
        2,
        3,
    ):

        ap = (
            classifier_diagnostics[
                class_id
            ][
                "ap"
            ]
        )

        if not np.isfinite(
            ap
        ):

            errors.append(
                f"{CLASS_NAMES[class_id]} "
                f"internal AP is invalid."
            )

    if (
        total_cg_nonconverged_all
        >
        0
    ):

        warnings.append(
            f"CG did not meet its requested "
            f"tolerance in "
            f"{total_cg_nonconverged_all} "
            f"inner solves. "
            f"Best-objective safeguards "
            f"were still applied."
        )

    if (
        z_rollback_count
        >
        0
    ):

        warnings.append(
            f"Z update was rolled back in "
            f"{z_rollback_count} "
            f"outer iteration(s)."
        )

    if (
        d_rollback_count
        >
        0
    ):

        warnings.append(
            f"Dictionary update was "
            f"rolled back in "
            f"{d_rollback_count} "
            f"outer iteration(s)."
        )

    if (
        theta_rollback_count
        >
        0
    ):

        warnings.append(
            f"Theta update was rolled back "
            f"in "
            f"{theta_rollback_count} "
            f"outer iteration(s)."
        )

    failed_classifier_optimisations = [
        diagnostics[
            "class_name"
        ]
        for diagnostics
        in classifier_diagnostics.values()
        if not diagnostics.get(
            "optimizer_success",
            True,
        )
    ]

    if failed_classifier_optimisations:

        warnings.append(
            "Internal classifier optimiser "
            "did not report success for: "
            +
            ", ".join(
                failed_classifier_optimisations
            )
        )

    final_status = (
        "PASSED"
        if len(
            errors
        ) == 0
        else
        "FAILED"
    )

    # ========================================================
    # REPORT
    # ========================================================

    report_lines = [
        "STEP 06: SUPERVISED CONVOLUTIONAL SPARSE CODING",
        "=" * 80,
        "",
        "EXPERIMENT",
        "-" * 80,
        f"Source STEP 04 run       : {args.step04_run_name}",
        f"Training images          : {len(train_manifest)}",
        "Validation images used   : 0",
        f"Dictionary shape         : {D.shape}",
        f"Scale                    : {scale}",
        f"Lambda                   : {lmbda:.8e}",
        f"Alpha                    : {args.alpha:.8e}",
        f"Effective gamma          : {gamma:.8e}",
        f"Gamma mode               : {gamma_source}",
        f"Gamma ratio              : {args.gamma_ratio}",
        "",
        "OPTIMISATION",
        "-" * 80,
        f"Outer iterations         : {args.outer_iter}",
        f"Z ADMM iterations        : {args.z_iter}",
        f"Z ADMM rho               : {SCSC_Z_ADMM_RHO}",
        f"Z ADMM tolerance         : {SCSC_Z_ADMM_TOL}",
        f"Z CG max iterations      : {SCSC_Z_CG_MAX_ITER}",
        f"Z CG tolerance           : {SCSC_Z_CG_TOL}",
        f"Dictionary iterations    : {args.d_iter}",
        f"Dictionary rho           : {SCSC_D_RHO}",
        (
            "Z safeguard              : "
            "best exact Eq. 6 primal objective"
        ),
        (
            "Stage safeguard          : "
            "rollback if full objective increases"
        ),
        (
            "Dictionary method        : "
            "SPORCO Consensus ADMM"
        ),
        (
            "Classifier method        : "
            "One-vs-All Logistic Regression"
        ),
        "",
        "INITIAL RESULT",
        "-" * 80,
        f"Initial total objective  : {initial_objective:.8e}",
        (
            f"Initial Person AP        : "
            f"{training_df['person_ap'].iloc[0]:.6f}"
        ),
        (
            f"Initial Car AP           : "
            f"{training_df['car_ap'].iloc[0]:.6f}"
        ),
        (
            f"Initial Motorcycle AP    : "
            f"{training_df['motorcycle_ap'].iloc[0]:.6f}"
        ),
        "",
        "FINAL RESULT",
        "-" * 80,
        f"Final total objective    : {final_objective:.8e}",
        (
            f"Final data fidelity      : "
            f"{final_objective_state['data_fidelity']:.8e}"
        ),
        f"Final L1 norm            : {final_l1:.8e}",
        (
            f"Final sparsity penalty   : "
            f"{final_objective_state['sparsity_penalty']:.8e}"
        ),
        (
            f"Final classification loss: "
            f"{final_objective_state['classification_loss']:.8e}"
        ),
        (
            f"Final Person AP          : "
            f"{classifier_diagnostics[1]['ap']:.6f}"
        ),
        (
            f"Final Car AP             : "
            f"{classifier_diagnostics[2]['ap']:.6f}"
        ),
        (
            f"Final Motorcycle AP      : "
            f"{classifier_diagnostics[3]['ap']:.6f}"
        ),
        (
            f"Dictionary rel. change   : "
            f"{final_dictionary_change:.8e}"
        ),
        (
            f"Filter norm min          : "
            f"{dictionary_stats['norm_min']:.8f}"
        ),
        (
            f"Filter norm max          : "
            f"{dictionary_stats['norm_max']:.8f}"
        ),
        f"Training time (seconds)  : {total_time:.3f}",
        (
            f"CG non-converged total   : "
            f"{total_cg_nonconverged_all}"
        ),
        f"Z rollback count         : {z_rollback_count}",
        f"D rollback count         : {d_rollback_count}",
        f"Theta rollback count     : {theta_rollback_count}",
        "",
        "IMPORTANT",
        "-" * 80,
        (
            "The Person/Car/Motorcycle AP values above are INTERNAL "
            "training-supervision diagnostics used during SCSC learning."
        ),
        (
            "They are NOT final localization AP values."
        ),
        (
            "The classifier parameters learned inside STEP 06 will NOT be "
            "used directly for the final CSC-vs-SCSC evaluation."
        ),
        (
            "A fresh logistic regression classifier will later be retrained "
            "on sparse maps inferred using the learned SCSC dictionary."
        ),
        "",
        "VALIDATION",
        "-" * 80,
        f"Errors                   : {len(errors)}",
        f"Warnings                 : {len(warnings)}",
        f"FINAL STATUS             : {final_status}",
    ]

    if errors:

        report_lines.extend(
            [
                "",
                "ERRORS",
                "-" * 80,
            ]
        )

        for error in errors:

            report_lines.append(
                f"- {error}"
            )

    if warnings:

        report_lines.extend(
            [
                "",
                "WARNINGS",
                "-" * 80,
            ]
        )

        for warning in warnings:

            report_lines.append(
                f"- {warning}"
            )

    report_path.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    # ========================================================
    # TERMINAL RESULT
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "STEP 06 RESULT"
    )

    print(
        "=" * 80
    )

    print(
        f"Training time       : "
        f"{total_time:.2f} s"
    )

    print(
        f"Effective gamma     : "
        f"{gamma:.8e}"
    )

    print(
        f"Initial objective   : "
        f"{initial_objective:.6e}"
    )

    print(
        f"Final objective     : "
        f"{final_objective:.6e}"
    )

    print(
        f"Final L1            : "
        f"{final_l1:.6e}"
    )

    print(
        f"Dictionary change   : "
        f"{final_dictionary_change:.6e}"
    )

    print()

    print(
        f"Person AP           : "
        f"{classifier_diagnostics[1]['ap']:.4f}"
    )

    print(
        f"Car AP              : "
        f"{classifier_diagnostics[2]['ap']:.4f}"
    )

    print(
        f"Motorcycle AP       : "
        f"{classifier_diagnostics[3]['ap']:.4f}"
    )

    print()

    print(
        f"Dictionary shape    : "
        f"{D.shape}"
    )

    print(
        f"Filter norm         : "
        f"{dictionary_stats['norm_min']:.6f}"
        f" .. "
        f"{dictionary_stats['norm_max']:.6f}"
    )

    print(
        f"CG non-converged    : "
        f"{total_cg_nonconverged_all}"
    )

    print(
        f"Stage rollbacks     : "
        f"Z={z_rollback_count}, "
        f"D={d_rollback_count}, "
        f"Theta={theta_rollback_count}"
    )

    print(
        f"Errors              : "
        f"{len(errors)}"
    )

    print(
        f"Warnings            : "
        f"{len(warnings)}"
    )

    print(
        f"FINAL STATUS        : "
        f"{final_status}"
    )

    print()

    print(
        "Outputs saved to:"
    )

    print(
        run_output
    )

    print(
        "=" * 80
    )

    if final_status == "FAILED":

        sys.exit(
            1
        )


if __name__ == "__main__":

    main()
    
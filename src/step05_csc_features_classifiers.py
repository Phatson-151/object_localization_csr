import argparse
import json
import shutil
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import (
    GroupKFold,
)

from src.config import (
    CLASS_NAMES,
    MANIFEST_PATH,

    STEP03_OUTPUT,
    STEP04_OUTPUT,
    STEP05_OUTPUT,

    CSC_BASELINE_RUN_NAME,

    CSC_INFERENCE_MAX_ITER,
    CSC_INFERENCE_REL_STOP_TOL,
    CSC_SPARSE_NONZERO_TOL,

    CSC_CLASSIFIER_C,
    CSC_CLASSIFIER_MAX_ITER,
    CSC_CLASSIFIER_CV_FOLDS,
)

from src.data.dataset_utils import (
    load_sample,
)

from src.csc.csc_utils import (
    load_highpass,
    resize_highpass,
)

from src.csc.step05_utils import (
    infer_sparse_maps,
    sample_sparse_features_bilinear,
    build_logistic_classifier,
    classifier_score_map,
    save_sparse_map_grid,
    save_class_score_visualization,
)

from src.utils.output_utils import (
    create_output_dir,
    sample_filename,
)


STEP_NUMBER = 5


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "STEP 05: Infer CSC sparse maps "
            "and train One-vs-All classifiers."
        )
    )

    parser.add_argument(
        "--step04-run-name",
        type=str,
        default=
            CSC_BASELINE_RUN_NAME,
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=
            "csc_baseline_k16",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# STEP 03 SUPERVISION
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


# ============================================================
# METRICS
# ============================================================

def calculate_binary_metrics(
    y_true,
    scores,
):
    """
    Diagnostic binary classification metrics.

    IMPORTANT:
    These are classification metrics on STEP 03
    supervision points, NOT final localization metrics.
    """

    y_true = np.asarray(
        y_true
    )

    scores = np.asarray(
        scores
    )

    predictions = np.where(
        scores >= 0.0,
        1,
        -1,
    )

    y_binary = (
        y_true == 1
    ).astype(
        np.int32
    )

    return {
        "ap":
            float(
                average_precision_score(
                    y_binary,
                    scores,
                )
            ),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    predictions,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    predictions,
                    pos_label=1,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    predictions,
                    pos_label=1,
                    zero_division=0,
                )
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

    step_output = create_output_dir(
        STEP05_OUTPUT
    )

    run_output = (
        step_output
        / args.run_name
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
                f"STEP 05 output already exists:\n"
                f"{run_output}\n"
                f"Use --overwrite."
            )

    create_output_dir(
        run_output
    )

    sparse_map_dir = create_output_dir(
        run_output
        / "sparse_maps"
    )

    score_map_dir = create_output_dir(
        run_output
        / "class_score_maps"
    )

    classifier_dir = create_output_dir(
        run_output
        / "classifiers"
    )

    feature_dir = create_output_dir(
        run_output
        / "training_features"
    )

    visualization_dir = create_output_dir(
        run_output
        / "visualizations"
    )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    config_output_path = (
        run_output
        / "step05_run_config.json"
    )

    inference_summary_path = (
        run_output
        / "step05_sparse_inference_summary.csv"
    )

    classifier_summary_path = (
        run_output
        / "step05_classifier_summary.csv"
    )

    cv_results_path = (
        run_output
        / "step05_classifier_cv_results.csv"
    )

    weights_path = (
        run_output
        / "step05_classifier_weights.csv"
    )

    report_path = (
        run_output
        / "step05_training_report.txt"
    )

    # ========================================================
    # LOAD STEP 04 MODEL
    # ========================================================

    step04_run = (
        STEP04_OUTPUT
        / args.step04_run_name
    )

    dictionary_path = (
        step04_run
        / "step04_final_dictionary.npy"
    )

    lambda_path = (
        step04_run
        / "step04_lambda_diagnostics.json"
    )

    step04_config_path = (
        step04_run
        / "step04_run_config.json"
    )

    for required_path in [
        dictionary_path,
        lambda_path,
        step04_config_path,
    ]:

        if not required_path.exists():

            raise FileNotFoundError(
                f"Required STEP 04 file "
                f"not found:\n"
                f"{required_path}"
            )

    dictionary = np.load(
        dictionary_path,
        allow_pickle=False,
    ).astype(
        np.float32
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

    effective_lambda = float(
        lambda_diagnostics[
            "effective_lambda"
        ]
    )

    scale = float(
        step04_config[
            "scale"
        ]
    )

    rho = float(
        step04_config[
            "cbpdn_rho"
        ]
    )

    num_filters = int(
        dictionary.shape[-1]
    )

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    total_samples = len(
        manifest
    )

    train_count = int(
        (
            manifest["split"]
            == "train"
        ).sum()
    )

    val_count = int(
        (
            manifest["split"]
            == "val"
        ).sum()
    )

    # ========================================================
    # HEADER
    # ========================================================

    print("=" * 72)

    print(
        "STEP 05: CSC SPARSE FEATURES "
        "+ ONE-vs-ALL CLASSIFIERS"
    )

    print("=" * 72)

    print(
        f"STEP 04 model     : "
        f"{args.step04_run_name}"
    )

    print(
        f"Samples           : "
        f"{total_samples}"
    )

    print(
        f"Train             : "
        f"{train_count}"
    )

    print(
        f"Validation        : "
        f"{val_count}"
    )

    print(
        f"Dictionary shape  : "
        f"{dictionary.shape}"
    )

    print(
        f"Model scale       : "
        f"{scale}"
    )

    print(
        f"CSC lambda        : "
        f"{effective_lambda:.8e}"
    )

    print(
        f"CSC rho           : "
        f"{rho:.8f}"
    )

    print(
        f"Inference max iter: "
        f"{CSC_INFERENCE_MAX_ITER}"
    )

    print(
        f"RelStopTol        : "
        f"{CSC_INFERENCE_REL_STOP_TOL}"
    )

    print()

    # ========================================================
    # FEATURE CONTAINERS
    # ========================================================

    class_feature_parts = {
        1: [],
        2: [],
        3: [],
    }

    class_label_parts = {
        1: [],
        2: [],
        3: [],
    }

    class_group_parts = {
        1: [],
        2: [],
        3: [],
    }

    class_type_parts = {
        1: [],
        2: [],
        3: [],
    }

    inference_rows = []

    # ========================================================
    # PASS 1:
    # INFER SPARSE MAPS FOR ALL 30 IMAGES
    # ========================================================

    print(
        "PASS 1/3: Inferring CSC sparse maps..."
    )

    print()

    for index, row in (
        manifest.iterrows()
    ):

        split = str(
            row["split"]
        )

        sample_id = str(
            row["sample_id"]
        )

        sample = load_sample(
            split=split,
            sample_id=sample_id,
        )

        image = sample[
            "image"
        ]

        target = sample[
            "target"
        ]

        highpass = load_highpass(
            sample_id
        )

        model_signal = resize_highpass(
            highpass,
            scale=scale,
        )

        sparse_maps, diagnostics = (
            infer_sparse_maps(
                dictionary=
                    dictionary,

                signal=
                    model_signal,

                lmbda=
                    effective_lambda,

                rho=
                    rho,

                max_iter=
                    CSC_INFERENCE_MAX_ITER,

                rel_stop_tol=
                    CSC_INFERENCE_REL_STOP_TOL,
            )
        )

        # ----------------------------------------------------
        # SPARSITY STATISTICS
        # ----------------------------------------------------

        nonzero = (
            np.abs(
                sparse_maps
            )
            >
            CSC_SPARSE_NONZERO_TOL
        )

        nonzero_count = int(
            np.count_nonzero(
                nonzero
            )
        )

        total_coefficients = int(
            sparse_maps.size
        )

        density = (
            nonzero_count
            /
            total_coefficients
        )

        mean_abs = float(
            np.mean(
                np.abs(
                    sparse_maps
                )
            )
        )

        l1_norm = float(
            np.sum(
                np.abs(
                    sparse_maps
                )
            )
        )

        if l1_norm <= 0.0:

            raise RuntimeError(
                f"{sample_id}: "
                f"all-zero sparse representation."
            )

        # ----------------------------------------------------
        # SAVE SPARSE MAP
        # ----------------------------------------------------

        sparse_path = (
            sparse_map_dir
            / sample_filename(
                step=STEP_NUMBER,
                sample_id=sample_id,
                name="csc_sparse_maps",
                extension="npz",
            )
        )

        np.savez_compressed(
            sparse_path,
            sparse_maps=
                sparse_maps.astype(
                    np.float32
                ),

            scale=
                np.float32(
                    scale
                ),

            original_shape=
                np.asarray(
                    target.shape,
                    dtype=np.int32,
                ),

            model_shape=
                np.asarray(
                    sparse_maps.shape[:2],
                    dtype=np.int32,
                ),
        )

        # ----------------------------------------------------
        # SPARSE MAP VISUALIZATION
        # ----------------------------------------------------

        sparse_visualization_path = (
            visualization_dir
            / sample_filename(
                step=STEP_NUMBER,
                sample_id=sample_id,
                name="csc_sparse_maps",
                extension="png",
            )
        )

        save_sparse_map_grid(
            sparse_maps=
                sparse_maps,

            output_path=
                sparse_visualization_path,

            sample_id=
                sample_id,
        )

        # ----------------------------------------------------
        # EXTRACT TRAIN SUPERVISION FEATURES
        # ----------------------------------------------------

        if split == "train":

            supervision_path = (
                get_supervision_path(
                    sample_id
                )
            )

            if not supervision_path.exists():

                raise FileNotFoundError(
                    f"STEP 03 supervision "
                    f"not found:\n"
                    f"{supervision_path}"
                )

            with np.load(
                supervision_path,
                allow_pickle=False,
            ) as supervision:

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

                    coordinates = (
                        supervision[
                            f"{class_name}_coords"
                        ]
                    )

                    labels = (
                        supervision[
                            f"{class_name}_labels"
                        ]
                    ).astype(
                        np.int8
                    )

                    point_types = (
                        supervision[
                            f"{class_name}_point_types"
                        ]
                    ).astype(
                        np.int8
                    )

                    features = (
                        sample_sparse_features_bilinear(
                            sparse_maps=
                                sparse_maps,

                            coordinates=
                                coordinates,

                            original_shape=
                                target.shape,
                        )
                    )

                    if (
                        features.shape[0]
                        != len(labels)
                    ):

                        raise RuntimeError(
                            f"{sample_id} "
                            f"{class_name}: "
                            f"feature / label "
                            f"length mismatch."
                        )

                    class_feature_parts[
                        class_id
                    ].append(
                        features
                    )

                    class_label_parts[
                        class_id
                    ].append(
                        labels
                    )

                    class_type_parts[
                        class_id
                    ].append(
                        point_types
                    )

                    class_group_parts[
                        class_id
                    ].append(
                        np.full(
                            len(labels),
                            sample_id,
                            dtype="U64",
                        )
                    )

        # ----------------------------------------------------
        # INFERENCE SUMMARY
        # ----------------------------------------------------

        inference_rows.append(
            {
                "sample_id":
                    sample_id,

                "split":
                    split,

                "model_height":
                    sparse_maps.shape[0],

                "model_width":
                    sparse_maps.shape[1],

                "num_filters":
                    sparse_maps.shape[2],

                "iterations":
                    diagnostics[
                        "iterations"
                    ],

                "solve_time_seconds":
                    diagnostics[
                        "solve_time_seconds"
                    ],

                "objective":
                    diagnostics[
                        "objective"
                    ],

                "data_fidelity":
                    diagnostics[
                        "data_fidelity"
                    ],

                "reg_l1":
                    diagnostics[
                        "reg_l1"
                    ],

                "primal_residual":
                    diagnostics[
                        "primal_residual"
                    ],

                "dual_residual":
                    diagnostics[
                        "dual_residual"
                    ],

                "nonzero_coefficients":
                    nonzero_count,

                "total_coefficients":
                    total_coefficients,

                "coefficient_density":
                    density,

                "mean_abs_coefficient":
                    mean_abs,

                "l1_norm":
                    l1_norm,
            }
        )

        print(
            f"[{index + 1:02d}/"
            f"{total_samples:02d}] "
            f"{split:<5} | "
            f"{sample_id} | "
            f"density="
            f"{density:.4f} | "
            f"iter="
            f"{diagnostics['iterations']:>3} | "
            f"PASS"
        )

    inference_df = pd.DataFrame(
        inference_rows
    )

    inference_df.to_csv(
        inference_summary_path,
        index=False,
    )

    # ========================================================
    # PASS 2:
    # TRAIN ONE-vs-ALL CLASSIFIERS
    # ========================================================

    print()

    print(
        "PASS 2/3: Training One-vs-All "
        "logistic regression classifiers..."
    )

    print()

    classifiers = {}

    classifier_summary_rows = []

    cv_rows = []

    weight_rows = []

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

        X = np.vstack(
            class_feature_parts[
                class_id
            ]
        ).astype(
            np.float32
        )

        y = np.concatenate(
            class_label_parts[
                class_id
            ]
        ).astype(
            np.int8
        )

        groups = np.concatenate(
            class_group_parts[
                class_id
            ]
        )

        point_types = np.concatenate(
            class_type_parts[
                class_id
            ]
        ).astype(
            np.int8
        )

        positive_count = int(
            np.count_nonzero(
                y == 1
            )
        )

        negative_count = int(
            np.count_nonzero(
                y == -1
            )
        )

        if (
            positive_count == 0
            or negative_count == 0
        ):

            raise RuntimeError(
                f"{class_name}: "
                f"classifier requires both "
                f"positive and negative samples."
            )

        # ----------------------------------------------------
        # SAVE TRAINING FEATURES
        # ----------------------------------------------------

        feature_path = (
            feature_dir
            / (
                f"step05_{class_name}_"
                f"training_features.npz"
            )
        )

        np.savez_compressed(
            feature_path,
            X=X,
            y=y,
            groups=groups,
            point_types=point_types,
        )

        # ----------------------------------------------------
        # GROUPED CROSS-VALIDATION
        # ----------------------------------------------------
        #
        # Group by sample_id:
        # pixels from one image cannot leak into
        # both train and test fold.
        # ----------------------------------------------------

        unique_groups = np.unique(
            groups
        )

        folds = min(
            CSC_CLASSIFIER_CV_FOLDS,
            len(
                unique_groups
            ),
        )

        group_kfold = GroupKFold(
            n_splits=folds
        )

        class_cv_metrics = []

        for fold_index, (
            train_index,
            test_index,
        ) in enumerate(
            group_kfold.split(
                X,
                y,
                groups,
            ),
            start=1,
        ):

            fold_classifier = (
                build_logistic_classifier(
                    C=
                        CSC_CLASSIFIER_C,

                    max_iter=
                        CSC_CLASSIFIER_MAX_ITER,
                )
            )

            fold_classifier.fit(
                X[
                    train_index
                ],
                y[
                    train_index
                ],
            )

            fold_scores = (
                fold_classifier
                .decision_function(
                    X[
                        test_index
                    ]
                )
            )

            metrics = (
                calculate_binary_metrics(
                    y[
                        test_index
                    ],
                    fold_scores,
                )
            )

            class_cv_metrics.append(
                metrics
            )

            cv_rows.append(
                {
                    "class_id":
                        class_id,

                    "class_name":
                        class_name,

                    "fold":
                        fold_index,

                    "train_samples":
                        len(
                            train_index
                        ),

                    "test_samples":
                        len(
                            test_index
                        ),

                    "ap":
                        metrics["ap"],

                    "accuracy":
                        metrics[
                            "accuracy"
                        ],

                    "precision":
                        metrics[
                            "precision"
                        ],

                    "recall":
                        metrics[
                            "recall"
                        ],
                }
            )

        mean_cv_ap = float(
            np.mean(
                [
                    item["ap"]
                    for item
                    in class_cv_metrics
                ]
            )
        )

        mean_cv_accuracy = float(
            np.mean(
                [
                    item["accuracy"]
                    for item
                    in class_cv_metrics
                ]
            )
        )

        mean_cv_precision = float(
            np.mean(
                [
                    item["precision"]
                    for item
                    in class_cv_metrics
                ]
            )
        )

        mean_cv_recall = float(
            np.mean(
                [
                    item["recall"]
                    for item
                    in class_cv_metrics
                ]
            )
        )

        # ----------------------------------------------------
        # FINAL CLASSIFIER
        # ----------------------------------------------------

        classifier = (
            build_logistic_classifier(
                C=
                    CSC_CLASSIFIER_C,

                max_iter=
                    CSC_CLASSIFIER_MAX_ITER,
            )
        )

        classifier.fit(
            X,
            y,
        )

        training_scores = (
            classifier
            .decision_function(
                X
            )
        )

        training_metrics = (
            calculate_binary_metrics(
                y,
                training_scores,
            )
        )

        classifiers[
            class_id
        ] = classifier

        classifier_path = (
            classifier_dir
            / (
                f"step05_{class_name}_"
                f"classifier.joblib"
            )
        )

        joblib.dump(
            classifier,
            classifier_path,
        )

        # ----------------------------------------------------
        # SAVE CLASSIFIER WEIGHTS
        # ----------------------------------------------------

        scaler = (
            classifier
            .named_steps[
                "scaler"
            ]
        )

        logistic = (
            classifier
            .named_steps[
                "logistic"
            ]
        )

        weights = (
            logistic.coef_[0]
        )

        for filter_index in range(
            num_filters
        ):

            weight_rows.append(
                {
                    "class_id":
                        class_id,

                    "class_name":
                        class_name,

                    "filter_index":
                        filter_index + 1,

                    "weight":
                        float(
                            weights[
                                filter_index
                            ]
                        ),

                    "abs_weight":
                        float(
                            abs(
                                weights[
                                    filter_index
                                ]
                            )
                        ),

                    "feature_mean":
                        float(
                            scaler.mean_[
                                filter_index
                            ]
                        ),

                    "feature_scale":
                        float(
                            scaler.scale_[
                                filter_index
                            ]
                        ),
                }
            )

        classifier_summary_rows.append(
            {
                "class_id":
                    class_id,

                "class_name":
                    class_name,

                "total_samples":
                    len(y),

                "positive_samples":
                    positive_count,

                "negative_samples":
                    negative_count,

                "training_ap":
                    training_metrics[
                        "ap"
                    ],

                "training_accuracy":
                    training_metrics[
                        "accuracy"
                    ],

                "training_precision":
                    training_metrics[
                        "precision"
                    ],

                "training_recall":
                    training_metrics[
                        "recall"
                    ],

                "cv_folds":
                    folds,

                "cv_ap_mean":
                    mean_cv_ap,

                "cv_accuracy_mean":
                    mean_cv_accuracy,

                "cv_precision_mean":
                    mean_cv_precision,

                "cv_recall_mean":
                    mean_cv_recall,

                "classifier_intercept":
                    float(
                        logistic.intercept_[0]
                    ),
            }
        )

        print(
            f"{class_name.title():<12} | "
            f"samples={len(y):>6} | "
            f"+={positive_count:>6} | "
            f"-={negative_count:>6} | "
            f"Train AP="
            f"{training_metrics['ap']:.4f} | "
            f"CV AP="
            f"{mean_cv_ap:.4f}"
        )

    classifier_summary_df = (
        pd.DataFrame(
            classifier_summary_rows
        )
    )

    classifier_summary_df.to_csv(
        classifier_summary_path,
        index=False,
    )

    cv_df = pd.DataFrame(
        cv_rows
    )

    cv_df.to_csv(
        cv_results_path,
        index=False,
    )

    weights_df = pd.DataFrame(
        weight_rows
    )

    weights_df.to_csv(
        weights_path,
        index=False,
    )

    # ========================================================
    # PASS 3:
    # GENERATE RAW CLASS SCORE MAPS
    # ========================================================

    print()

    print(
        "PASS 3/3: Generating raw "
        "class score maps..."
    )

    print()

    for index, row in (
        manifest.iterrows()
    ):

        split = str(
            row["split"]
        )

        sample_id = str(
            row["sample_id"]
        )

        sparse_path = (
            sparse_map_dir
            / sample_filename(
                step=STEP_NUMBER,
                sample_id=sample_id,
                name="csc_sparse_maps",
                extension="npz",
            )
        )

        with np.load(
            sparse_path,
            allow_pickle=False,
        ) as sparse_file:

            sparse_maps = (
                sparse_file[
                    "sparse_maps"
                ]
                .astype(
                    np.float32
                )
            )

        person_score = (
            classifier_score_map(
                classifiers[1],
                sparse_maps,
            )
        )

        car_score = (
            classifier_score_map(
                classifiers[2],
                sparse_maps,
            )
        )

        motorcycle_score = (
            classifier_score_map(
                classifiers[3],
                sparse_maps,
            )
        )

        score_path = (
            score_map_dir
            / sample_filename(
                step=STEP_NUMBER,
                sample_id=sample_id,
                name="raw_class_scores",
                extension="npz",
            )
        )

        np.savez_compressed(
            score_path,

            person_score=
                person_score,

            car_score=
                car_score,

            motorcycle_score=
                motorcycle_score,

            class_ids=
                np.asarray(
                    [
                        1,
                        2,
                        3,
                    ],
                    dtype=np.int8,
                ),
        )

        sample = load_sample(
            split=split,
            sample_id=sample_id,
        )

        score_visualization_path = (
            visualization_dir
            / sample_filename(
                step=STEP_NUMBER,
                sample_id=sample_id,
                name="raw_class_scores",
                extension="png",
            )
        )

        save_class_score_visualization(
            image=
                sample["image"],

            person_score=
                person_score,

            car_score=
                car_score,

            motorcycle_score=
                motorcycle_score,

            output_path=
                score_visualization_path,

            sample_id=
                sample_id,
        )

        print(
            f"[{index + 1:02d}/"
            f"{total_samples:02d}] "
            f"{split:<5} | "
            f"{sample_id} | PASS"
        )

    # ========================================================
    # RUN CONFIG
    # ========================================================

    run_config = {
        "step":
            STEP_NUMBER,

        "source_step04_run":
            args.step04_run_name,

        "dictionary_shape":
            list(
                dictionary.shape
            ),

        "num_filters":
            num_filters,

        "scale":
            scale,

        "effective_lambda":
            effective_lambda,

        "cbpdn_rho":
            rho,

        "inference_max_iterations":
            CSC_INFERENCE_MAX_ITER,

        "inference_rel_stop_tol":
            CSC_INFERENCE_REL_STOP_TOL,

        "classifier":
            "StandardScaler + LogisticRegression",

        "classifier_C":
            CSC_CLASSIFIER_C,

        "classifier_max_iterations":
            CSC_CLASSIFIER_MAX_ITER,

        "classifier_cv_folds":
            CSC_CLASSIFIER_CV_FOLDS,

        "classifier_cv_grouping":
            "sample_id",

        "classes": {
            "1":
                "person",

            "2":
                "car",

            "3":
                "motorcycle",
        },
    }

    config_output_path.write_text(
        json.dumps(
            run_config,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # REPORT
    # ========================================================

    mean_density = float(
        inference_df[
            "coefficient_density"
        ].mean()
    )

    mean_inference_time = float(
        inference_df[
            "solve_time_seconds"
        ].mean()
    )

    report_lines = [
        "STEP 05: CSC SPARSE FEATURES + ONE-vs-ALL CLASSIFIERS",
        "=" * 72,
        "",
        "INPUT",
        "-" * 72,
        f"STEP 04 run              : {args.step04_run_name}",
        f"Dictionary shape         : {dictionary.shape}",
        f"Training images          : {train_count}",
        f"Validation images        : {val_count}",
        f"Model scale              : {scale}",
        f"Effective lambda         : {effective_lambda:.8e}",
        "",
        "SPARSE INFERENCE",
        "-" * 72,
        f"Total images inferred    : {total_samples}",
        f"Maximum ADMM iterations  : {CSC_INFERENCE_MAX_ITER}",
        f"Relative stop tolerance  : {CSC_INFERENCE_REL_STOP_TOL}",
        f"Mean coefficient density : {mean_density:.8f}",
        f"Mean solve time / image  : {mean_inference_time:.3f} s",
        "",
        "CLASSIFIERS",
        "-" * 72,
        "Method                    : One-vs-All Logistic Regression",
        "Feature dimension         : 16 CSC coefficients",
        f"Logistic C                : {CSC_CLASSIFIER_C}",
        f"Grouped CV folds          : {CSC_CLASSIFIER_CV_FOLDS}",
        "CV grouping               : image / sample_id",
        "",
    ]

    for _, result in (
        classifier_summary_df.iterrows()
    ):

        report_lines.extend(
            [
                result[
                    "class_name"
                ].upper(),

                (
                    f"  Samples            : "
                    f"{int(result['total_samples']):,}"
                ),

                (
                    f"  Positive           : "
                    f"{int(result['positive_samples']):,}"
                ),

                (
                    f"  Negative           : "
                    f"{int(result['negative_samples']):,}"
                ),

                (
                    f"  Training AP        : "
                    f"{result['training_ap']:.6f}"
                ),

                (
                    f"  5-fold CV AP       : "
                    f"{result['cv_ap_mean']:.6f}"
                ),

                (
                    f"  CV Precision       : "
                    f"{result['cv_precision_mean']:.6f}"
                ),

                (
                    f"  CV Recall          : "
                    f"{result['cv_recall_mean']:.6f}"
                ),

                "",
            ]
        )

    report_lines.extend(
        [
            "IMPORTANT",
            "-" * 72,
            (
                "Training and cross-validation AP values in STEP 05 "
                "are classifier diagnostics on STEP 03 supervision "
                "points. They are NOT the final localization AP/mAP."
            ),
            (
                "The six validation images were never used to fit "
                "the logistic regression classifiers."
            ),
            (
                "Final localization metrics will be measured from "
                "full prediction maps against validation ground truth "
                "after the CSC and SCSC pipelines are both available."
            ),
            "",
            "FINAL STATUS",
            "-" * 72,
            "PASSED",
        ]
    )

    report_path.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print()

    print("=" * 72)
    print("STEP 05 RESULT")
    print("=" * 72)

    print(
        f"Sparse maps        : "
        f"{total_samples} images"
    )

    print(
        f"Mean density       : "
        f"{mean_density:.6f}"
    )

    print(
        f"Mean inference time: "
        f"{mean_inference_time:.2f} s/image"
    )

    print()

    for _, result in (
        classifier_summary_df.iterrows()
    ):

        print(
            f"{result['class_name'].title():<12} | "
            f"Train AP="
            f"{result['training_ap']:.4f} | "
            f"CV AP="
            f"{result['cv_ap_mean']:.4f} | "
            f"CV P="
            f"{result['cv_precision_mean']:.4f} | "
            f"CV R="
            f"{result['cv_recall_mean']:.4f}"
        )

    print()

    print(
        "FINAL STATUS: PASSED"
    )

    print()

    print(
        "Outputs saved to:"
    )

    print(
        run_output
    )

    print("=" * 72)


if __name__ == "__main__":

    main()
    
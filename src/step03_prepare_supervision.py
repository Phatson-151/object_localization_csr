import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.colors import (
    ListedColormap,
)

import numpy as np
import pandas as pd

from src.config import (
    CLASS_NAMES,
    MANIFEST_PATH,
    STEP02_OUTPUT,
    STEP03_OUTPUT,
    SUPERVISION_RANDOM_SEED,
    SUPERVISION_POINTS_PER_INSTANCE,
    SUPERVISION_BOUNDARY_FRACTION,
    SUPERVISION_NEGATIVE_RATIO,
    SUPERVISION_OTHER_CLASS_NEGATIVE_FRACTION,
)

from src.data.dataset_utils import (
    load_sample,
)

from src.supervision.supervision_sampling import (
    get_instance_ids,
    sample_instance_positive_points,
    sample_balanced_negative_points,
    validate_class_supervision,
    POINT_INTERIOR_POSITIVE,
    POINT_BOUNDARY_POSITIVE,
    POINT_BACKGROUND_NEGATIVE,
    POINT_OTHER_CLASS_NEGATIVE,
)

from src.utils.output_utils import (
    create_output_dir,
    step_filename,
    sample_filename,
)


STEP_NUMBER = 3


def add_error(
    errors,
    sample_id,
    error_type,
    message,
):
    errors.append(
        {
            "sample_id":
                sample_id,

            "error_type":
                error_type,

            "message":
                message,
        }
    )


def save_supervision_visualization(
    image,
    target,
    class_results,
    output_path,
    sample_id,
):
    """
    Visualize ground truth and sampled
    supervision points.
    """

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(18, 10),
    )

    # --------------------------------------------------------
    # TARGET MASK
    # --------------------------------------------------------

    target_cmap = ListedColormap(
        [
            "black",
            "deepskyblue",
            "hotpink",
            "gold",
        ]
    )

    axes[0, 0].imshow(
        target,
        cmap=target_cmap,
        vmin=0,
        vmax=3,
    )

    axes[0, 0].set_title(
        "Ground Truth Target Mask"
    )

    axes[0, 0].axis(
        "off"
    )

    # --------------------------------------------------------
    # CLASS SUPERVISION
    # --------------------------------------------------------

    class_axes = {
        1: axes[0, 1],
        2: axes[1, 0],
        3: axes[1, 1],
    }

    style = {
        POINT_INTERIOR_POSITIVE: {
            "label":
                "Interior Positive",

            "color":
                "lime",

            "marker":
                "o",

            "size":
                6,
        },

        POINT_BOUNDARY_POSITIVE: {
            "label":
                "Boundary Positive",

            "color":
                "yellow",

            "marker":
                "o",

            "size":
                6,
        },

        POINT_BACKGROUND_NEGATIVE: {
            "label":
                "Background Negative",

            "color":
                "red",

            "marker":
                "x",

            "size":
                5,
        },

        POINT_OTHER_CLASS_NEGATIVE: {
            "label":
                "Other-Class Negative",

            "color":
                "magenta",

            "marker":
                "x",

            "size":
                5,
        },
    }

    for class_id in (
        1,
        2,
        3,
    ):

        axis = class_axes[
            class_id
        ]

        axis.imshow(
            image
        )

        result = class_results[
            class_id
        ]

        coordinates = result[
            "coordinates"
        ]

        point_types = result[
            "point_types"
        ]

        for point_type, settings in (
            style.items()
        ):

            selected = (
                point_types
                == point_type
            )

            points = (
                coordinates[
                    selected
                ]
            )

            if len(points) == 0:
                continue

            axis.scatter(
                points[:, 1],
                points[:, 0],
                s=settings[
                    "size"
                ],
                c=settings[
                    "color"
                ],
                marker=settings[
                    "marker"
                ],
                alpha=0.75,
                linewidths=0.5,
                label=settings[
                    "label"
                ],
            )

        positive_count = int(
            np.count_nonzero(
                result["labels"]
                == 1
            )
        )

        negative_count = int(
            np.count_nonzero(
                result["labels"]
                == -1
            )
        )

        axis.set_title(
            f"{CLASS_NAMES[class_id].title()} "
            f"vs All\n"
            f"+ {positive_count:,} | "
            f"- {negative_count:,}"
        )

        axis.axis(
            "off"
        )

        axis.legend(
            loc="lower right",
            fontsize=7,
            markerscale=1.5,
        )

    figure.suptitle(
        f"STEP 03 Supervision — "
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


def main():

    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

    output_dir = (
        create_output_dir(
            STEP03_OUTPUT
        )
    )

    data_dir = (
        create_output_dir(
            output_dir
            / "data"
        )
    )

    visualization_dir = (
        create_output_dir(
            output_dir
            / "visualizations"
        )
    )

    report_path = (
        output_dir
        / step_filename(
            STEP_NUMBER,
            "supervision_report",
            "txt",
        )
    )

    sampling_summary_path = (
        output_dir
        / step_filename(
            STEP_NUMBER,
            "sampling_summary",
            "csv",
        )
    )

    class_summary_path = (
        output_dir
        / step_filename(
            STEP_NUMBER,
            "class_summary",
            "csv",
        )
    )

    instance_statistics_path = (
        output_dir
        / step_filename(
            STEP_NUMBER,
            "instance_statistics",
            "csv",
        )
    )

    errors_path = (
        output_dir
        / step_filename(
            STEP_NUMBER,
            "supervision_errors",
            "csv",
        )
    )

    # ========================================================
    # LOAD MANIFEST
    # ========================================================

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    # IMPORTANT:
    # Only TRAIN images generate supervision.

    train_manifest = (
        manifest[
            manifest["split"]
            == "train"
        ]
        .reset_index(
            drop=True
        )
    )

    total_train = len(
        train_manifest
    )

    errors = []

    sampling_rows = []
    instance_rows = []

    print("=" * 72)
    print(
        "STEP 03: SUPERVISION PREPARATION"
    )
    print("=" * 72)

    print(
        f"Training images             : "
        f"{total_train}"
    )

    print(
        f"Max points / instance       : "
        f"{SUPERVISION_POINTS_PER_INSTANCE}"
    )

    print(
        f"Interior fraction           : "
        f"{1.0 - SUPERVISION_BOUNDARY_FRACTION:.2f}"
    )

    print(
        f"Boundary fraction           : "
        f"{SUPERVISION_BOUNDARY_FRACTION:.2f}"
    )

    print(
        f"Negative / Positive ratio   : "
        f"{SUPERVISION_NEGATIVE_RATIO:.2f}"
    )

    print(
        f"Other-class negative ratio  : "
        f"{SUPERVISION_OTHER_CLASS_NEGATIVE_FRACTION:.2f}"
    )

    print(
        f"Random seed                 : "
        f"{SUPERVISION_RANDOM_SEED}"
    )

    print()

    # ========================================================
    # PROCESS TRAINING IMAGES
    # ========================================================

    for sample_index, row in (
        train_manifest.iterrows()
    ):

        sample_id = str(
            row["sample_id"]
        )

        split = "train"

        error_start = len(
            errors
        )

        # ----------------------------------------------------
        # VERIFY STEP 02 INPUT EXISTS
        # ----------------------------------------------------

        highpass_path = (
            STEP02_OUTPUT
            / "arrays"
            / sample_filename(
                step=2,
                sample_id=sample_id,
                name="highpass",
                extension="npy",
            )
        )

        if not highpass_path.exists():

            add_error(
                errors,
                sample_id,
                "missing_step02_output",
                str(
                    highpass_path
                ),
            )

            print(
                f"[{sample_index + 1:02d}/"
                f"{total_train:02d}] "
                f"{sample_id} | FAIL "
                f"(missing STEP 02)"
            )

            continue

        try:

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

            instance_mask = sample[
                "instance"
            ]

            class_results = {}

            # ------------------------------------------------
            # EACH ONE-vs-ALL CLASS
            # ------------------------------------------------

            for class_id in (
                1,
                2,
                3,
            ):

                # Deterministic class-specific
                # random generator.

                rng = np.random.default_rng(
                    SUPERVISION_RANDOM_SEED
                    +
                    sample_index
                    * 100
                    +
                    class_id
                )

                instance_ids = (
                    get_instance_ids(
                        instance_mask,
                        class_id,
                    )
                )

                positive_coordinates = []
                positive_types = []
                positive_instance_ids = []

                # --------------------------------------------
                # POSITIVE SAMPLING PER INSTANCE
                # --------------------------------------------

                for instance_id in (
                    instance_ids
                ):

                    instance_binary = (
                        (
                            instance_mask
                            == instance_id
                        )
                        &
                        (
                            target
                            == class_id
                        )
                    )

                    sampled = (
                        sample_instance_positive_points(
                            instance_binary_mask=
                                instance_binary,

                            rng=rng,

                            max_points=
                                SUPERVISION_POINTS_PER_INSTANCE,

                            boundary_fraction=
                                SUPERVISION_BOUNDARY_FRACTION,
                        )
                    )

                    coordinates = (
                        sampled[
                            "coordinates"
                        ]
                    )

                    types = (
                        sampled[
                            "types"
                        ]
                    )

                    positive_coordinates.append(
                        coordinates
                    )

                    positive_types.append(
                        types
                    )

                    positive_instance_ids.append(
                        np.full(
                            len(
                                coordinates
                            ),
                            instance_id,
                            dtype=np.int32,
                        )
                    )

                    instance_rows.append(
                        {
                            "sample_id":
                                sample_id,

                            "class_id":
                                class_id,

                            "class_name":
                                CLASS_NAMES[
                                    class_id
                                ],

                            "instance_id":
                                instance_id,

                            "object_pixels":
                                sampled[
                                    "object_pixels"
                                ],

                            "selected_total":
                                sampled[
                                    "selected_total"
                                ],

                            "interior_positive":
                                sampled[
                                    "interior_selected"
                                ],

                            "boundary_positive":
                                sampled[
                                    "boundary_selected"
                                ],

                            "interior_region_pixels":
                                sampled[
                                    "interior_region_pixels"
                                ],

                            "boundary_region_pixels":
                                sampled[
                                    "boundary_region_pixels"
                                ],
                        }
                    )

                # --------------------------------------------
                # COMBINE POSITIVES
                # --------------------------------------------

                if positive_coordinates:

                    positive_coordinates = (
                        np.vstack(
                            positive_coordinates
                        ).astype(
                            np.int32
                        )
                    )

                    positive_types = (
                        np.concatenate(
                            positive_types
                        ).astype(
                            np.int8
                        )
                    )

                    positive_instance_ids = (
                        np.concatenate(
                            positive_instance_ids
                        ).astype(
                            np.int32
                        )
                    )

                else:

                    positive_coordinates = (
                        np.empty(
                            (0, 2),
                            dtype=np.int32,
                        )
                    )

                    positive_types = (
                        np.empty(
                            (0,),
                            dtype=np.int8,
                        )
                    )

                    positive_instance_ids = (
                        np.empty(
                            (0,),
                            dtype=np.int32,
                        )
                    )

                n_positive = len(
                    positive_coordinates
                )

                if n_positive == 0:

                    raise ValueError(
                        f"No positive points "
                        f"for class "
                        f"{CLASS_NAMES[class_id]} "
                        f"in {sample_id}"
                    )

                # --------------------------------------------
                # BALANCED NEGATIVES
                # --------------------------------------------

                n_negative = int(
                    round(
                        n_positive
                        * SUPERVISION_NEGATIVE_RATIO
                    )
                )

                negative = (
                    sample_balanced_negative_points(
                        target_mask=
                            target,

                        target_class_id=
                            class_id,

                        n_negative=
                            n_negative,

                        other_class_fraction=
                            SUPERVISION_OTHER_CLASS_NEGATIVE_FRACTION,

                        rng=
                            rng,
                    )
                )

                negative_coordinates = (
                    negative[
                        "coordinates"
                    ]
                )

                negative_types = (
                    negative[
                        "types"
                    ]
                )

                # --------------------------------------------
                # FINAL CLASS-SPECIFIC SET
                # --------------------------------------------

                all_coordinates = (
                    np.vstack(
                        [
                            positive_coordinates,
                            negative_coordinates,
                        ]
                    )
                    .astype(
                        np.int32
                    )
                )

                labels = np.concatenate(
                    [
                        np.ones(
                            n_positive,
                            dtype=np.int8,
                        ),

                        -np.ones(
                            len(
                                negative_coordinates
                            ),
                            dtype=np.int8,
                        ),
                    ]
                )

                point_types = np.concatenate(
                    [
                        positive_types,
                        negative_types,
                    ]
                ).astype(
                    np.int8
                )

                all_instance_ids = (
                    np.concatenate(
                        [
                            positive_instance_ids,

                            np.full(
                                len(
                                    negative_coordinates
                                ),
                                -1,
                                dtype=np.int32,
                            ),
                        ]
                    )
                )

                # --------------------------------------------
                # VALIDATE
                # --------------------------------------------

                validation_errors = (
                    validate_class_supervision(
                        coordinates=
                            all_coordinates,

                        labels=
                            labels,

                        point_types=
                            point_types,

                        target_mask=
                            target,

                        target_class_id=
                            class_id,
                    )
                )

                for message in (
                    validation_errors
                ):

                    add_error(
                        errors,
                        sample_id,
                        "supervision_validation",
                        (
                            f"{CLASS_NAMES[class_id]}: "
                            f"{message}"
                        ),
                    )

                interior_count = int(
                    np.count_nonzero(
                        positive_types
                        == POINT_INTERIOR_POSITIVE
                    )
                )

                boundary_count = int(
                    np.count_nonzero(
                        positive_types
                        == POINT_BOUNDARY_POSITIVE
                    )
                )

                background_negative = int(
                    np.count_nonzero(
                        negative_types
                        == POINT_BACKGROUND_NEGATIVE
                    )
                )

                other_negative = int(
                    np.count_nonzero(
                        negative_types
                        == POINT_OTHER_CLASS_NEGATIVE
                    )
                )

                class_results[
                    class_id
                ] = {
                    "coordinates":
                        all_coordinates,

                    "labels":
                        labels,

                    "point_types":
                        point_types,

                    "instance_ids":
                        all_instance_ids,
                }

                sampling_rows.append(
                    {
                        "sample_id":
                            sample_id,

                        "class_id":
                            class_id,

                        "class_name":
                            CLASS_NAMES[
                                class_id
                            ],

                        "instances":
                            len(
                                instance_ids
                            ),

                        "positive_total":
                            n_positive,

                        "interior_positive":
                            interior_count,

                        "boundary_positive":
                            boundary_count,

                        "boundary_ratio":
                            (
                                boundary_count
                                / n_positive
                            ),

                        "negative_total":
                            len(
                                negative_coordinates
                            ),

                        "background_negative":
                            background_negative,

                        "other_class_negative":
                            other_negative,

                        "total_supervision_points":
                            len(
                                all_coordinates
                            ),
                    }
                )

            # ------------------------------------------------
            # SAVE NPZ
            # ------------------------------------------------

            npz_name = (
                sample_filename(
                    step=STEP_NUMBER,
                    sample_id=sample_id,
                    name="supervision",
                    extension="npz",
                )
            )

            npz_path = (
                data_dir
                / npz_name
            )

            payload = {}

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

                result = (
                    class_results[
                        class_id
                    ]
                )

                payload[
                    f"{class_name}_coords"
                ] = result[
                    "coordinates"
                ]

                payload[
                    f"{class_name}_labels"
                ] = result[
                    "labels"
                ]

                payload[
                    f"{class_name}_point_types"
                ] = result[
                    "point_types"
                ]

                payload[
                    f"{class_name}_instance_ids"
                ] = result[
                    "instance_ids"
                ]

            np.savez_compressed(
                npz_path,
                **payload,
            )

            # ------------------------------------------------
            # SAVE VISUALIZATION
            # ------------------------------------------------

            visualization_name = (
                sample_filename(
                    step=STEP_NUMBER,
                    sample_id=sample_id,
                    name="supervision",
                    extension="png",
                )
            )

            visualization_path = (
                visualization_dir
                / visualization_name
            )

            save_supervision_visualization(
                image=image,
                target=target,
                class_results=
                    class_results,
                output_path=
                    visualization_path,
                sample_id=
                    sample_id,
            )

            sample_errors = (
                len(errors)
                - error_start
            )

            status = (
                "PASS"
                if sample_errors == 0
                else "FAIL"
            )

            print(
                f"[{sample_index + 1:02d}/"
                f"{total_train:02d}] "
                f"{sample_id} | "
                f"{status}"
            )

        except Exception as exc:

            add_error(
                errors,
                sample_id,
                "processing_error",
                repr(exc),
            )

            print(
                f"[{sample_index + 1:02d}/"
                f"{total_train:02d}] "
                f"{sample_id} | FAIL"
            )

            print(
                f"    {exc}"
            )

    # ========================================================
    # SAVE CSV RESULTS
    # ========================================================

    sampling_df = pd.DataFrame(
        sampling_rows
    )

    sampling_df.to_csv(
        sampling_summary_path,
        index=False,
    )

    instance_df = pd.DataFrame(
        instance_rows
    )

    instance_df.to_csv(
        instance_statistics_path,
        index=False,
    )

    errors_df = pd.DataFrame(
        errors,
        columns=[
            "sample_id",
            "error_type",
            "message",
        ],
    )

    errors_df.to_csv(
        errors_path,
        index=False,
    )

    # ========================================================
    # CLASS SUMMARY
    # ========================================================

    class_summary_rows = []

    for class_id in (
        1,
        2,
        3,
    ):

        class_df = (
            sampling_df[
                sampling_df[
                    "class_id"
                ]
                == class_id
            ]
        )

        positive_total = int(
            class_df[
                "positive_total"
            ].sum()
        )

        interior_total = int(
            class_df[
                "interior_positive"
            ].sum()
        )

        boundary_total = int(
            class_df[
                "boundary_positive"
            ].sum()
        )

        negative_total = int(
            class_df[
                "negative_total"
            ].sum()
        )

        background_negative = int(
            class_df[
                "background_negative"
            ].sum()
        )

        other_negative = int(
            class_df[
                "other_class_negative"
            ].sum()
        )

        class_instance_df = (
            instance_df[
                instance_df[
                    "class_id"
                ]
                == class_id
            ]
        )

        class_summary_rows.append(
            {
                "class_id":
                    class_id,

                "class_name":
                    CLASS_NAMES[
                        class_id
                    ],

                "instances":
                    len(
                        class_instance_df
                    ),

                "positive_total":
                    positive_total,

                "interior_positive":
                    interior_total,

                "boundary_positive":
                    boundary_total,

                "boundary_ratio":
                    (
                        boundary_total
                        / positive_total
                        if positive_total
                        else np.nan
                    ),

                "negative_total":
                    negative_total,

                "background_negative":
                    background_negative,

                "other_class_negative":
                    other_negative,

                "total_classifier_samples":
                    (
                        positive_total
                        +
                        negative_total
                    ),
            }
        )

    class_summary_df = pd.DataFrame(
        class_summary_rows
    )

    class_summary_df.to_csv(
        class_summary_path,
        index=False,
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    final_status = (
        "PASSED"
        if len(errors) == 0
        else "FAILED"
    )

    # ========================================================
    # TEXT REPORT
    # ========================================================

    lines = []

    lines.append(
        "STEP 03: SUPERVISION PREPARATION"
    )

    lines.append(
        "=" * 72
    )

    lines.append("")
    lines.append(
        f"Training images              : "
        f"{total_train}"
    )

    lines.append(
        f"Validation images used       : 0"
    )

    lines.append("")

    lines.append(
        "SUPERVISION STRATEGY"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        f"Maximum points / instance    : "
        f"{SUPERVISION_POINTS_PER_INSTANCE}"
    )

    lines.append(
        f"Interior target fraction     : "
        f"{1.0 - SUPERVISION_BOUNDARY_FRACTION:.2f}"
    )

    lines.append(
        f"Boundary target fraction     : "
        f"{SUPERVISION_BOUNDARY_FRACTION:.2f}"
    )

    lines.append(
        f"Negative / positive ratio    : "
        f"{SUPERVISION_NEGATIVE_RATIO:.2f}"
    )

    lines.append(
        f"Other-class negative fraction: "
        f"{SUPERVISION_OTHER_CLASS_NEGATIVE_FRACTION:.2f}"
    )

    lines.append(
        f"Random seed                  : "
        f"{SUPERVISION_RANDOM_SEED}"
    )

    lines.append("")

    lines.append(
        "CLASS SUMMARY"
    )

    lines.append(
        "-" * 72
    )

    for _, row in (
        class_summary_df.iterrows()
    ):

        lines.append(
            f"{row['class_name'].upper()}"
        )

        lines.append(
            f"  Instances            : "
            f"{int(row['instances']):,}"
        )

        lines.append(
            f"  Positive             : "
            f"{int(row['positive_total']):,}"
        )

        lines.append(
            f"  Interior positive    : "
            f"{int(row['interior_positive']):,}"
        )

        lines.append(
            f"  Boundary positive    : "
            f"{int(row['boundary_positive']):,}"
        )

        lines.append(
            f"  Boundary ratio       : "
            f"{row['boundary_ratio']:.4f}"
        )

        lines.append(
            f"  Negative             : "
            f"{int(row['negative_total']):,}"
        )

        lines.append(
            f"  Background negative  : "
            f"{int(row['background_negative']):,}"
        )

        lines.append(
            f"  Other-class negative : "
            f"{int(row['other_class_negative']):,}"
        )

        lines.append("")

    lines.append(
        "VALIDATION"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        f"Errors                        : "
        f"{len(errors)}"
    )

    lines.append(
        f"FINAL STATUS                  : "
        f"{final_status}"
    )

    report = "\n".join(
        lines
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    # ========================================================
    # TERMINAL RESULT
    # ========================================================

    print()
    print("=" * 72)
    print("CLASS SUMMARY")
    print("=" * 72)

    for _, row in (
        class_summary_df.iterrows()
    ):

        print(
            f"{row['class_name'].title():<12} | "
            f"instances="
            f"{int(row['instances']):>3} | "
            f"positive="
            f"{int(row['positive_total']):>6} | "
            f"interior="
            f"{int(row['interior_positive']):>6} | "
            f"boundary="
            f"{int(row['boundary_positive']):>6} | "
            f"negative="
            f"{int(row['negative_total']):>6}"
        )

    print()

    print(
        f"Errors       : "
        f"{len(errors)}"
    )

    print(
        f"FINAL STATUS : "
        f"{final_status}"
    )

    print()

    print(
        "Outputs saved to:"
    )

    print(
        output_dir
    )

    print()

    print("Created:")

    print(
        f"  {report_path.name}"
    )

    print(
        f"  {sampling_summary_path.name}"
    )

    print(
        f"  {class_summary_path.name}"
    )

    print(
        f"  {instance_statistics_path.name}"
    )

    print(
        f"  {errors_path.name}"
    )

    print(
        "  data/"
    )

    print(
        "  visualizations/"
    )

    print("=" * 72)

    if final_status == "FAILED":
        sys.exit(1)


if __name__ == "__main__":
    main()
    